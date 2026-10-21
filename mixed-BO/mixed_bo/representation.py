"""Full-design physics-shaped VAE; high-fidelity responses never enter pretraining."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import CATEGORICAL, CONTINUOUS, DESIGN, LOWER, UPPER, fingerprint, check_design, initial_design


class ShapeVAE(nn.Module):
    def __init__(self, design_dim, geometry_dim, physics_dim, latent_dim=4, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(design_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, design_dim))
        # Auxiliary predictors act on z, independently of design reconstruction.
        self.geometry_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                                           nn.Linear(hidden_dim, geometry_dim))
        self.physics_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                                          nn.Linear(hidden_dim, physics_dim))

    def encode(self, design):
        hidden = self.encoder(design)
        return self.mu(hidden), self.logvar(hidden).clamp(-10., 6.)

    def decode(self, latent):
        return self.decoder(latent), self.geometry_head(latent), self.physics_head(latent)

    def forward(self, design):
        mu, logvar = self.encode(design)
        latent = mu + torch.exp(.5*logvar)*torch.randn_like(mu)
        return (*self.decode(latent), mu, logvar)


def shape_losses(design, geometry, physics, outputs, beta, geometry_weight, physics_weight):
    reconstructed, g_hat, p_hat, mu, logvar = outputs
    # Mean over designs of ||x - x_hat||_2^2, NOT cross entropy or mean over coordinates.
    rec = (reconstructed-design).square().sum(dim=1).mean()
    kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(dim=1).mean()
    g_loss, p_loss = F.mse_loss(g_hat, geometry), F.mse_loss(p_hat, physics)
    total = rec + beta*kl + geometry_weight*g_loss + physics_weight*p_loss
    return total, {'loss': total, 'reconstruction_l2': rec, 'kl': kl,
                   'geometry_mse': g_loss, 'physics_mse': p_loss}


def standardized(values):
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    # Very small physical units must retain their information after scaling.
    scale = np.where(scale > 0., scale, 1.)
    return (values-center)/scale, center, scale


def descriptor_arrays(block):
    arrays = []
    for family in ['geometry', 'physics']:
        try:
            raw = [block['descriptors'][t][family] for t in block['types']]
            if any(v is None for row in raw for v in row):
                raise ValueError('null descriptor')
            array = np.asarray(raw, dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Fill the catalog descriptors, or supply both g__ and p__ columns in --pretrain-csv') from exc
        if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
            raise ValueError('Catalog descriptor arrays must be nonempty and finite')
        arrays.append(array)
    return arrays


def design_matrix(frame, catalog):
    """Fixed numeric representation of x: three [0,1] values and six one-hot entries."""
    frame = check_design(frame, catalog)
    pieces = [(frame[CONTINUOUS].to_numpy(float)-LOWER)/(UPPER-LOWER)]
    for name in CATEGORICAL:
        types = catalog['variables'][name]['types']
        indices = np.array([types.index(v) for v in frame[name]], dtype=int)
        pieces.append(np.eye(len(types))[indices])
    return np.concatenate(pieces, axis=1)


def prepare_pretraining(catalog, designs=None, per_combination=64, seed=42):
    """Read only x/g/p; neither row filtering nor losses access expensive y/status."""
    source = initial_design(catalog, per_combination, seed) if designs is None else designs
    clean = check_design(source, catalog)
    columns = {family: [c for c in clean.columns if str(c).startswith(prefix)]
               for family, prefix in [('geometry', 'g__'), ('physics', 'p__')]}
    if any(columns.values()):
        if not all(columns.values()):
            raise ValueError('Pretraining CSV must supply both g__... and p__... auxiliary columns')
        mode = 'per_design_columns'
    else:
        mode = 'catalog_lookup'
        for name in CATEGORICAL:
            block = catalog['variables'][name]
            indices = [block['types'].index(t) for t in clean[name]]
            for family, prefix, values in zip(['geometry', 'physics'], ['g__', 'p__'], descriptor_arrays(block)):
                for j, field in enumerate(block[family]):
                    column = f'{prefix}{name}__{field["name"]}'
                    if column in columns[family]:
                        raise ValueError('Catalog auxiliary column names must be unique')
                    columns[family].append(column)
                    clean[column] = values[indices, j]
    # Explicit projection prevents y, status, predictions, or other metadata from
    # influencing filtering, descriptor normalization, gradients, or saved targets.
    clean = clean[DESIGN+columns['geometry']+columns['physics']].copy()
    for family in columns:
        values = clean[columns[family]].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f'All {family} auxiliary values must be finite; compute missing cheap quantities first')
        clean[columns[family]] = values
    if len(clean) < 12 or clean[DESIGN].duplicated().any():
        raise ValueError('Pretraining requires at least 12 unique full designs')
    continuous = (clean[CONTINUOUS].to_numpy(float)-LOWER)/(UPPER-LOWER)
    if np.linalg.matrix_rank(continuous-continuous.mean(axis=0)) < len(CONTINUOUS):
        raise ValueError('Pretraining designs must span all three continuous inputs')
    for name in CATEGORICAL:
        if set(clean[name]) != set(catalog['variables'][name]['types']):
            raise ValueError(f'Pretraining must cover all catalog types of {name}')
    return clean.reset_index(drop=True), columns, mode


def fit_representation(catalog, output, epochs=2000, seed=42, latent_dim=4,
                       hidden_dim=64, beta=.01, geometry_weight=1., physics_weight=1., lr=.001,
                       designs=None, per_combination=64, batch_size=128):
    if epochs < 1 or not 1 <= latent_dim <= 9 or hidden_dim < 4 or batch_size < 1:
        raise ValueError('Require epochs >= 1, latent dimension 1–9, hidden dimension >= 4, and batch size >= 1')
    if (not np.isfinite([beta, geometry_weight, physics_weight, lr]).all() or
            min(beta, geometry_weight, physics_weight) < 0 or lr <= 0):
        raise ValueError('Loss weights must be finite and nonnegative; learning rate must be positive')
    frame, columns, mode = prepare_pretraining(catalog, designs, per_combination, seed)
    x = design_matrix(frame, catalog)
    g_raw, p_raw = [frame[columns[k]].to_numpy(float) for k in ['geometry', 'physics']]
    g, g_mean, g_scale = standardized(g_raw)
    p, p_mean, p_scale = standardized(p_raw)
    architecture = dict(design_dim=x.shape[1], geometry_dim=g.shape[1], physics_dim=p.shape[1],
                        latent_dim=latent_dim, hidden_dim=hidden_dim)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    history = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ShapeVAE(**architecture).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            xt, gt, pt = map(torch.tensor, (x, g, p))
            print(f'Pretraining one full-design VAE on {len(x)} cheap designs: {x.shape[1]} → {latent_dim} dimensions', flush=True)
            for epoch in range(epochs):
                totals = dict.fromkeys(['loss', 'reconstruction_l2', 'kl', 'geometry_mse', 'physics_mse'], 0.)
                order = torch.randperm(len(x))
                for indices in order.split(batch_size):
                    outputs = model(xt[indices])
                    loss, terms = shape_losses(xt[indices], gt[indices], pt[indices], outputs,
                                               beta, geometry_weight, physics_weight)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite VAE loss; check cheap descriptors and training settings')
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    optimizer.step()
                    for key, value in terms.items():
                        totals[key] += value.item()*len(indices)/len(x)
                history.append({'epoch': epoch+1, **totals, 'beta': beta,
                                'lambda_g': geometry_weight, 'lambda_p': physics_weight})
                if epoch == 0 or (epoch+1) % max(1, epochs//5) == 0:
                    print(f"VAE epoch {epoch+1}/{epochs}: L_shape={totals['loss']:.6g}", flush=True)
            model.eval().requires_grad_(False)
            with torch.inference_mode():
                mu, logvar = model.encode(xt)
                reconstructed, g_hat, p_hat = model.decode(mu)
            means, logvars = mu.numpy(), logvar.numpy()
            xr, gr, pr = reconstructed.numpy(), g_hat.numpy(), p_hat.numpy()
            state = {key: value.tolist() for key, value in model.state_dict().items()}
    finally:
        torch.set_num_threads(old_threads)
    diagnostics = {
        'reconstruction_l2_at_mean': float(np.mean(np.sum((xr-x)**2, axis=1))),
        'continuous_rmse_normalized_at_mean': np.sqrt(np.mean((xr[:, :3]-x[:, :3])**2, axis=0)).tolist(),
        'category_accuracy_at_mean': {name: float(np.mean(xr[:, 3+3*i:6+3*i].argmax(1) ==
                                                       x[:, 3+3*i:6+3*i].argmax(1)))
                                      for i, name in enumerate(CATEGORICAL)},
        'geometry_mse_standardized_at_mean': float(np.mean((gr-g)**2)),
        'physics_mse_standardized_at_mean': float(np.mean((pr-p)**2)),
        'latent_standard_deviations_across_designs': means.std(axis=0).tolist(),
        'constant_geometry_columns': np.flatnonzero(np.ptp(g_raw, axis=0) == 0).tolist(),
        'constant_physics_columns': np.flatnonzero(np.ptp(p_raw, axis=0) == 0).tolist(),
    }
    representation = {
        'schema': 2, 'model_type': 'full_design_physics_shaped_vae', 'catalog': catalog,
        'catalog_sha256': fingerprint(catalog), 'architecture': architecture,
        'state_dict': state, 'state_sha256': fingerprint(state),
        'input_transform': {'continuous_columns': CONTINUOUS, 'lower': LOWER.tolist(), 'upper': UPPER.tolist(),
                            'categorical_columns': CATEGORICAL, 'categorical_encoding': 'onehot'},
        'settings': {'epochs': epochs, 'seed': seed, 'beta': beta, 'geometry_weight': geometry_weight,
                     'physics_weight': physics_weight, 'learning_rate': lr, 'batch_size': batch_size},
        'auxiliaries': {'mode': mode, 'columns': columns, 'geometry_mean': g_mean.tolist(),
                        'geometry_scale': g_scale.tolist(), 'physics_mean': p_mean.tolist(), 'physics_scale': p_scale.tolist()},
        'pretraining_sample_count': len(frame), 'pretraining_data_sha256': fingerprint(frame.to_dict('list')),
        'encoder_input': 'full numeric design x: normalized continuous values and one-hot categories',
        'loss': 'mean(||x-x_hat||_2^2) + beta*mean(KL) + lambda_g*MSE(g_standardized) + lambda_p*MSE(p_standardized)',
        'uses_high_fidelity_y': False, 'gp_representation': 'mu_phi(x) only; frozen; no appended raw inputs or rescaling',
        'diagnostics': diagnostics,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'representation.json').write_text(json.dumps(representation, indent=2, allow_nan=False)+'\n')
    torch.save(model.state_dict(), output/'shape_vae.pt')
    frame.to_csv(output/'pretraining_data.csv', index=False)
    pd.DataFrame(history).to_csv(output/'vae_training.csv', index=False)
    embeddings = frame[DESIGN].copy()
    for j in range(latent_dim):
        embeddings[f'z_{j}'] = means[:, j]
        embeddings[f'posterior_std_{j}'] = np.exp(.5*logvars[:, j])
    embeddings.to_csv(output/'embeddings.csv', index=False)
    recon = frame.copy()
    for j, column in enumerate(CONTINUOUS):
        recon[f'{column}_reconstructed'] = LOWER[j]+xr[:, j]*(UPPER[j]-LOWER[j])
    for i, column in enumerate(CATEGORICAL):
        types = catalog['variables'][column]['types']
        recon[f'{column}_reconstructed'] = [types[j] for j in xr[:, 3+3*i:6+3*i].argmax(1)]
    for family, values in [('geometry', gr*g_scale+g_mean), ('physics', pr*p_scale+p_mean)]:
        for j, column in enumerate(columns[family]):
            recon[f'{column}_reconstructed'] = values[:, j]
    recon.to_csv(output/'pretraining_reconstruction.csv', index=False)
    return representation


def load_representation(directory):
    value = json.loads((Path(directory)/'representation.json').read_text())
    if value.get('schema') != 2 or value.get('model_type') != 'full_design_physics_shaped_vae':
        raise ValueError('This is an older categorical-only representation. Re-run fit-vae and GP training for the full-design model')
    if value.get('catalog_sha256') != fingerprint(value['catalog']) or value.get('state_sha256') != fingerprint(value['state_dict']):
        raise ValueError('Representation weights or catalog were modified; re-run fit-vae and GP training')
    transform = value.get('input_transform', {})
    if (transform.get('continuous_columns') != CONTINUOUS or transform.get('categorical_columns') != CATEGORICAL
            or transform.get('lower') != LOWER.tolist() or transform.get('upper') != UPPER.tolist()):
        raise ValueError('Saved design transform does not match the current study bounds; retrain the representation')
    return value


def frozen_model(representation):
    if representation.get('schema') != 2:
        raise ValueError('Re-run fit-vae to create a full-design representation')
    with torch.random.fork_rng(devices=[]):
        model = ShapeVAE(**representation['architecture']).double()
    model.load_state_dict({key: torch.tensor(value, dtype=torch.double)
                           for key, value in representation['state_dict'].items()})
    return model.eval().requires_grad_(False)


def feature_matrix(frame, representation, encoding='vae'):
    catalog = representation['catalog']
    x = design_matrix(frame, catalog)
    if encoding == 'vae':
        # This is exactly T_shape(x)=mu_phi(x). No stochastic sample, concatenation,
        # descriptor input, or additional learned transform reaches the GP.
        model = frozen_model(representation)
        with torch.inference_mode():
            mu, _ = model.encode(torch.tensor(x))
        return mu.numpy()
    if encoding == 'onehot':
        return x
    if encoding != 'descriptors':
        raise ValueError('Encoding must be vae, descriptors, or onehot')
    pieces = [x[:, :3]]
    for name in CATEGORICAL:
        block = catalog['variables'][name]
        indices = np.array([block['types'].index(v) for v in frame[name]], dtype=int)
        table = np.concatenate([standardized(a)[0] for a in descriptor_arrays(block)], axis=1)
        pieces.append(table[indices])
    return np.concatenate(pieces, axis=1)
