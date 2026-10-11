"""Identity-conditioned variational models with human-descriptor decoder targets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import CATEGORICAL, CONTINUOUS, LOWER, UPPER, fingerprint, check_design


class DescriptorVAE(nn.Module):
    def __init__(self, categories, geometry_dim, physics_dim, latent_dim=2, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(categories, hidden_dim), nn.Tanh())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh())
        self.identity_head = nn.Linear(hidden_dim, categories)
        self.geometry_head = nn.Linear(hidden_dim, geometry_dim)
        self.physics_head = nn.Linear(hidden_dim, physics_dim)

    def encode(self, identity):
        hidden = self.encoder(identity)
        return self.mu(hidden), self.logvar(hidden).clamp(-10., 6.)

    def decode(self, latent):
        hidden = self.decoder(latent)
        return self.identity_head(hidden), self.geometry_head(hidden), self.physics_head(hidden)

    def forward(self, identity):
        mu, logvar = self.encode(identity)
        latent = mu + torch.exp(.5*logvar)*torch.randn_like(mu)
        return (*self.decode(latent), mu, logvar)


def descriptor_arrays(block):
    return [np.array([block['descriptors'][t][family] for t in block['types']], dtype=float)
            for family in ['geometry', 'physics']]


def standardized(values):
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    # Material properties such as permeabilities may be far below 1e-12 in
    # physical units. Only a truly constant column should bypass scaling.
    scale = np.where(scale > 0., scale, 1.)
    return (values-center)/scale, center, scale


def fit_representation(catalog, output, epochs=2000, seed=42, latent_dim=2,
                       hidden_dim=16, beta=.01, geometry_weight=1., physics_weight=1., lr=.01):
    if epochs < 1 or not 1 <= latent_dim <= 8 or hidden_dim < 2:
        raise ValueError('Require epochs >= 1, latent dimension 1–8, and hidden dimension >= 2')
    if not np.isfinite([beta, geometry_weight, physics_weight, lr]).all() or min(
            beta, geometry_weight, physics_weight, lr) <= 0:
        raise ValueError('KL weight, geometry/physics weights, and learning rate must be finite and positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    representation = {'schema': 1, 'catalog': catalog, 'catalog_sha256': fingerprint(catalog),
                      'settings': {'epochs': epochs, 'seed': seed, 'latent_dim': latent_dim,
                                   'hidden_dim': hidden_dim, 'beta': beta, 'geometry_weight': geometry_weight,
                                   'physics_weight': physics_weight, 'learning_rate': lr},
                      'encoder_input': 'one-hot category identity only',
                      'decoder_targets': 'category identity plus standardized geometry and physics',
                      'gp_representation': 'deterministic posterior mean, frozen before GP fitting', 'variables': {}}
    history, reconstructions, embeddings = [], [], []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, name in enumerate(CATEGORICAL):
            print(f'Training descriptor VAE for {name} ({epochs} epochs; three catalog types)', flush=True)
            block = catalog['variables'][name]
            geometry, physics = descriptor_arrays(block)
            g, g_mean, g_scale = standardized(geometry)
            p, p_mean, p_scale = standardized(physics)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+index)
                model = DescriptorVAE(3, g.shape[1], p.shape[1], latent_dim, hidden_dim).double()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                identity = torch.eye(3, dtype=torch.double)
                g_target, p_target = torch.tensor(g), torch.tensor(p)
                for epoch in range(epochs):
                    logits, g_hat, p_hat, mu, logvar = model(identity)
                    reconstruction = F.cross_entropy(logits, torch.arange(3))
                    g_loss, p_loss = F.mse_loss(g_hat, g_target), F.mse_loss(p_hat, p_target)
                    kl = -.5*torch.mean(torch.sum(1+logvar-mu.square()-logvar.exp(), dim=1))
                    annealed_beta = beta*min(1., (epoch+1)/max(1, epochs*.25))
                    loss = reconstruction + geometry_weight*g_loss + physics_weight*p_loss + annealed_beta*kl
                    if not torch.isfinite(loss):
                        raise ValueError(f'{name}: non-finite VAE loss; check descriptor scales and settings')
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    optimizer.step()
                    history.append({'variable': name, 'epoch': epoch+1, 'loss': loss.item(),
                                    'identity_ce': reconstruction.item(), 'geometry_mse': g_loss.item(),
                                    'physics_mse': p_loss.item(), 'kl': kl.item(), 'beta': annealed_beta})
                model.eval()
                with torch.no_grad():
                    mu, logvar = model.encode(identity)
                    logits, g_hat, p_hat = model.decode(mu)
                means, logvars = mu.numpy(), logvar.numpy()
                geometry_hat, physics_hat = g_hat.numpy()*g_scale+g_mean, p_hat.numpy()*p_scale+p_mean
                pairs = [(i, j) for i in range(3) for j in range(i+1, 3)]
                diagnostics = {
                    'identity_accuracy_at_mean': float(np.mean(logits.argmax(1).numpy() == np.arange(3))),
                    'geometry_mse_standardized_at_mean': float(np.mean((g_hat.numpy()-g)**2)),
                    'physics_mse_standardized_at_mean': float(np.mean((p_hat.numpy()-p)**2)),
                    'pair_distances': [{'types': [block['types'][i], block['types'][j]],
                                       'latent_distance': float(np.linalg.norm(means[i]-means[j])),
                                       'descriptor_distance': float(np.linalg.norm(np.r_[g[i]-g[j], p[i]-p[j]]))}
                                      for i, j in pairs],
                    'constant_geometry_columns': np.flatnonzero(np.ptp(geometry, axis=0) == 0).tolist(),
                    'constant_physics_columns': np.flatnonzero(np.ptp(physics, axis=0) == 0).tolist(),
                }
                representation['variables'][name] = {
                    'means': means.tolist(), 'log_variances': logvars.tolist(), 'diagnostics': diagnostics,
                    'geometry_mean': g_mean.tolist(), 'geometry_scale': g_scale.tolist(),
                    'physics_mean': p_mean.tolist(), 'physics_scale': p_scale.tolist()}
                torch.save(model.state_dict(), output/f'{name}_vae.pt')
                for i, category in enumerate(block['types']):
                    embeddings.append({'variable': name, 'category': category,
                                       **{f'z_{j}': value for j, value in enumerate(means[i])},
                                       **{f'posterior_std_{j}': value for j, value in enumerate(np.exp(.5*logvars[i]))}})
                    for family, actual, predicted in [('geometry', geometry, geometry_hat), ('physics', physics, physics_hat)]:
                        for j, field in enumerate(block[family]):
                            reconstructions.append({'variable': name, 'category': category, 'family': family,
                                                    'descriptor': field['name'], 'unit': field['unit'],
                                                    'human_value': actual[i, j], 'reconstructed_at_mean': predicted[i, j]})
    finally:
        torch.set_num_threads(old_threads)
    (output/'representation.json').write_text(json.dumps(representation, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(history).to_csv(output/'vae_training.csv', index=False)
    pd.DataFrame(embeddings).to_csv(output/'embeddings.csv', index=False)
    pd.DataFrame(reconstructions).to_csv(output/'descriptor_reconstruction.csv', index=False)
    return representation


def load_representation(directory):
    value = json.loads((Path(directory)/'representation.json').read_text())
    if value.get('schema') != 1 or value.get('catalog_sha256') != fingerprint(value['catalog']):
        raise ValueError('Invalid representation artifact or modified catalog; retrain the VAE')
    return value


def feature_matrix(frame, representation, encoding='vae'):
    catalog = representation['catalog']
    frame = check_design(frame, catalog)
    pieces = [(frame[CONTINUOUS].to_numpy(float)-LOWER)/(UPPER-LOWER)]
    for name in CATEGORICAL:
        block = catalog['variables'][name]
        indices = np.array([block['types'].index(v) for v in frame[name]], dtype=int)
        if encoding == 'vae':
            table = np.array(representation['variables'][name]['means'])
            # One scalar per category family preserves the learned Euclidean
            # geometry, unlike independent rescaling of the latent coordinates.
            table = table-table.mean(axis=0)
            scale = np.sqrt(np.mean(table**2))
            if scale < 1e-10:
                raise ValueError(f'{name}: collapsed VAE posterior means; inspect reconstruction and reduce beta')
            table = table/scale
        elif encoding == 'descriptors':
            table = np.concatenate([standardized(a)[0] for a in descriptor_arrays(block)], axis=1)
        elif encoding == 'onehot':
            table = np.eye(3)
        else:
            raise ValueError('Encoding must be vae, descriptors, or onehot')
        pieces.append(table[indices])
    return np.concatenate(pieces, axis=1)
