# Mixed-variable GP validation

36 observations; encoding `vae`; CV `random` (5 folds).

| Objective | CV R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| cell_voltage_v | 0.843076 | 0.0406436 | 0.0320622 |
| crossover_rate_mol_s | 0.977088 | 1.03146e-10 | 7.06383e-11 |

The full-design VAE is pretrained on x, g(x), and p(x), with no expensive y, then frozen; the VAE encoding gives the GP only mu_phi(x), without appended raw continuous inputs. GP output normalization and hyperparameter fits use training-fold observations only.
GP predictive intervals condition on the fitted embedding and kernel; VAE uncertainty is not propagated.
Gaussian objective predictions can be negative and are not clipped; inspect extrapolative predictions.
Descriptor reconstruction alone does not guarantee preservation of material distances or BO improvement.
GP CV is conditional on the fixed cheap-data representation. If held-out designs are included in pretraining, their x/g/p are known transductively, but their expensive y is never used.
Random CV estimates within-catalog prediction. Group-held-out CV tests transfer across specified groups.
Captured GP optimization warnings: 0; details are recorded in metadata.json.
SYNTHETIC DEMONSTRATION: descriptors/objectives are not AEMEC material measurements.
