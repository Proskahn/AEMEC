# Mixed-variable GP validation



36 observations; encoding `vae`; CV `random` (5 folds).



| Objective | CV R² | RMSE | MAE |

| --- | ---: | ---: | ---: |

| cell_voltage_v | 1 | 3.0469e-05 | 2.20468e-05 |

| crossover_rate_mol_s | 0.998182 | 2.90512e-11 | 1.97815e-11 |



VAE/catalog features are fitted only to human descriptors and frozen before GP fitting; GP output normalization and hyperparameter fits use training-fold observations only.

GP predictive intervals condition on the fitted embedding and kernel; VAE uncertainty is not propagated.

Descriptor reconstruction alone does not guarantee preservation of material distances or BO improvement.

Random CV estimates within-catalog prediction. Group-held-out CV tests transfer across specified groups.

SYNTHETIC DEMONSTRATION: descriptors/objectives are not AEMEC material measurements.
