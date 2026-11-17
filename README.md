# canopy_jax

A minimal, differentiable, CLM5-style canopy model in JAX.

- Farquhar C3 photosynthesis with CLM5 temperature responses
- Medlyn stomatal conductance
- Leaf energy balance, 10-layer canopy (Beer's-law light, exponential Vcmax profile)
- Nested root solves (ci and leaf temperature) use "solve without gradients, then one differentiable Newton step", which gives exact gradients via the implicit function theorem

`canopy_jax_demo.ipynb` runs the model over 5 synthetic days, checks autodiff gradients against finite differences (relative error ~1e-10), and recovers `Vcmax25` and `g1` from noisy synthetic GPP and latent heat observations with Adam.

## Install

```bash
pip install jax optax matplotlib numpy jupyter
# GPU: pip install "jax[cuda12]"
```

## Limitations

Not validated against CLM. No sunlit/shaded split or two-stream radiation, no canopy air profiles, no soil-moisture stress, no TPU limitation, cs = ca.
