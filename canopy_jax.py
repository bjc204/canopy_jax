"""
canopy_jax.py -- a minimal, differentiable, CLM-style canopy model in JAX.

Physics (simplified from CLM5's PhotosynthesisMod / CanopyFluxesMod):
  * Farquhar/von Caemmerer/Berry C3 photosynthesis with CLM5 temperature
    responses (Arrhenius + high-temperature deactivation, Bernacchi 2001 kinetics)
  * Medlyn (2011) stomatal conductance (CLM5 default)
  * Leaf energy balance: absorbed radiation = emitted LW + sensible + latent heat
  * Multilayer canopy: Beer's-law light and exponential Vcmax decline with depth

Numerics:
  Both nested solves (ci for photosynthesis-stomata coupling, Tleaf for the
  energy balance) use the same pattern:
      1. find the root WITHOUT gradients (robust bisection / damped Newton)
      2. take ONE differentiable Newton step from the converged root
  Step 2 yields exact first-order gradients via the implicit function theorem,
  so we never backpropagate through iterative loops (cheap + numerically safe).

Deliberate simplifications vs. CLM (see README):
  no sunlit/shaded split, no within-canopy air profiles, no soil-moisture
  stress (btran = 1), no TPU limitation, constant latent heat, cs = ca.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)  # float64: safer for nested solvers

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
R_GAS = 8.314            # J mol-1 K-1
T25 = 298.15             # K
TFRZ = 273.15            # K
SIGMA = 5.67e-8          # Stefan-Boltzmann, W m-2 K-4
EMISS = 0.98             # leaf emissivity
CP_MOL = 29.3            # molar heat capacity of air, J mol-1 K-1
LAMBDA_MOL = 44.0e3      # latent heat of vaporization, J mol-1 (approx. 25 C)
O2 = 209.0e3             # O2 mole fraction, umol mol-1
PAR_PER_W = 4.6          # umol photons per J of PAR
SG = lax.stop_gradient


class Params(NamedTuple):
    """Model parameters (all differentiable)."""
    vcmax25: float = 60.0       # top-of-canopy Vcmax at 25 C, umol m-2 s-1
    g1: float = 4.0             # Medlyn slope, kPa^0.5
    g0: float = 1.0e-4          # minimum stomatal conductance, mol m-2 s-1 (CLM5: 100 umol)
    jmax_ratio: float = 1.67    # Jmax25 / Vcmax25
    rd_ratio: float = 0.015     # Rd25 / Vcmax25
    kn: float = 0.3             # Vcmax decline coefficient with canopy depth
    k_ext: float = 0.5          # light extinction coefficient
    theta_j: float = 0.7        # curvature, electron transport (CLM)
    theta_cj: float = 0.98      # Ac/Aj co-limitation curvature (CLM)
    leaf_width: float = 0.05    # m, for boundary-layer conductance


class Forcing(NamedTuple):
    """Atmospheric forcing at one time step (scalars; vmap for series/sites)."""
    sw: float      # incoming shortwave, W m-2
    tair: float    # air temperature, K
    ea: float      # vapor pressure, Pa
    wind: float    # wind speed, m s-1
    co2: float     # CO2 mole fraction, umol mol-1
    pres: float    # air pressure, Pa
    lai: float     # leaf area index, m2 m-2


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def esat(t):
    """Saturation vapor pressure (Pa) at temperature t (K)."""
    tc = t - TFRZ
    return 611.2 * jnp.exp(17.67 * tc / (tc + 243.5))


def arrhenius(t, ha):
    return jnp.exp(ha / (R_GAS * T25) * (1.0 - T25 / t))


def high_temp_inhibition(t, hd, ds):
    """CLM's fth(T) / fth(25 C)."""
    fth = lambda tk: 1.0 + jnp.exp((-hd + ds * tk) / (R_GAS * tk))
    return fth(T25) / fth(t)


def photosynthesis(ci, tleaf, apar, vcmax25_leaf, p: Params):
    """Gross and net assimilation (umol m-2 s-1) given intercellular CO2 ci (umol/mol)."""
    vcmax = vcmax25_leaf * arrhenius(tleaf, 65330.) * high_temp_inhibition(tleaf, 149250., 485.)
    jmax = p.jmax_ratio * vcmax25_leaf * arrhenius(tleaf, 43540.) * high_temp_inhibition(tleaf, 152040., 495.)
    rd = p.rd_ratio * vcmax25_leaf * arrhenius(tleaf, 46390.) * high_temp_inhibition(tleaf, 150650., 490.)
    kc = 404.9 * arrhenius(tleaf, 79430.)
    ko = 278.4e3 * arrhenius(tleaf, 36380.)
    gstar = 42.75 * arrhenius(tleaf, 37830.)

    # electron transport: smaller root of theta*J^2 - (I + Jmax) J + I Jmax = 0
    i_psii = 0.5 * 0.85 * apar
    b = i_psii + jmax
    j = (b - jnp.sqrt(b * b - 4.0 * p.theta_j * i_psii * jmax)) / (2.0 * p.theta_j)

    ac = vcmax * (ci - gstar) / (ci + kc * (1.0 + O2 / ko))   # Rubisco-limited
    aj = 0.25 * j * (ci - gstar) / (ci + 2.0 * gstar)         # light-limited

    # smooth co-limitation (smaller root); tiny epsilon keeps sqrt differentiable
    s = ac + aj
    a_gross = (s - jnp.sqrt(s * s - 4.0 * p.theta_cj * ac * aj + 1e-12)) / (2.0 * p.theta_cj)
    return a_gross, a_gross - rd


def boundary_layer(wind, p: Params):
    """Leaf boundary-layer conductances (mol m-2 s-1), Campbell & Norman forced convection."""
    u_d = jnp.sqrt(jnp.maximum(wind, 0.1) / p.leaf_width)
    return 1.4 * 0.135 * u_d, 1.4 * 0.147 * u_d   # heat, water vapor


def stomatal_conductance(an, tleaf, ea, co2, p: Params):
    """Medlyn conductance to water vapor (mol m-2 s-1), leaf-to-air VPD."""
    vpd_kpa = jnp.maximum(esat(tleaf) - ea, 50.0) / 1000.0
    return p.g0 + 1.6 * (1.0 + p.g1 / jnp.sqrt(vpd_kpa)) * jnp.maximum(an, 0.0) / co2


# ----------------------------------------------------------------------------
# Solver 1: intercellular CO2 (supply = demand)
# ----------------------------------------------------------------------------
def _ci_residual(ci, tleaf, apar, vcmax25_leaf, ea, co2, gbv, p):
    _, an = photosynthesis(ci, tleaf, apar, vcmax25_leaf, p)
    gs = stomatal_conductance(an, tleaf, ea, co2, p)
    gc = 1.0 / (1.6 / gs + 1.37 / gbv)            # total CO2 conductance
    return an - gc * (co2 - ci)                    # demand - supply


def _bisect(f, lo, hi, n_iter=60):
    def body(_, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0.0
        return jnp.where(pos, lo, mid), jnp.where(pos, mid, hi)
    lo, hi = lax.fori_loop(0, n_iter, body, (lo, hi))
    return 0.5 * (lo + hi)


def solve_ci(tleaf, apar, vcmax25_leaf, ea, co2, gbv, p):
    args = (tleaf, apar, vcmax25_leaf, ea, co2, gbv, p)
    frozen = jax.tree.map(SG, args)
    # residual < 0 at ci = 0; > 0 at very high ci (also covers night, when ci > ca)
    ci_star = SG(_bisect(lambda c: _ci_residual(c, *frozen), 0.0, co2 + 1.0e5))
    r, dr = jax.value_and_grad(_ci_residual)(ci_star, *args)
    return ci_star - r / dr                        # implicit-gradient Newton step


# ----------------------------------------------------------------------------
# Solver 2: leaf temperature (energy balance)
# ----------------------------------------------------------------------------
def leaf_state(tleaf, apar, sw_abs, vcmax25_leaf, f: Forcing, p: Params):
    """All leaf fluxes at a given leaf temperature, plus energy-balance residual."""
    gbh, gbv = boundary_layer(f.wind, p)
    ci = solve_ci(tleaf, apar, vcmax25_leaf, f.ea, f.co2, gbv, p)
    a_gross, an = photosynthesis(ci, tleaf, apar, vcmax25_leaf, p)
    gs = stomatal_conductance(an, tleaf, f.ea, f.co2, p)
    gw = 1.0 / (1.0 / gs + 1.0 / gbv)
    e = gw * (esat(tleaf) - f.ea) / f.pres                       # mol H2O m-2 s-1
    h = 2.0 * CP_MOL * gbh * (tleaf - f.tair)                    # W m-2 (two sides)
    r_abs = sw_abs + 2.0 * EMISS * SIGMA * f.tair ** 4           # surroundings at Tair
    r_emit = 2.0 * EMISS * SIGMA * tleaf ** 4
    resid = r_abs - r_emit - h - LAMBDA_MOL * e
    return dict(tleaf=tleaf, ci=ci, a_gross=a_gross, an=an, gs=gs, e=e,
                h=h, le=LAMBDA_MOL * e, energy_resid=resid)


def solve_leaf(apar, sw_abs, vcmax25_leaf, f: Forcing, p: Params, n_iter=12):
    args = (apar, sw_abs, vcmax25_leaf, f, p)
    resid = lambda tl, *a: leaf_state(tl, *a)["energy_resid"]
    frozen = jax.tree.map(SG, args)

    def newton(_, tl):
        r, dr = jax.value_and_grad(resid)(tl, *frozen)
        return tl + jnp.clip(-r / dr, -5.0, 5.0)

    tl_star = SG(lax.fori_loop(0, n_iter, newton, frozen[3].tair))
    r, dr = jax.value_and_grad(resid)(tl_star, *args)
    return leaf_state(tl_star - r / dr, *args)


# ----------------------------------------------------------------------------
# Canopy: integrate leaves over layers
# ----------------------------------------------------------------------------
def canopy(f: Forcing, p: Params, n_layers=10):
    """Canopy fluxes per unit ground area at one time step."""
    dl = f.lai / n_layers
    lcum = (jnp.arange(n_layers) + 0.5) * dl                 # cumulative LAI at layer mid
    beer = p.k_ext * jnp.exp(-p.k_ext * lcum)                # per-leaf-area attenuation
    par_top = 0.5 * f.sw * PAR_PER_W
    apar = 0.85 * beer * par_top                             # umol m-2 leaf s-1
    sw_abs = beer * 0.5 * f.sw * (0.85 + 0.20)               # PAR + NIR absorbed, W m-2 leaf
    vcmax_layer = p.vcmax25 * jnp.exp(-p.kn * lcum)

    leaves = jax.vmap(solve_leaf, in_axes=(0, 0, 0, None, None))(apar, sw_abs, vcmax_layer, f, p)
    integ = lambda x: jnp.sum(x) * dl
    return dict(
        gpp=integ(leaves["a_gross"]),          # umol CO2 m-2 s-1
        an=integ(leaves["an"]),
        le=integ(leaves["le"]),                # W m-2
        h=integ(leaves["h"]),                  # W m-2
        transp=integ(leaves["e"]) * 18.015e-3 * 3600.0,   # mm h-1
        gc=integ(leaves["gs"]),                # canopy conductance, mol m-2 s-1
        tleaf_top=leaves["tleaf"][0],
        max_energy_resid=jnp.max(jnp.abs(leaves["energy_resid"])),
    )


# vectorized over time (and, with another vmap, over sites)
run = jax.jit(jax.vmap(canopy, in_axes=(0, None)))


# ----------------------------------------------------------------------------
# Synthetic forcing
# ----------------------------------------------------------------------------
def make_forcing(n_days=5, dt_hours=0.5, seed=0, lai=4.0):
    import numpy as np
    rng = np.random.default_rng(seed)
    hours = np.arange(0, 24 * n_days, dt_hours)
    day = (hours // 24).astype(int)
    hod = hours % 24
    sw_max = rng.uniform(450, 950, n_days)[day]            # cloudy-to-clear days
    # daily mean T and RH interpolated between days (no jumps at midnight)
    noon = np.arange(n_days) * 24 + 12
    t_mean = np.interp(hours, noon, rng.uniform(16, 28, n_days))
    rh_mean = np.interp(hours, noon, rng.uniform(0.45, 0.8, n_days))
    sw = sw_max * np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None)
    diurnal = np.sin(2 * np.pi * (hod - 9) / 24)            # peaks ~15:00
    tair = t_mean + 5.0 * diurnal + TFRZ
    rh = np.clip(rh_mean - 0.2 * diurnal, 0.15, 0.98)
    ea = rh * np.asarray(esat(tair))
    n = hours.size
    f = Forcing(sw=jnp.asarray(sw), tair=jnp.asarray(tair), ea=jnp.asarray(ea),
                wind=jnp.full(n, 2.0), co2=jnp.full(n, 420.0),
                pres=jnp.full(n, 101325.0), lai=jnp.full(n, lai))
    return hours, f
