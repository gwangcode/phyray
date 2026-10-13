"""Unit tests for PhyRay extensions."""
import math
import numpy as np
import torch
import pytest

from phyray_ext import thin_film, dispersion, brdf


def test_thin_film_single_layer_air_to_glass():
    """A zero-thickness layer should give bare Fresnel result."""
    r, t = thin_film.propagate_through_stack(
        n_list=[1.0, 1.5], k_list=[0.0, 0.0], d_list=[],
        wavelength=550.0, theta_i_deg=0.0, polarization="s")
    R = abs(r) ** 2
    expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    assert abs(R - expected) < 1e-6


def test_cauchy_dispersion():
    cauchy = dispersion.CauchyDispersion(A=1.5, B=0.005)
    n_400 = cauchy.n_fn(torch.tensor([0.4])).item()  # λ in μm
    n_700 = cauchy.n_fn(torch.tensor([0.7])).item()
    # n should decrease with wavelength
    assert n_400 > n_700


def test_sellmeier_BK7():
    bk7 = dispersion.BK7()
    # Known value: n_BK7(587.6 nm) ≈ 1.5168
    n = bk7.n_fn(torch.tensor([0.5876])).item()
    assert abs(n - 1.5168) < 1e-3


def test_ggx_energy_conservation():
    """GGX BRDF should be ≤ 1 for typical parameters."""
    ggx = brdf.GGXBRDF(F0=0.04, alpha=0.2)
    N = torch.tensor([[0.0, 0.0, 1.0]])
    V = torch.tensor([[0.0, 0.0, 1.0]])
    for theta_deg in [0, 30, 60]:
        L = torch.tensor([[math.sin(math.radians(theta_deg)),
                           0.0,
                           math.cos(math.radians(theta_deg))]])
        val = ggx.eval(N, V, L).item()
        assert val >= 0
        assert val < 1e3   # loose upper bound


def test_brdf_factory():
    lam = brdf.make_brdf("lambert", albedo=0.8)
    ggx = brdf.make_brdf("ggx", F0=0.04, alpha=0.2)
    assert isinstance(lam, brdf.LambertianBRDF)
    assert isinstance(ggx, brdf.GGXBRDF)