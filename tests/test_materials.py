"""Unit tests for PhyRay's physical material model."""
import math
import numpy as np
import torch

from phyray.materials import (
    complex_fresnel,
    fresnel_reflectance,
    fresnel_transmittance_flow,
    beer_lambert,
    roughness_specular_factor,
)


def test_normal_incidence_dielectric():
    """At normal incidence, R = ((n1-n2)/(n1+n2))^2."""
    n1, k1 = 1.0, 0.0
    n2, k2 = 1.5, 0.0
    cos_i = torch.tensor([[1.0]])

    R_s, R_p, R_avg = fresnel_reflectance(
        torch.tensor([[n1]]), torch.tensor([[k1]]),
        torch.tensor([[n2]]), torch.tensor([[k2]]), cos_i)

    expected = ((n1 - n2) / (n1 + n2)) ** 2
    assert abs(R_avg.item() - expected) < 1e-6, \
        f"Expected {expected}, got {R_avg.item()}"


def test_energy_conservation_no_absorption():
    """With k=0, R + T = 1."""
    n1, k1 = 1.0, 0.0
    n2, k2 = 1.5, 0.0
    cos_i = torch.tensor([[0.8]])

    R_s, R_p, R_avg = fresnel_reflectance(
        torch.tensor([[n1]]), torch.tensor([[k1]]),
        torch.tensor([[n2]]), torch.tensor([[k2]]), cos_i)

    T_s, T_p, T_avg = fresnel_transmittance_flow(
        torch.tensor([[n1]]), torch.tensor([[k1]]),
        torch.tensor([[n2]]), torch.tensor([[k2]]), cos_i)

    total = R_avg + T_avg
    assert abs(total.item() - 1.0) < 1e-5, \
        f"R + T = {total.item()}, expected 1.0"


def test_beer_lambert():
    """I/I0 = exp(-4 pi k d / lambda)."""
    k = torch.tensor([[0.01]])
    d = 1e-3
    wl = torch.tensor([[550e-9]])

    atten = beer_lambert(k, d, wl)
    expected = math.exp(-4 * math.pi * 0.01 * 1e-3 / 550e-9)

    assert abs(atten.item() - expected) < 1e-6


def test_roughness_limits():
    """Specular factor → 1 for smooth, → 0 for very rough."""
    wl = torch.tensor([[550e-9]])
    cos_i = torch.tensor([[1.0]])

    # Smooth
    f_smooth = roughness_specular_factor(torch.tensor([[0.0]]), wl, cos_i)
    assert f_smooth.item() > 0.999

    # Very rough
    f_rough = roughness_specular_factor(torch.tensor([[10.0]]), wl, cos_i)
    assert f_rough.item() < 1e-6