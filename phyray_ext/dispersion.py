"""
phyray_ext/dispersion.py
========================
Wavelength-dependent refractive index models.

Provides:
    - Cauchy:     n(λ) = A + B/λ² + C/λ⁴
    - Sellmeier:  n²(λ) = 1 + Σ Bᵢ λ² / (λ² - Cᵢ)
    - Drude:      ε(ω) = ε_∞ - ω_p² / (ω² + i γ ω)   (metals)
    - Constant:   n(λ) = n₀

Each model returns a pair of callables (n_fn, k_fn) that fit PhyRay's
FunctionMaterial interface.
"""
import math
import numpy as np
import torch


# ==============================================================================
# Cauchy dispersion (transparent dielectrics)
# ==============================================================================

class CauchyDispersion:
    """
    Cauchy model: n(λ) = A + B/λ² + C/λ⁴, k(λ) = 0 (transparent).

    Wavelength in micrometers or nanometers — you choose, but stay consistent.
    """

    def __init__(self, A, B=0.0, C=0.0):
        self.A = A
        self.B = B
        self.C = C

    def n_fn(self, wavelength):
        wl = wavelength if isinstance(wavelength, torch.Tensor) else torch.tensor(wavelength)
        # Ensure positive
        wl2 = wl ** 2
        wl4 = wl ** 4
        return self.A + self.B / (wl2 + 1e-12) + self.C / (wl4 + 1e-12)

    def k_fn(self, wavelength):
        return torch.zeros_like(wavelength if isinstance(wavelength, torch.Tensor)
                                 else torch.tensor(wavelength))

    def as_phyray_material(self, roughness=0.0):
        """Return a PhyRay FunctionMaterial config."""
        # Note: This returns a config; PhyRay's build_material uses n_fn
        return {
            "n_fn": lambda x, y, z, wl: self.n_fn(wl),
            "k_fn": lambda x, y, z, wl: self.k_fn(wl),
            "roughness": roughness,
        }


# ==============================================================================
# Sellmeier dispersion (BK7, fused silica, etc.)
# ==============================================================================

class SellmeierDispersion:
    """
    Sellmeier model: n²(λ) = 1 + Σ Bᵢ λ² / (λ² - Cᵢ).

    Reference: Schott glass data sheets.
    """

    def __init__(self, B_list, C_list):
        assert len(B_list) == len(C_list)
        self.B = list(B_list)
        self.C = list(C_list)

    def n_fn(self, wavelength):
        wl = wavelength if isinstance(wavelength, torch.Tensor) else torch.tensor(wavelength)
        wl2 = wl ** 2
        n2 = torch.ones_like(wl2)
        for B, C in zip(self.B, self.C):
            n2 = n2 + B * wl2 / (wl2 - C + 1e-12)
        return torch.sqrt(torch.clamp(n2, min=1e-12))

    def k_fn(self, wavelength):
        return torch.zeros_like(wavelength if isinstance(wavelength, torch.Tensor)
                                 else torch.tensor(wavelength))

    def as_phyray_material(self, roughness=0.0):
        return {
            "n_fn": lambda x, y, z, wl: self.n_fn(wl),
            "k_fn": lambda x, y, z, wl: self.k_fn(wl),
            "roughness": roughness,
        }


# ==============================================================================
# Drude model (metals)
# ==============================================================================

class DrudeDispersion:
    """
    Drude model for metals:
        ε(ω) = ε_∞ - ω_p² / (ω² + i γ ω)
        N = sqrt(ε)  →  N = n + i k

    Parameters:
        epsilon_inf : high-frequency dielectric constant
        omega_p     : plasma frequency (rad/s)
        gamma       : damping rate (rad/s)
    """

    def __init__(self, epsilon_inf, omega_p, gamma, c=3e8):
        self.eps_inf = epsilon_inf
        self.omega_p = omega_p
        self.gamma = gamma
        self.c = c   # speed of light

    def _complex_N(self, wavelength):
        wl = wavelength if isinstance(wavelength, torch.Tensor) else torch.tensor(wavelength)
        omega = 2.0 * math.pi * self.c / (wl + 1e-12)
        eps = self.eps_inf - self.omega_p ** 2 / (
            omega ** 2 + 1j * self.gamma * omega + 1e-30)
        N = torch.sqrt(eps.to(torch.complex64))
        n = torch.real(N).to(wl.dtype)
        k = torch.imag(N).to(wl.dtype)
        return n, k

    def n_fn(self, wavelength):
        n, _ = self._complex_N(wavelength)
        return n

    def k_fn(self, wavelength):
        _, k = self._complex_N(wavelength)
        return k

    def as_phyray_material(self, roughness=0.0):
        return {
            "n_fn": lambda x, y, z, wl: self.n_fn(wl),
            "k_fn": lambda x, y, z, wl: self.k_fn(wl),
            "roughness": roughness,
        }


# ==============================================================================
# Constant (baseline)
# ==============================================================================

class ConstantDispersion:
    """n(λ) = n₀, k(λ) = k₀."""

    def __init__(self, n0, k0=0.0):
        self.n0 = n0
        self.k0 = k0

    def n_fn(self, wavelength):
        wl = wavelength if isinstance(wavelength, torch.Tensor) else torch.tensor(wavelength)
        return torch.full_like(wl, float(self.n0))

    def k_fn(self, wavelength):
        wl = wavelength if isinstance(wavelength, torch.Tensor) else torch.tensor(wavelength)
        return torch.full_like(wl, float(self.k0))

    def as_phyray_material(self, roughness=0.0):
        return {
            "n_fn": lambda x, y, z, wl: self.n_fn(wl),
            "k_fn": lambda x, y, z, wl: self.k_fn(wl),
            "roughness": roughness,
        }


# ==============================================================================
# Library of common materials
# ==============================================================================

def BK7():
    """Schott N-BK7 glass."""
    return SellmeierDispersion(
        B_list=[1.03961212, 0.231792344, 1.01046945],
        C_list=[0.00600069867, 0.0200179144, 103.560653])

def fused_silica():
    """Fused silica (Malitson)."""
    return SellmeierDispersion(
        B_list=[0.6961663, 0.4079426, 0.8974794],
        C_list=[0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2])

def SF11():
    """Schott N-SF11 dense flint."""
    return SellmeierDispersion(
        B_list=[1.73759695, 0.313747346, 1.89878101],
        C_list=[0.013188707, 0.0623068142, 155.23629])

def water():
    """Water at 20°C (Daimon & Masumura)."""
    return SellmeierDispersion(
        B_list=[5.684027565e-1, 1.726177391e-1, 2.086189578e-2,
                1.130748688e-1],
        C_list=[5.101829712e-3, 1.821153936e-2, 2.620722293e-2,
                1.069792721e1])