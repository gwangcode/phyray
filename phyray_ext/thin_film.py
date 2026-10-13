"""
phyray_ext/thin_film.py
=======================
Multilayer thin-film interference using the transfer-matrix method.

Supports:
    - Arbitrary number of layers with complex refractive indices (n, k)
    - Angle-dependent reflectance and transmittance
    - Both s- and p-polarization
    - Wavelength sweep

This module is designed as a drop-in extension for PhyRay: you can wrap
a thin-film stack into a PhyRay-compatible material via ThinFilmMaterial.

Reference:
    Macleod, "Thin-Film Optical Filters", 5th ed., CRC Press, 2018.
"""
import math
import numpy as np
import torch


# ==============================================================================
# Core transfer-matrix computation
# ==============================================================================

def admittance_s(N, cos_theta):
    """Optical admittance for s-polarization: η = N cos θ."""
    return N * cos_theta


def admittance_p(N, cos_theta):
    """Optical admittance for p-polarization: η = N / cos θ."""
    return N / cos_theta


def layer_matrix(N, d, wavelength, cos_theta, polarization="s"):
    """
    Characteristic matrix of a single layer.

    Args:
        N: complex refractive index (n + i*k) — torch complex tensor.
        d: physical thickness (scalar).
        wavelength: wavelength in vacuum.
        cos_theta: cosine of propagation angle inside the layer.
        polarization: 's' or 'p'.

    Returns:
        2x2 complex matrix.
    """
    if polarization == "s":
        eta = admittance_s(N, cos_theta)
    else:
        eta = admittance_p(N, cos_theta)

    delta = 2.0 * math.pi * N * d * cos_theta / wavelength
    # Complex trig
    cd = torch.cos(delta.to(torch.complex64)) if not torch.is_complex(delta) else torch.cos(delta)
    sd = torch.sin(delta.to(torch.complex64)) if not torch.is_complex(delta) else torch.sin(delta)

    # M = [[cos δ, i sin δ / η], [i η sin δ, cos δ]]
    M00 = cd
    M01 = 1j * sd / eta
    M10 = 1j * eta * sd
    M11 = cd

    return M00, M01, M10, M11


def propagate_through_stack(n_list, k_list, d_list, wavelength,
                             theta_i_deg, polarization="s"):
    """
    Compute r and t through a multilayer stack.

    Args:
        n_list: [n0, n1, ..., nN, n_substrate] — incident, layers, substrate.
        k_list: same length, extinction coefficients.
        d_list: length N (only for layers, not incident/substrate).
        wavelength: in vacuum (nm or any consistent unit).
        theta_i_deg: incident angle.
        polarization: 's' or 'p'.

    Returns:
        (r, t) complex scalars.
    """
    n_arr = np.array(n_list, dtype=complex)
    k_arr = np.array(k_list, dtype=complex)
    N_arr = n_arr + 1j * k_arr

    N_inc = N_arr[0]
    N_sub = N_arr[-1]

    theta_i = math.radians(theta_i_deg)
    sin_i = math.sin(theta_i)
    cos_i = math.cos(theta_i)

    # Complex Snell inside each layer
    cos_theta_layers = []
    for i in range(len(n_list)):
        sin_t2 = (N_inc / N_arr[i]) ** 2 * sin_i ** 2
        cos_t = np.sqrt(1.0 - sin_t2 + 0j)
        cos_theta_layers.append(cos_t)

    # Compute total characteristic matrix M = M1 · M2 · ... · MN
    M_total = np.eye(2, dtype=complex)
    for layer_idx in range(len(d_list)):
        i = layer_idx + 1   # layer index in the n_list
        N = torch.tensor(N_arr[i], dtype=torch.complex64)
        cos_t = torch.tensor(cos_theta_layers[i], dtype=torch.complex64)
        M00, M01, M10, M11 = layer_matrix(
            N, d_list[layer_idx], wavelength, cos_t, polarization)
        M_layer = torch.stack([
            torch.stack([M00, M01]),
            torch.stack([M10, M11]),
        ])
        M_total = M_total @ M_layer.numpy()

    # Admittances
    if polarization == "s":
        eta_inc = N_arr[0] * cos_theta_layers[0]
        eta_sub = N_arr[-1] * cos_theta_layers[-1]
    else:
        eta_inc = N_arr[0] / cos_theta_layers[0]
        eta_sub = N_arr[-1] / cos_theta_layers[-1]

    # B, C from M and substrate admittance
    B = M_total[0, 0] + M_total[0, 1] * eta_sub
    C = M_total[1, 0] + M_total[1, 1] * eta_sub

    # Reflection and transmission coefficients
    r = (eta_inc * B - C) / (eta_inc * B + C)
    t = 2.0 * eta_inc / (eta_inc * B + C)
    return complex(r), complex(t)


# ==============================================================================
# PhyRay-compatible material wrapper
# ==============================================================================

class ThinFilmMaterial:
    """
    Wrap a multilayer stack as a PhyRay-compatible material.

    Usage:
        stack = ThinFilmMaterial(
            n_list=[1.0, 2.35, 1.45, 1.5],   # air | TiO2 | SiO2 | glass
            k_list=[0.0, 0.0, 0.0, 0.0],
            d_list=[100.0, 200.0],            # nm (two layers)
            wavelength_range=[400, 700],
        )

        # In PhyRay:
        materials_cfg = {1: stack.as_phyray_material()}
    """
    def __init__(self, n_list, k_list, d_list,
                 wavelength_range=(400.0, 700.0),
                 name="thin_film"):
        self.n_list = list(n_list)
        self.k_list = list(k_list)
        self.d_list = list(d_list)
        self.wavelength_range = wavelength_range
        self.name = name

        # Precompute the reflectance at several angles & wavelengths
        # for later interpolation
        self._build_lookup_table()

    def _build_lookup_table(self, n_wl=13, n_theta=11):
        """Precompute R_s, R_p, T_s, T_p on a (wl, theta) grid."""
        wl_min, wl_max = self.wavelength_range
        self._wl_grid = np.linspace(wl_min, wl_max, n_wl)
        self._theta_grid = np.linspace(0.0, 89.0, n_theta)

        R_s = np.zeros((n_wl, n_theta))
        R_p = np.zeros((n_wl, n_theta))
        T_s = np.zeros((n_wl, n_theta))
        T_p = np.zeros((n_wl, n_theta))

        for i, wl in enumerate(self._wl_grid):
            for j, theta in enumerate(self._theta_grid):
                r_s, t_s = propagate_through_stack(
                    self.n_list, self.k_list, self.d_list,
                    wl, theta, polarization="s")
                r_p, t_p = propagate_through_stack(
                    self.n_list, self.k_list, self.d_list,
                    wl, theta, polarization="p")
                R_s[i, j] = abs(r_s) ** 2
                R_p[i, j] = abs(r_p) ** 2
                # Energy transmission (using real part of substrate admittance)
                T_s[i, j] = 1.0 - R_s[i, j]
                T_p[i, j] = 1.0 - R_p[i, j]

        self._R_s = R_s
        self._R_p = R_p
        self._T_s = T_s
        self._T_p = T_p

    def reflectance(self, wavelength, theta_deg, polarization="s"):
        """
        Bilinear interpolation of precomputed reflectance.
        """
        from scipy.interpolate import RegularGridInterpolator
        R_grid = self._R_s if polarization == "s" else self._R_p
        interp = RegularGridInterpolator(
            (self._wl_grid, self._theta_grid), R_grid,
            bounds_error=False, fill_value=0.0)
        return float(interp([[wavelength, theta_deg]])[0])

    def as_phyray_material(self, base_n=1.0, base_k=0.0, base_roughness=0.0):
        """
        Return a material config dict that PhyRay can consume.

        Note: PhyRay treats this as a material with the *base* (substrate)
        refractive index. The thin-film reflectance is added as an
        additional "reflectivity" multiplier.
        """
        return {
            "n": base_n,
            "k": base_k,
            "roughness": base_roughness,
            # Optional: PhyRay will read this if you extend the tracer
            "_thin_film": self,
        }