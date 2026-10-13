"""Physical material models based on complex refractive index (n, k) and
surface roughness.

Provides:
    - Complex Fresnel coefficients for absorbing media
    - Beer-Lambert volumetric absorption
    - Beckmann-Kirchhoff rough surface scattering
    - Unified specular + diffuse reflectance model
"""
import math
import torch


# ==============================================================================
# Complex Fresnel coefficients
# ==============================================================================

def complex_fresnel(n1, k1, n2, k2, cos_theta_i):
    """
    Complex Fresnel coefficients for an interface between two absorbing media.

    N1 = n1 + i*k1, N2 = n2 + i*k2.

    Args:
        n1, k1: complex index of incident medium, tensors broadcastable to cos_theta_i.
        n2, k2: complex index of transmitted medium.
        cos_theta_i: cosine of incidence angle, real tensor [N, 1].

    Returns:
        (r_s, r_p, t_s, t_p): complex tensors [N, 1].
    """
    # Complex refractive indices
    N1 = torch.complex(n1, k1)
    N2 = torch.complex(n2, k2)

    cos_i = cos_theta_i
    sin_i2 = torch.clamp(1.0 - cos_i ** 2, min=0.0)

    # Complex Snell's law: N1 sin(theta_i) = N2 sin(theta_t)
    sin_t2 = (N1 / N2) ** 2 * sin_i2
    cos_t = torch.sqrt(1.0 - sin_t2 + 0j)

    # s-polarization (TE)
    denom_s = N1 * cos_i + N2 * cos_t
    r_s = (N1 * cos_i - N2 * cos_t) / denom_s
    t_s = 2.0 * N1 * cos_i / denom_s

    # p-polarization (TM)
    denom_p = N2 * cos_i + N1 * cos_t
    r_p = (N2 * cos_i - N1 * cos_t) / denom_p
    t_p = 2.0 * N1 * cos_i / denom_p

    return r_s, r_p, t_s, t_p


def fresnel_reflectance(n1, k1, n2, k2, cos_theta_i):
    """
    Real energy reflectance from complex Fresnel coefficients.

    Returns:
        (R_s, R_p, R_avg): real tensors [N, 1].
    """
    r_s, r_p, _, _ = complex_fresnel(n1, k1, n2, k2, cos_theta_i)
    R_s = torch.abs(r_s) ** 2
    R_p = torch.abs(r_p) ** 2
    R_avg = 0.5 * (R_s + R_p)
    return R_s, R_p, R_avg


def fresnel_transmittance_flow(n1, k1, n2, k2, cos_theta_i):
    """
    Energy transmittance (normalized to incident irradiance) for absorbing media.

    T = Re(N2 * cos_theta_t) / (N1 * cos_theta_i) * |t|^2

    Returns:
        (T_s, T_p, T_avg): real tensors [N, 1].
    """
    r_s, r_p, t_s, t_p = complex_fresnel(n1, k1, n2, k2, cos_theta_i)

    N1 = torch.complex(n1, k1)
    N2 = torch.complex(n2, k2)
    cos_i = cos_theta_i
    sin_i2 = torch.clamp(1.0 - cos_i ** 2, min=0.0)
    sin_t2 = (N1 / N2) ** 2 * sin_i2
    cos_t = torch.sqrt(1.0 - sin_t2 + 0j)

    # Energy flow factor
    factor = torch.real(N2 * cos_t) / (torch.real(N1) * cos_i + 1e-12)

    T_s = factor * torch.abs(t_s) ** 2
    T_p = factor * torch.abs(t_p) ** 2
    T_avg = 0.5 * (T_s + T_p)
    return T_s, T_p, T_avg


def beer_lambert(k, path_length, wavelength):
    """
    Volumetric attenuation over a path length.

    I(d) = I_0 * exp(-4*pi*k*d / lambda)

    Args:
        k: extinction coefficient tensor [N, 1].
        path_length: propagation distance (tensor or scalar).
        wavelength: wavelength in vacuum [N, 1].

    Returns:
        attenuation factor [N, 1], in (0, 1].
    """
    alpha = 4.0 * math.pi * k / (wavelength + 1e-12)
    return torch.exp(-alpha * path_length)


# ==============================================================================
# Roughness model (Beckmann-Kirchhoff)
# ==============================================================================

def roughness_specular_factor(roughness, wavelength, cos_theta_i):
    """
    Beckmann-Kirchhoff specular attenuation factor for a rough surface.

    g = (4 * pi * roughness * cos_theta_i / lambda)^2
    specular_factor = exp(-g)

    When roughness -> 0, factor -> 1 (perfect specular).
    When roughness >> lambda, factor -> 0 (all diffuse).

    Args:
        roughness: RMS height sigma [N, 1].
        wavelength: [N, 1].
        cos_theta_i: [N, 1].

    Returns:
        specular_factor [N, 1], in (0, 1].
    """
    g = (4.0 * math.pi * roughness * cos_theta_i / (wavelength + 1e-12)) ** 2
    return torch.exp(-g)


def roughness_diffuse_fraction(roughness, wavelength, cos_theta_i):
    """
    Fraction of incident energy scattered diffusely by a rough surface.

    diffuse_frac = 1 - exp(-g)

    This is a simplification of the full Beckmann-Kirchhoff model,
    capturing the essential physics: roughness increases diffuse scattering.
    """
    return 1.0 - roughness_specular_factor(roughness, wavelength, cos_theta_i)


# ==============================================================================
# Material classes
# ==============================================================================

class BaseMaterial:
    """
    Base class for physical materials.

    Subclasses must implement:
        get_complex_index(wavelength) -> (n, k)
        get_roughness(wavelength) -> roughness
        get_axis() -> optical axis for anisotropy
    """

    def __init__(self, name="material"):
        self.name = name

    def get_complex_index(self, wavelength):
        """Return (n, k) at the given wavelength. Override in subclasses."""
        raise NotImplementedError

    def get_roughness(self, wavelength):
        """Return RMS roughness in world units. Default: 0 (smooth)."""
        # By default, roughness is zero
        n, _ = self.get_complex_index(wavelength)
        return torch.zeros_like(n)

    def get_axis(self):
        """Optical axis for anisotropic materials. Default: None (isotropic)."""
        return None


class DielectricMaterial(BaseMaterial):
    """Non-absorbing or weakly absorbing dielectric: n real, k >= 0."""

    def __init__(self, n, k=0.0, roughness=0.0, name="dielectric"):
        super().__init__(name)
        self._n = float(n)
        self._k = float(k)
        self._roughness = float(roughness)

    def get_complex_index(self, wavelength):
        n = torch.full_like(wavelength, self._n)
        k = torch.full_like(wavelength, self._k)
        return n, k

    def get_roughness(self, wavelength):
        return torch.full_like(wavelength, self._roughness)


class ConductorMaterial(BaseMaterial):
    """Metallic conductor: n and k both wavelength-dependent."""

    def __init__(self, n_data, k_data, wavelengths_nm, roughness=0.0,
                 name="conductor"):
        """
        Args:
            n_data: list of n values.
            k_data: list of k values.
            wavelengths_nm: corresponding wavelengths.
        """
        super().__init__(name)
        self._n_pts = torch.tensor(n_data, dtype=torch.float32)
        self._k_pts = torch.tensor(k_data, dtype=torch.float32)
        self._wl_pts = torch.tensor(wavelengths_nm, dtype=torch.float32)
        self._roughness = float(roughness)

    def _interp(self, wl, data):
        # Simple linear interpolation (fallback to nearest)
        wl = wl.squeeze(-1)
        idx = torch.clamp(
            torch.searchsorted(self._wl_pts, wl, right=False),
            1, len(self._wl_pts) - 1)
        x0 = self._wl_pts[idx - 1]
        x1 = self._wl_pts[idx]
        y0 = data[idx - 1]
        y1 = data[idx]
        t = ((wl - x0) / (x1 - x0 + 1e-12)).clamp(0, 1)
        return (y0 + t * (y1 - y0)).unsqueeze(-1)

    def get_complex_index(self, wavelength):
        n = self._interp(wavelength, self._n_pts)
        k = self._interp(wavelength, self._k_pts)
        return n, k

    def get_roughness(self, wavelength):
        return torch.full_like(wavelength, self._roughness)


class RoughDielectricMaterial(DielectricMaterial):
    """Dielectric with explicit roughness parameter."""

    def __init__(self, n, k=0.0, roughness=0.0, name="rough_dielectric"):
        super().__init__(n, k, roughness, name)


class FunctionMaterial(BaseMaterial):
    """Material defined by callables n(wl), k(wl), roughness(wl)."""

    def __init__(self, n_fn, k_fn=None, roughness_fn=None, name="function"):
        super().__init__(name)
        self._n_fn = n_fn
        self._k_fn = k_fn
        self._roughness_fn = roughness_fn

    def get_complex_index(self, wavelength):
        n = self._n_fn(wavelength)
        k = self._k_fn(wavelength) if self._k_fn else torch.zeros_like(n)
        return n, k

    def get_roughness(self, wavelength):
        if self._roughness_fn:
            return self._roughness_fn(wavelength)
        n, _ = self.get_complex_index(wavelength)
        return torch.zeros_like(n)


# ==============================================================================
# Factory: build a material from a user-supplied dict
# ==============================================================================

def build_material(cfg):
    """
    Build a material object from a configuration dict.

    Supported keys:
        - "n", "k", "roughness": scalar values -> DielectricMaterial
        - "n_fn", "k_fn", "roughness_fn": callables -> FunctionMaterial
        - "n_data", "k_data", "wavelengths_nm": conductor data
    """
    if "n_data" in cfg and "k_data" in cfg and "wavelengths_nm" in cfg:
        return ConductorMaterial(
            cfg["n_data"], cfg["k_data"], cfg["wavelengths_nm"],
            roughness=cfg.get("roughness", 0.0))

    if "n_fn" in cfg:
        return FunctionMaterial(
            cfg["n_fn"],
            cfg.get("k_fn"),
            cfg.get("roughness_fn"))

    # Simple scalar
    return DielectricMaterial(
        n=cfg.get("n", 1.0),
        k=cfg.get("k", 0.0),
        roughness=cfg.get("roughness", 0.0))