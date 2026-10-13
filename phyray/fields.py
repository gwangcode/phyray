"""Refractive index fields for PhyRay.

A field defines how the refractive index varies in space.
Each field has a base material (n, k, roughness) and an optional
spatial modulation.
"""
import torch
import torch.nn as nn

from .materials import BaseMaterial, DielectricMaterial


class BaseRefractiveField(nn.Module):
    """
    Base class for spatially varying refractive index fields.

    A field must provide:
        - get_complex_index(r, wavelength) -> (n, k)
        - get_roughness(r, wavelength) -> roughness
        - get_gradient(r, wavelength) -> grad_n (for GRIN propagation)
    """

    def __init__(self, material: BaseMaterial):
        super().__init__()
        self.material = material
        self._current_pol = None
        self._current_dop = None

    def set_context(self, pol, dop):
        self._current_pol = pol
        self._current_dop = dop

    def clear_context(self):
        self._current_pol = None
        self._current_dop = None

    def get_complex_index(self, r, wavelength):
        raise NotImplementedError

    def get_roughness(self, r, wavelength):
        raise NotImplementedError

    def get_gradient(self, r, wavelength, eps=1e-3):
        """Numerical gradient of n (real part)."""
        grads = []
        for a in range(3):
            e = torch.zeros_like(r)
            e[..., a] = eps
            n_p, _ = self.get_complex_index(r + e, wavelength)
            n_m, _ = self.get_complex_index(r - e, wavelength)
            grads.append((n_p - n_m) / (2.0 * eps))
        return torch.cat(grads, dim=-1)


class ConstantField(BaseRefractiveField):
    """Uniform material (no spatial variation)."""

    def get_complex_index(self, r, wavelength):
        n, k = self.material.get_complex_index(wavelength)
        # Broadcast to r.shape[0]
        n = n.expand(r.shape[0], 1) if n.shape[0] != r.shape[0] else n
        k = k.expand(r.shape[0], 1) if k.shape[0] != r.shape[0] else k
        return n, k

    def get_roughness(self, r, wavelength):
        sig = self.material.get_roughness(wavelength)
        return sig.expand(r.shape[0], 1) if sig.shape[0] != r.shape[0] else sig


class FunctionField(BaseRefractiveField):
    """
    Field defined by user-supplied callables:
        n_fn(x, y, z, wavelength) -> n
        k_fn(x, y, z, wavelength) -> k  (optional, defaults to material's k)
        roughness_fn(x, y, z, wavelength) -> roughness
    """

    def __init__(self, material, n_fn, k_fn=None, roughness_fn=None):
        super().__init__(material)
        self.n_fn = n_fn
        self.k_fn = k_fn
        self.roughness_fn = roughness_fn

    def get_complex_index(self, r, wavelength):
        x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
        n = self.n_fn(x, y, z, wavelength)
        if n.dim() == 1:
            n = n.unsqueeze(-1)
        if self.k_fn is not None:
            k = self.k_fn(x, y, z, wavelength)
            if k.dim() == 1:
                k = k.unsqueeze(-1)
        else:
            _, k = self.material.get_complex_index(wavelength)
            k = k.expand_as(n) if k.shape[0] != n.shape[0] else k
        return n, k

    def get_roughness(self, r, wavelength):
        if self.roughness_fn is not None:
            x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
            sig = self.roughness_fn(x, y, z, wavelength)
            if sig.dim() == 1:
                sig = sig.unsqueeze(-1)
            return sig
        return self.material.get_roughness(wavelength).expand(r.shape[0], 1)


class GridField(BaseRefractiveField):
    """
    Field defined on a regular grid with trilinear interpolation.

    Config keys:
        "x", "y", "z": grid coordinates.
        "values": 3D array of n values.
        "k_values": 3D array of k values (optional).
        "roughness_values": 3D array of roughness values (optional).
    """

    def __init__(self, material, config):
        super().__init__(material)
        self.x = torch.tensor(config["x"], dtype=torch.float32)
        self.y = torch.tensor(config["y"], dtype=torch.float32)
        self.z = torch.tensor(config["z"], dtype=torch.float32)
        self.values = torch.tensor(config["values"], dtype=torch.float32)
        self.k_values = torch.tensor(
            config.get("k_values", torch.zeros_like(self.values)),
            dtype=torch.float32)
        self.roughness_values = torch.tensor(
            config.get("roughness_values", torch.zeros_like(self.values)),
            dtype=torch.float32)

    def _trilinear(self, r, grid_x, grid_y, grid_z, values):
        """Trilinear interpolation. values shape: [nx, ny, nz]."""
        x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]

        def interp_axis(v, g):
            idx = torch.clamp(
                torch.searchsorted(g, v.squeeze(-1), right=False),
                1, g.shape[0] - 1)
            x0, x1 = g[idx - 1], g[idx]
            t = ((v.squeeze(-1) - x0) / (x1 - x0 + 1e-12)).clamp(0, 1)
            return idx, t

        ix, tx = interp_axis(x, grid_x)
        iy, ty = interp_axis(y, grid_y)
        iz, tz = interp_axis(z, grid_z)

        v000 = values[ix - 1, iy - 1, iz - 1]
        v001 = values[ix - 1, iy - 1, iz]
        v010 = values[ix - 1, iy, iz - 1]
        v011 = values[ix - 1, iy, iz]
        v100 = values[ix, iy - 1, iz - 1]
        v101 = values[ix, iy - 1, iz]
        v110 = values[ix, iy, iz - 1]
        v111 = values[ix, iy, iz]

        v00 = v000 * (1 - tz) + v001 * tz
        v01 = v010 * (1 - tz) + v011 * tz
        v10 = v100 * (1 - tz) + v101 * tz
        v11 = v110 * (1 - tz) + v111 * tz

        v0 = v00 * (1 - ty) + v01 * ty
        v1 = v10 * (1 - ty) + v11 * ty

        return (v0 * (1 - tx) + v1 * tx).unsqueeze(-1)

    def get_complex_index(self, r, wavelength):
        n = self._trilinear(r, self.x, self.y, self.z, self.values)
        k = self._trilinear(r, self.x, self.y, self.z, self.k_values)
        return n, k

    def get_roughness(self, r, wavelength):
        return self._trilinear(r, self.x, self.y, self.z,
                               self.roughness_values)


class AnisotropicField(FunctionField):
    """
    Field with anisotropic (birefringent) refractive index.

    Additional parameters:
        delta_n_fn: function of (x, y, z, wavelength) giving birefringence.
        axis: optical axis direction (tuple, default (1,0,0)).
    """

    def __init__(self, material, n_fn, delta_n_fn,
                 axis=(1.0, 0.0, 0.0), k_fn=None, roughness_fn=None):
        super().__init__(material, n_fn, k_fn, roughness_fn)
        self.delta_n_fn = delta_n_fn
        self.axis = axis

    def get_complex_index(self, r, wavelength):
        n_avg, k = super().get_complex_index(r, wavelength)

        # Anisotropic correction
        x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
        dn = self.delta_n_fn(x, y, z, wavelength)
        if dn.dim() == 1:
            dn = dn.unsqueeze(-1)

        # Polarization-dependent effective index
        if self._current_pol is not None:
            axis_t = torch.tensor(self.axis, dtype=r.dtype, device=r.device)
            axis_t = axis_t.unsqueeze(0).expand(r.shape[0], 3)
            cos2 = (self._current_pol * axis_t).sum(-1, keepdim=True) ** 2
            n_eff = n_avg + dn * (cos2 - 0.5)
        else:
            n_eff = n_avg

        return n_eff, k


# ==============================================================================
# Factory
# ==============================================================================

def build_field(cfg):
    """Build a field from a configuration dict."""
    material = cfg.get("material") or DielectricMaterial(
        n=cfg.get("n", 1.5), k=cfg.get("k", 0.0),
        roughness=cfg.get("roughness", 0.0))

    if "n_fn" in cfg and "delta_n_fn" in cfg:
        return AnisotropicField(
            material, cfg["n_fn"], cfg["delta_n_fn"],
            axis=cfg.get("axis", (1.0, 0.0, 0.0)),
            k_fn=cfg.get("k_fn"),
            roughness_fn=cfg.get("roughness_fn"))

    if "n_fn" in cfg:
        return FunctionField(
            material, cfg["n_fn"],
            k_fn=cfg.get("k_fn"),
            roughness_fn=cfg.get("roughness_fn"))

    if "values" in cfg:
        return GridField(material, cfg)

    return ConstantField(material)