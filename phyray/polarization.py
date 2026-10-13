"""Polarization utilities with complex Jones-Mueller support."""
import torch


def _choose_perpendicular(d):
    """Physical s-polarization direction = cross(ref, d)."""
    ref = torch.zeros_like(d)
    ref[:, 2] = 1.0
    near_z = torch.abs(d[:, 2]) > 0.99
    ref[near_z] = torch.tensor([0.0, 1.0, 0.0], device=d.device)

    s = torch.cross(ref, d, dim=-1)

    norm = s.norm(dim=-1, keepdim=True)
    eps = 1e-12
    safe_norm = norm.clamp_min(eps)
    s_normalized = s / safe_norm

    fallback = torch.zeros_like(s)
    fallback[:, 0] = 1.0
    degenerate = (norm < eps).expand_as(s)
    return torch.where(degenerate, fallback, s_normalized)


def rotate_stokes_reference(stokes, s_old, s_new, d):
    """Rotate Stokes vector from (s_old, p_old) basis to (s_new, p_new)."""
    cos_t = torch.sum(s_old * s_new, dim=-1, keepdim=True)
    cross_val = torch.cross(s_old, s_new, dim=-1)
    sin_t = torch.sum(cross_val * d, dim=-1, keepdim=True)
    cos_2t = cos_t * cos_t - sin_t * sin_t
    sin_2t = 2.0 * cos_t * sin_t
    S0, S1, S2, S3 = stokes[..., 0:1], stokes[..., 1:2], stokes[..., 2:3], stokes[..., 3:4]
    S1_new = cos_2t * S1 + sin_2t * S2
    S2_new = -sin_2t * S1 + cos_2t * S2
    return torch.cat([S0, S1_new, S2_new, S3], dim=-1)


# ==============================================================================
# ★ Complex Jones-Mueller (NEW)
# ==============================================================================

def apply_jones_mueller(stokes, r_s, r_p):
    """
    Apply a diagonal Jones matrix J = diag(r_s, r_p) to a Stokes vector,
    preserving phase information.

    Args:
        stokes: [N, 4] Stokes vectors.
        r_s, r_p: [N, 1] Fresnel coefficients. May be REAL or COMPLEX.

    Returns:
        [N, 4] Stokes vector, normalized to S0 = 1.

    The full Mueller matrix for J = diag(r_s, r_p) is:

        M = [[ a,  b,  0,  0 ],
             [ b,  a,  0,  0 ],
             [ 0,  0,  c, -d ],
             [ 0,  0,  d,  c ]]

    where
        a = (|r_s|² + |r_p|²) / 2
        b = (|r_s|² - |r_p|²) / 2
        c = Re(r_s · conj(r_p))
        d = Im(r_s · conj(r_p))

    When r_s, r_p are real, d = 0 and we recover the classical Mueller matrix.
    The d-term captures the RELATIVE PHASE between s and p — this is what
    turns TIR into elliptical/circular polarization.

    References:
        Chipman, Lam, Young, "Polarized Light and Optical Systems", CRC 2018.
        Born & Wolf, "Principles of Optics", §1.6.
    """
    # Compute magnitudes
    abs_rs2 = torch.abs(r_s) ** 2
    abs_rp2 = torch.abs(r_p) ** 2

    a = 0.5 * (abs_rs2 + abs_rp2)
    b = 0.5 * (abs_rs2 - abs_rp2)

    # Cross term r_s · conj(r_p) — complex
    rs_conj_rp = r_s * torch.conj(r_p)
    c = torch.real(rs_conj_rp)         # [N, 1]
    d = torch.imag(rs_conj_rp)         # [N, 1]

    S0, S1, S2, S3 = stokes[..., 0:1], stokes[..., 1:2], stokes[..., 2:3], stokes[..., 3:4]

    S0_new = a * S0 + b * S1
    S1_new = b * S0 + a * S1
    S2_new = c * S2 - d * S3           # ★ d-term preserves phase
    S3_new = d * S2 + c * S3           # ★ d-term preserves phase

    # Normalize to S0 = 1
    S0_safe = torch.clamp(S0_new, min=1e-12)
    return torch.cat([
        torch.ones_like(S1_new),
        S1_new / S0_safe,
        S2_new / S0_safe,
        S3_new / S0_safe,
    ], dim=-1)


def apply_jones_mueller_complex_full(stokes, r_s, r_p):
    """
    Same as apply_jones_mueller but WITHOUT normalization.

    Use this when you want to track BOTH polarization state AND intensity.
    The S0 component will carry the Fresnel energy loss.
    """
    abs_rs2 = torch.abs(r_s) ** 2
    abs_rp2 = torch.abs(r_p) ** 2

    a = 0.5 * (abs_rs2 + abs_rp2)
    b = 0.5 * (abs_rs2 - abs_rp2)
    rs_conj_rp = r_s * torch.conj(r_p)
    c = torch.real(rs_conj_rp)
    d = torch.imag(rs_conj_rp)

    S0, S1, S2, S3 = stokes[..., 0:1], stokes[..., 1:2], stokes[..., 2:3], stokes[..., 3:4]

    S0_new = a * S0 + b * S1
    S1_new = b * S0 + a * S1
    S2_new = c * S2 - d * S3
    S3_new = d * S2 + c * S3

    return torch.cat([S0_new, S1_new, S2_new, S3_new], dim=-1)


def randomize_stokes_diffuse(stokes):
    """Fully randomized Stokes vector (unpolarized after diffuse scattering)."""
    return torch.cat([torch.ones_like(stokes[..., 0:1]),
                      torch.zeros_like(stokes[..., 1:4])], dim=-1)


def stokes_to_pol_vector(stokes, s_ref, p_ref):
    """Derive the main polarization direction from a Stokes vector."""
    S1 = stokes[..., 1:2]
    S2 = stokes[..., 2:3]
    psi = 0.5 * torch.atan2(S2, S1)
    pol = torch.cos(psi) * s_ref + torch.sin(psi) * p_ref
    return pol / (torch.norm(pol, dim=-1, keepdim=True) + 1e-12)


def degree_of_polarization(stokes):
    """DoP = sqrt(S1^2 + S2^2 + S3^2) / S0."""
    S0 = stokes[..., 0:1]
    S1 = stokes[..., 1:2]
    S2 = stokes[..., 2:3]
    S3 = stokes[..., 3:4]
    return torch.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / (S0 + 1e-12)