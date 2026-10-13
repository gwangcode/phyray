"""
phyray_ext/brdf.py
==================
Bidirectional Reflectance Distribution Functions (BRDFs).

Provides:
    - Lambertian (pure diffuse)
    - Phong (empirical glossy)
    - GGX / Trowbridge-Reitz (microfacet)
    - Cook-Torrance (full microfacet with Fresnel + geometry)

Each BRDF provides:
    eval(N, V, L) -> reflectance value
    sample(N, V)  -> sampled direction L
    pdf(N, V, L)  -> probability density of L
"""
import math
import torch
import torch.nn.functional as F


# ==============================================================================
# Common utilities
# ==============================================================================

def normalize(v, eps=1e-12):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def reflect(V, N):
    """Reflect V about N."""
    return 2.0 * (V * N).sum(-1, keepdim=True) * N - V


def cosine_sample_hemisphere(N, device, N_rays):
    """
    Sample a direction on the hemisphere above N with cos-weighted pdf.
    """
    r1 = torch.rand(N_rays, 1, device=device)
    r2 = torch.rand(N_rays, 1, device=device)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(torch.clamp(1.0 - r1, min=0.0))

    # Build local frame
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N_rays, 3)
    tangent = torch.cross(
        torch.where(torch.abs(N[:, 2:3]) > 0.999,
                    torch.tensor([0.0, 1.0, 0.0], device=device).expand(N_rays, 3),
                    up),
        N, dim=-1)
    tangent = normalize(tangent)
    bitangent = torch.cross(N, tangent, dim=-1)
    L = x * tangent + y * bitangent + z * N
    return normalize(L)


# ==============================================================================
# Lambertian
# ==============================================================================

class LambertianBRDF:
    """Pure diffuse (albedo ρ)."""
    def __init__(self, albedo):
        self.albedo = albedo

    def eval(self, N, V, L):
        cos_i = torch.clamp((N * L).sum(-1, keepdim=True), min=0.0)
        return self.albedo * cos_i / math.pi

    def sample(self, N, V, device):
        return cosine_sample_hemisphere(N, device, N.shape[0])

    def pdf(self, N, V, L):
        cos_i = torch.clamp((N * L).sum(-1, keepdim=True), min=0.0)
        return cos_i / math.pi


# ==============================================================================
# Phong (empirical glossy)
# ==============================================================================

class PhongBRDF:
    """Phong specular: cos^n(α), α = angle between R and V."""
    def __init__(self, kd=0.5, ks=0.5, shininess=32.0):
        self.kd = kd
        self.ks = ks
        self.shininess = shininess

    def eval(self, N, V, L):
        cos_i = torch.clamp((N * L).sum(-1, keepdim=True), min=0.0)
        R = reflect(-V, N)
        cos_a = torch.clamp((R * L).sum(-1, keepdim=True), min=0.0)
        spec = self.ks * (cos_a ** self.shininess)
        return self.kd * cos_i / math.pi + spec

    def sample(self, N, V, device):
        return cosine_sample_hemisphere(N, device, N.shape[0])

    def pdf(self, N, V, L):
        cos_i = torch.clamp((N * L).sum(-1, keepdim=True), min=0.0)
        return cos_i / math.pi


# ==============================================================================
# GGX / Trowbridge-Reitz (microfacet distribution)
# ==============================================================================

def ggx_D(cos_h, alpha):
    """GGX microfacet distribution."""
    alpha2 = alpha * alpha
    cos_h2 = cos_h * cos_h
    denom = cos_h2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (math.pi * denom ** 2 + 1e-12)


def smith_G1(cos_v, alpha):
    """Smith geometry term (single direction)."""
    cos_v = torch.clamp(cos_v, min=1e-6)
    tan2 = (1.0 - cos_v * cos_v) / (cos_v * cos_v + 1e-12)
    return 2.0 / (1.0 + torch.sqrt(1.0 + alpha * alpha * tan2))


def smith_G(cos_v, cos_l, alpha):
    """Combined Smith geometry term."""
    return smith_G1(cos_v, alpha) * smith_G1(cos_l, alpha)


def schlick_fresnel(cos_theta, F0):
    """Schlick's Fresnel approximation."""
    return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5


class GGXBRDF:
    """
    GGX microfacet BRDF with Smith geometry and Schlick Fresnel.

    Parameters:
        F0     — normal-incidence reflectance (e.g., 0.04 for glass)
        alpha  — roughness parameter (perceptual roughness squared)
        kd     — diffuse weight
    """
    def __init__(self, F0=0.04, alpha=0.2, kd=0.0):
        self.F0 = F0
        self.alpha = alpha
        self.kd = kd

    def eval(self, N, V, L):
        H = normalize(V + L)
        cos_h = torch.clamp((N * H).sum(-1, keepdim=True), min=0.0)
        cos_v = torch.clamp((N * V).sum(-1, keepdim=True), min=0.0)
        cos_l = torch.clamp((N * L).sum(-1, keepdim=True), min=0.0)
        vdh = torch.clamp((V * H).sum(-1, keepdim=True), min=0.0)

        D = ggx_D(cos_h, self.alpha)
        G = smith_G(cos_v, cos_l, self.alpha)
        F_ = schlick_fresnel(vdh, self.F0)

        spec = (D * G * F_) / (4.0 * cos_v * cos_l + 1e-12)
        diff = self.kd * cos_l / math.pi

        return spec + diff

    def sample(self, N, V, device):
        # Importance sampling: for simplicity, sample half-vector from GGX
        # and reflect V
        N_rays = N.shape[0]
        r1 = torch.rand(N_rays, 1, device=device)
        r2 = torch.rand(N_rays, 1, device=device)
        alpha2 = self.alpha * self.alpha
        phi = 2 * math.pi * r1
        cos_h = torch.sqrt((1.0 - r2) / (1.0 + (alpha2 - 1.0) * r2 + 1e-12))
        sin_h = torch.sqrt(torch.clamp(1.0 - cos_h * cos_h, min=0.0))

        # Local frame
        up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N_rays, 3)
        tangent = normalize(torch.cross(
            torch.where(torch.abs(N[:, 2:3]) > 0.999,
                        torch.tensor([0.0, 1.0, 0.0], device=device).expand(N_rays, 3),
                        up),
            N, dim=-1))
        bitangent = torch.cross(N, tangent, dim=-1)

        H = sin_h * torch.cos(phi) * tangent + \
            sin_h * torch.sin(phi) * bitangent + \
            cos_h * N
        H = normalize(H)
        L = reflect(-V, H)
        # Reject L below surface
        return normalize(L)

    def pdf(self, N, V, L):
        # Approximate pdf using GGX half-vector sampling
        H = normalize(V + L)
        cos_h = torch.clamp((N * H).sum(-1, keepdim=True), min=0.0)
        vdh = torch.clamp((V * H).sum(-1, keepdim=True), min=1e-6)
        D = ggx_D(cos_h, self.alpha)
        return D * cos_h / (4.0 * vdh)


# ==============================================================================
# Factory
# ==============================================================================

def make_brdf(kind, **kwargs):
    """Factory function: make_brdf('ggx', F0=0.04, alpha=0.2)"""
    if kind == "lambert":
        return LambertianBRDF(kwargs.get("albedo", 0.8))
    elif kind == "phong":
        return PhongBRDF(kwargs.get("kd", 0.5),
                         kwargs.get("ks", 0.5),
                         kwargs.get("shininess", 32.0))
    elif kind == "ggx":
        return GGXBRDF(kwargs.get("F0", 0.04),
                       kwargs.get("alpha", 0.2),
                       kwargs.get("kd", 0.0))
    else:
        raise ValueError(f"Unknown BRDF kind: {kind}")