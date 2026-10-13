"""PhyRay: A physically-based GPU ray tracer for GRIN and polarization optics.

Materials are defined by (n, k, roughness):
    n         — real refractive index
    k         — extinction coefficient (imaginary part of complex index)
    roughness — RMS surface roughness in world units

All interface physics (Fresnel reflection, absorption) are derived
from these three parameters.
"""

from .tracer import PhysicalRayTracer
from .materials import (
    BaseMaterial,
    DielectricMaterial,
    ConductorMaterial,
    RoughDielectricMaterial,
)
from .fields import (
    BaseRefractiveField,
    ConstantField,
    FunctionField,
    GridField,
    AnisotropicField,
)
from .sources import generate_rays
from .io import save_trace_output_to_npz, export_ray_history_to_stl

__version__ = "1.0.0"
__all__ = [
    "PhysicalRayTracer",
    "BaseMaterial",
    "DielectricMaterial",
    "ConductorMaterial",
    "RoughDielectricMaterial",
    "BaseRefractiveField",
    "ConstantField",
    "FunctionField",
    "GridField",
    "AnisotropicField",
    "generate_rays",
    "save_trace_output_to_npz",
    "export_ray_history_to_stl",
]