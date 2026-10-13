"""
phyray_ext — PhyRay Extensions
==============================

Optional extension modules for PhyRay. These do NOT modify the core
PhyRay package. Import only what you need.

Modules:
    thin_film    — Multilayer thin-film interference
    dispersion   — Cauchy / Sellmeier / Drude dispersion models
    brdf         — GGX / Cook-Torrance / Lambertian / Phong BRDFs
    scene        — Scene graph for complex assemblies
"""

from . import thin_film
from . import dispersion
from . import brdf
from . import scene

__version__ = "1.0.0"
__all__ = ["thin_film", "dispersion", "brdf", "scene"]