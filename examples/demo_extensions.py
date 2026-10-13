"""
demo_extensions.py
==================
Demonstrate PhyRay extension modules.

Cases:
    1. Thin-film interference: 4-layer anti-reflection coating.
    2. Dispersion: BK7 vs SF11 chromatic aberration comparison.
    3. BRDF: GGX vs Lambertian surface reflectance.
    4. Scene: assembling 3 meshes with different materials.
"""
import math
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt

from phyray_ext import thin_film, dispersion, brdf, scene
from phyray import PhysicalRayTracer, generate_rays


def case_1_thin_film():
    print("\n" + "=" * 70)
    print("Case 1: Thin-film interference (anti-reflection coating)")
    print("=" * 70)

    # 4-layer AR coating: air | TiO2 | SiO2 | TiO2 | SiO2 | glass
    stack = thin_film.ThinFilmMaterial(
        n_list=[1.0, 2.35, 1.45, 2.35, 1.45, 1.5],
        k_list=[0.0] * 6,
        d_list=[80.0, 120.0, 80.0, 120.0],
        wavelength_range=(400.0, 700.0),
    )

    # Sweep wavelength at normal incidence
    wavelengths = np.linspace(400, 700, 60)
    R_s = [stack.reflectance(wl, 0.0, "s") for wl in wavelengths]
    R_p = [stack.reflectance(wl, 0.0, "p") for wl in wavelengths]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wavelengths, R_s, 'b-', label='s-pol')
    ax.plot(wavelengths, R_p, 'r--', label='p-pol')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("4-layer AR coating: R(λ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_thin_film_ar.png", dpi=150)
    plt.close()
    print("  Saved: fig_thin_film_ar.png")


def case_2_dispersion():
    print("\n" + "=" * 70)
    print("Case 2: Dispersion (BK7 vs SF11)")
    print("=" * 70)

    bk7 = dispersion.BK7()
    sf11 = dispersion.SF11()

    wl = torch.linspace(400.0, 700.0, 100)
    n_bk7 = bk7.n_fn(wl).numpy()
    n_sf11 = sf11.n_fn(wl).numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wl.numpy(), n_bk7, 'b-', label='BK7')
    ax.plot(wl.numpy(), n_sf11, 'r-', label='SF11')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive index n")
    ax.set_title("Dispersion curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_dispersion.png", dpi=150)
    plt.close()
    print("  Saved: fig_dispersion.png")


def case_3_brdf():
    print("\n" + "=" * 70)
    print("Case 3: BRDF comparison (Lambertian vs GGX)")
    print("=" * 70)

    lam = brdf.LambertianBRDF(albedo=0.8)
    ggx = brdf.GGXBRDF(F0=0.04, alpha=0.2, kd=0.1)

    N = torch.tensor([[0.0, 0.0, 1.0]])
    V = torch.tensor([[0.0, 0.3, 0.95]])
    V = V / V.norm()

    thetas = np.linspace(-80, 80, 50)
    vals_lam = []
    vals_ggx = []
    for t in thetas:
        L = torch.tensor([[math.sin(math.radians(t)),
                           0.0,
                           math.cos(math.radians(t))]])
        vals_lam.append(lam.eval(N, V, L).item())
        vals_ggx.append(ggx.eval(N, V, L).item())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(thetas, vals_lam, 'b-', label='Lambertian')
    ax.semilogy(thetas, vals_ggx, 'r-', label='GGX (α=0.2)')
    ax.set_xlabel("L angle (deg)")
    ax.set_ylabel("BRDF value")
    ax.set_title("BRDF lobe shapes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_brdf.png", dpi=150)
    plt.close()
    print("  Saved: fig_brdf.png")


def case_4_scene():
    print("\n" + "=" * 70)
    print("Case 4: Scene assembly (3 objects)")
    print("=" * 70)

    s = scene.Scene()
    s.add(trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
          material_id=1, name="cube",
          material_cfg={"n": 1.5, "k": 0.0, "roughness": 0.0})
    s.add(trimesh.creation.icosphere(radius=0.5),
          material_id=2, name="sphere",
          material_cfg={"n": 1.7, "k": 0.0, "roughness": 0.0})
    s.add(trimesh.creation.cylinder(radius=0.3, height=1.0),
          material_id=3, name="cylinder",
          material_cfg={"n": 2.0, "k": 0.1, "roughness": 0.05})

    s.summary()

    device = torch.device("cpu")
    verts, faces, face_mats = s.compile(device)
    materials_cfg = s.materials_cfg()

    print(f"\n  Compiled: {verts.shape[0]} verts, {faces.shape[0]} faces")
    print(f"  Materials: {list(materials_cfg.keys())}")

    # Trace a few rays
    tracer = PhysicalRayTracer(verts, faces, face_mats, materials_cfg, device)
    rays = generate_rays(source_type="grid", num_rays=16,
                         origin=(0, 0, -3), direction=(0, 0, 1),
                         bounds=(-1.5, 1.5, -1.5, 1.5),
                         polarization="random", device=device)
    with torch.no_grad():
        out = tracer.trace(rays, max_steps=100, ds_init=0.02)
    print(f"  Energy conserved: {out['energy_conserved']}")


def main():
    case_1_thin_film()
    case_2_dispersion()
    case_3_brdf()
    case_4_scene()
    print("\n" + "=" * 70)
    print("All extension demos completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()