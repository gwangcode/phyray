"""
demo_polarization_full.py
=========================
Four-in-one demo showing PhyRay's polarization capabilities:

    Case 1: s-polarized light through a glass slab
    Case 2: p-polarized light through a glass slab
    Case 3: Circular polarization through a TIR prism
    Case 4: Birefringent GRIN medium — s vs p focus split

Run:
    python examples/demo_polarization_full.py
"""
import math
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt

from phyray import PhysicalRayTracer, generate_rays


def build_flat_slab(device):
    """A single flat glass slab (2 triangles)."""
    verts = torch.tensor([
        [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    face_mats = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
    return verts, faces, face_mats


def build_tir_prism(device):
    """A 45-degree prism for TIR demonstration.
    Light enters at normal incidence and hits the hypotenuse at 45° > critical angle.
    """
    # Prism: vertices at (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
    #               (-1,-1, 1), (1,-1, 1), (1,1, 1), (-1,1, 1)
    # We use a cube with a slanted face — simplest: a triangular prism extruded.
    # Actually we simplify: use a half-cube where the diagonal face acts as TIR mirror.
    verts = torch.tensor([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ], dtype=torch.float32, device=device)

    # Two cubes: simple, but we place the prism interior as material 1
    faces_list = [
        # Front face (z=-1)
        [0, 2, 1], [0, 3, 2],
        # Back face (z=+1)
        [4, 5, 6], [4, 6, 7],
        # Bottom (y=-1)
        [0, 1, 5], [0, 5, 4],
        # Top (y=+1)
        [3, 7, 6], [3, 6, 2],
        # Left (x=-1)
        [0, 4, 7], [0, 7, 3],
        # Right (x=+1)
        [1, 2, 6], [1, 6, 5],
    ]
    faces = torch.tensor(faces_list, dtype=torch.long, device=device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0  # outside: air
    face_mats[:, 1] = 1  # inside: glass
    return verts, faces, face_mats


def build_grin_cube(device):
    """A cube with birefringent GRIN inside."""
    cube = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0
    face_mats[:, 1] = 1
    return verts, faces, face_mats


# ==============================================================================
# Case 1 & 2: s and p polarization through a slab
# ==============================================================================

def case_1_2_s_and_p(device):
    print("\n" + "=" * 70)
    print("Case 1 & 2: s and p polarization through a glass slab")
    print("=" * 70)

    verts, faces, face_mats = build_flat_slab(device)
    materials_cfg = {
        0: {"n": 1.0, "k": 0.0, "roughness": 0.0},
        1: {"n": 1.5, "k": 0.0, "roughness": 0.0},
    }
    tracer = PhysicalRayTracer(verts, faces, face_mats, materials_cfg, device)

    # 60° incidence for visible s/p difference
    theta_deg = 60.0
    theta = math.radians(theta_deg)
    direction = (math.sin(theta), 0.0, -math.cos(theta))  # light goes -z
    origin = (0.0, 0.0, 1.0)

    for pol in ["s", "p"]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=origin, direction=direction,
            bounds=(-0.001, 0.001, -0.001, 0.001),
            polarization=pol, wavelength_nm=550.0, device=device)

        out = tracer.trace(rays, max_steps=10, fresnel=True)

        R = out["reflectivity"][0, 1, 0].item()
        print(f"  {pol}-polarized at {theta_deg}°: R = {R:.6f}")

    # Analytical reference
    n1, n2 = 1.0, 1.5
    cos_i = math.cos(theta)
    sin_t_sq = (n1/n2)**2 * (1 - cos_i**2)
    cos_t = math.sqrt(1 - sin_t_sq)
    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    print(f"  Analytical (Fresnel):  R_s = {r_s**2:.6f}   R_p = {r_p**2:.6f}")


# ==============================================================================
# Case 3: TIR produces circular polarization from linear input
# ==============================================================================

def case_3_tir(device):
    print("\n" + "=" * 70)
    print("Case 3: TIR converts 45° linear polarization to elliptical")
    print("=" * 70)

    # Flat slab (acts as mirror under TIR)
    verts, faces, face_mats = build_flat_slab(device)
    materials_cfg = {
        0: {"n": 1.5, "k": 0.0, "roughness": 0.0},   # incident medium: glass
        1: {"n": 1.0, "k": 0.0, "roughness": 0.0},   # outside: air
    }
    tracer = PhysicalRayTracer(verts, faces, face_mats, materials_cfg, device)

    # Incident angle beyond critical: sin(theta_c) = n2/n1 = 1/1.5, theta_c ≈ 41.8°
    theta_deg = 50.0
    theta = math.radians(theta_deg)
    direction = (math.sin(theta), 0.0, -math.cos(theta))

    # Incident polarization: 45° linear (equal s and p amplitude)
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 1.0), direction=direction,
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization=45.0, wavelength_nm=550.0, device=device)

    out = tracer.trace(rays, max_steps=10, fresnel=True)

    stokes_before = out["stokes"][0, 0, :].tolist()
    stokes_after = out["stokes"][0, 1, :].tolist()

    print(f"  Incident Stokes:  {[round(x, 4) for x in stokes_before]}")
    print(f"  After TIR:        {[round(x, 4) for x in stokes_after]}")
    print(f"  Incident DoP:     {out['dop'][0, 0, 0].item():.4f}")
    print(f"  After TIR DoP:    {out['dop'][0, 1, 0].item():.4f}")

    # Analytical TIR phase shift
    n1, n2 = 1.5, 1.0
    sin_i = math.sin(theta)
    sin_t_sq = (n1/n2)**2 * sin_i**2
    if sin_t_sq > 1:
        # TIR
        cos_i = math.cos(theta)
        tan_delta_s_2 = math.sqrt(sin_t_sq - 1) / (n1 * cos_i / n2 * (n1/n2))
        # Simplified formula
        delta_s = 2 * math.atan(math.sqrt(sin_i**2 - (n2/n1)**2) / math.cos(theta))
        delta_p = 2 * math.atan(n1**2 * math.sqrt(sin_i**2 - (n2/n1)**2) / (n2**2 * math.cos(theta)))
        phase_diff = delta_s - delta_p
        print(f"  Analytical phase difference (δ_s - δ_p) = {math.degrees(phase_diff):.2f}°")
        print(f"  → TIR rotates linear polarization into elliptical")
    else:
        print(f"  No TIR at this angle")


# ==============================================================================
# Case 4: Birefringent GRIN — s vs p focus split
# ==============================================================================

def case_4_birefringent_grin(device):
    print("\n" + "=" * 70)
    print("Case 4: Birefringent GRIN — s vs p focus split")
    print("=" * 70)

    verts, faces, face_mats = build_grin_cube(device)

    # Anisotropic GRIN field
    materials_cfg = {
        0: {"n": 1.0, "k": 0.0, "roughness": 0.0},   # air
        1: {
            "n_fn": lambda x, y, z, wl: 1.5 - 0.2 * (x**2 + y**2),
            "delta_n_fn": lambda x, y, z, wl: 0.02,   # ★ birefringence
            "axis": (1.0, 0.0, 0.0),                   # ★ optical axis
            "k_fn": lambda x, y, z, wl: 0.0,
            "roughness": 0.0,
        },
    }

    tracer = PhysicalRayTracer(verts, faces, face_mats, materials_cfg, device)

    # s and p polarized parallel beams
    results = {}
    for pol in ["s", "p"]:
        rays = generate_rays(
            source_type="grid", num_rays=64,
            origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0),
            bounds=(-0.8, 0.8, -0.8, 0.8),
            polarization=pol, wavelength_nm=550.0, device=device)

        out = tracer.trace(rays, max_steps=300, ds_init=0.02, verbose=False)
        results[pol] = out

        final_pos = out["positions"][:, -1, :]
        focus_x = final_pos[:, 0].mean().item()
        focus_y = final_pos[:, 1].mean().item()
        print(f"  {pol}-polarized focus: ({focus_x:.6f}, {focus_y:.6f})")

    # Focus split
    f_s = results["s"]["positions"][:, -1, :].mean(dim=0)
    f_p = results["p"]["positions"][:, -1, :].mean(dim=0)
    split = torch.norm(f_s - f_p).item()
    print(f"\n  ★ Focus split (s vs p): {split:.6e}")
    print(f"  → Due to birefringence Δn = 0.02")

    # Plot rays
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (pol, out) in enumerate(results.items()):
        pos = out["positions"].cpu().numpy()
        for j in range(pos.shape[0]):
            axes[i].plot(pos[j, :, 0], pos[j, :, 2], alpha=0.3)
        axes[i].set_title(f"{pol}-polarized")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("z")
        axes[i].grid(True)
        axes[i].set_xlim(-1.5, 1.5)
        axes[i].set_ylim(-2.5, 2.5)
    plt.tight_layout()
    plt.savefig("fig_case4_birefringent_grin.png", dpi=150)
    print(f"  Saved: fig_case4_birefringent_grin.png")
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    case_1_2_s_and_p(device)
    case_3_tir(device)
    case_4_birefringent_grin(device)

    print("\n" + "=" * 70)
    print("All four cases completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()