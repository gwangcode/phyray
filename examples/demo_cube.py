"""Demo: Fresnel reflection from a dielectric cube using physical materials."""
import torch
import trimesh

from phyray import PhysicalRayTracer, generate_rays, DielectricMaterial


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build a cube
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0   # outside: air
    face_mats[:, 1] = 1   # inside: glass

    # ★ Physical materials: only (n, k, roughness)
    materials_cfg = {
        0: {"n": 1.0, "k": 0.0, "roughness": 0.0},              # air
        1: {"n": 1.5, "k": 0.0, "roughness": 0.0},              # smooth glass
    }

    # Fire 16 rays from a grid
    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", wavelength_nm=550.0, device=device,
    )

    # Trace
    tracer = PhysicalRayTracer(verts, faces, face_mats, materials_cfg, device)
    out = tracer.trace(rays, max_steps=40)

    print(f"Energy conserved: {out['energy_conserved']}")
    print(f"R at first surface: {out['reflectivity'][0, 1, 0].item():.10f}")
    # Expected: ((1.0 - 1.5)/(1.0 + 1.5))^2 = 0.04


if __name__ == "__main__":
    main()