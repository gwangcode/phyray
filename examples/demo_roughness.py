"""Demo: Effect of surface roughness on reflectance."""
import torch
import trimesh

from phyray import PhysicalRayTracer, generate_rays


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0
    face_mats[:, 1] = 1

    # Sweep roughness
    for roughness in [0.0, 0.05, 0.1, 0.2, 0.5]:
        materials_cfg = {
            0: {"n": 1.0, "k": 0.0, "roughness": 0.0},
            1: {"n": 1.5, "k": 0.0, "roughness": roughness},
        }

        rays = generate_rays(
            source_type="grid", num_rays=16,
            origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(-0.5, 0.5, -0.5, 0.5),
            polarization="s", wavelength_nm=550.0, device=device,
        )

        tracer = PhysicalRayTracer(verts, faces, face_mats,
                                   materials_cfg, device)
        out = tracer.trace(rays, max_steps=40)

        R = out["reflectivity"][:, 1, 0].mean().item()
        print(f"roughness = {roughness:.2f}  ->  R = {R:.6f}")


if __name__ == "__main__":
    main()