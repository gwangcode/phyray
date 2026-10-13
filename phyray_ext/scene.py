"""
phyray_ext/scene.py
===================
Scene graph for multi-object assemblies.

Assembles multiple meshes into a single (vertices, faces, face_materials)
tuple that PhyRay can consume directly.

Features:
    - Add multiple meshes with individual materials.
    - Automatic material ID assignment.
    - Bounding-box computation.
    - Import from trimesh.Trimesh.
"""
import numpy as np
import torch
import trimesh


class SceneNode:
    """A single mesh + material entry."""

    def __init__(self, mesh, material_id, name="node"):
        self.mesh = mesh
        self.material_id = material_id
        self.name = name
        self._bmin = mesh.vertices.min(axis=0)
        self._bmax = mesh.vertices.max(axis=0)

    def bounds(self):
        return self._bmin, self._bmax


class Scene:
    """
    Build a complete scene from multiple meshes.

    Usage:
        scene = Scene()
        scene.add(trimesh.creation.box(extents=[1,1,1]), material_id=1)
        scene.add(trimesh.creation.icosphere(), material_id=2)

        verts, faces, face_mats = scene.compile(device)
        # → pass directly to PhysicalRayTracer
    """

    def __init__(self):
        self.nodes = []
        self.materials = {}   # {mat_id: material_cfg}

    def add(self, mesh, material_id, material_cfg=None, name="node"):
        """
        Add a mesh with a specific material.

        Args:
            mesh: trimesh.Trimesh or (vertices, faces) tuple.
            material_id: int (must be > 0).
            material_cfg: optional dict for materials_cfg registration.
            name: label for debugging.
        """
        if material_id <= 0:
            raise ValueError("material_id must be > 0 (0 is reserved for air)")

        if isinstance(mesh, trimesh.Trimesh):
            m = mesh
        else:
            v, f = mesh
            m = trimesh.Trimesh(vertices=v, faces=f)

        self.nodes.append(SceneNode(m, material_id, name))

        if material_cfg is not None:
            self.materials[material_id] = material_cfg

    def compile(self, device):
        """
        Merge all meshes into a single (vertices, faces, face_materials) tuple.

        Note: this does NOT compute interior/exterior split. All faces are
        marked with [outside_material, inside_material] = [0, node_material_id].

        Returns:
            (verts [N, 3], faces [F, 3], face_mats [F, 2])
        """
        all_verts = []
        all_faces = []
        all_mats = []
        offset = 0

        for node in self.nodes:
            v = torch.tensor(node.mesh.vertices, dtype=torch.float32)
            f = torch.tensor(node.mesh.faces, dtype=torch.long)
            # Shift face indices by current offset
            f = f + offset

            # Face materials: [outside=0, inside=node.material_id]
            fm = torch.zeros((f.shape[0], 2), dtype=torch.long)
            fm[:, 0] = 0
            fm[:, 1] = node.material_id

            all_verts.append(v)
            all_faces.append(f)
            all_mats.append(fm)
            offset += v.shape[0]

        verts = torch.cat(all_verts, dim=0).to(device)
        faces = torch.cat(all_faces, dim=0).to(device)
        face_mats = torch.cat(all_mats, dim=0).to(device)

        return verts, faces, face_mats

    def materials_cfg(self, outside_material=0):
        """
        Return a full materials_cfg dict including the outside (air).
        """
        cfg = {
            outside_material: {"n": 1.0, "k": 0.0, "roughness": 0.0},
        }
        cfg.update(self.materials)
        return cfg

    def bounds(self):
        """Union of all node bounds."""
        if not self.nodes:
            return None, None
        mins = np.array([n._bmin for n in self.nodes])
        maxs = np.array([n._bmax for n in self.nodes])
        return mins.min(axis=0), maxs.max(axis=0)

    def summary(self):
        """Print scene summary."""
        print(f"Scene: {len(self.nodes)} nodes")
        bmin, bmax = self.bounds()
        if bmin is not None:
            print(f"  Bounds: min={bmin.tolist()}, max={bmax.tolist()}")
        for i, node in enumerate(self.nodes):
            print(f"  [{i}] {node.name}: "
                  f"{node.mesh.vertices.shape[0]} verts, "
                  f"{node.mesh.faces.shape[0]} faces, "
                  f"material={node.material_id}")