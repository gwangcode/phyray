"""
tests/test_mbvh.py
==================
Unit tests for the SAH + MBVH BVH.

Test coverage:
    1. Correctness vs brute-force on a sphere mesh.
    2. No missing triangles (tetrahedron).
    3. Miss rays (no hit).
    4. Parallel rays (same direction).
    5. Zero-direction rays (numerical safety).
    6. Many-ray stress test (10k rays).
    7. allowed_face_mask filtering.
    8. prev_face_ids exclusion (anti-self-intersection).

Run:
    pytest tests/test_mbvh.py -v
"""
import math
import numpy as np
import torch
import trimesh
import pytest

from phyray.bvh import BVHMeshIntersector


# ==============================================================================
# Reference implementation (brute force)
# ==============================================================================

def brute_force_intersect(triangles, origins, dirs, eps=1e-4):
    """
    Brute-force ray-triangle intersection (Möller-Trumbore).

    Args:
        triangles: [F, 3, 3]
        origins:   [N, 3]
        dirs:      [N, 3]
    Returns:
        (hit_t [N], hit_face [N])
    """
    N = origins.shape[0]
    F = triangles.shape[0]
    hit_t = torch.full((N,), float("inf"))
    hit_f = torch.full((N,), -1, dtype=torch.long)

    for i in range(N):
        o = origins[i]
        d = dirs[i]
        for j in range(F):
            v0, v1, v2 = triangles[j]
            e1 = v1 - v0
            e2 = v2 - v0
            p = torch.cross(d, e2, dim=-1)
            det = (e1 * p).sum()
            if abs(det) < 1e-10:
                continue
            inv_det = 1.0 / det
            tv = o - v0
            u = (tv * p).sum() * inv_det
            if u < 0 or u > 1:
                continue
            q = torch.cross(tv, e1, dim=-1)
            v = (d * q).sum() * inv_det
            if v < 0 or u + v > 1:
                continue
            t = (e2 * q).sum() * inv_det
            if t > eps and t < hit_t[i]:
                hit_t[i] = t
                hit_f[i] = j
    return hit_t, hit_f


# ==============================================================================
# Test 1: Correctness on a sphere mesh
# ==============================================================================

def test_mbvh_matches_brute_force_on_sphere():
    """Random rays through a sphere: MBVH must match brute-force distances."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Sphere with ~1280 triangles
    sphere = trimesh.creation.icosphere(subdivisions=3)
    verts = torch.tensor(sphere.vertices, dtype=torch.float32)
    faces = torch.tensor(sphere.faces, dtype=torch.long)

    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    # Random rays from outside
    N = 100
    origins = torch.randn(N, 3) * 3.0
    dirs = torch.randn(N, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    dist_mbvh, face_mbvh, _ = inter.intersect(origins, dirs)

    triangles = verts[faces]
    dist_bf, face_bf = brute_force_intersect(triangles, origins, dirs)

    # Both must find hits on the same set of rays
    hit_mbvh = dist_mbvh < float("inf")
    hit_bf = dist_bf < float("inf")
    assert torch.equal(hit_mbvh, hit_bf), \
        f"Hit sets differ: {hit_mbvh.sum()} vs {hit_bf.sum()}"

    # Distances must match to within epsilon
    if hit_mbvh.any():
        diff = (dist_mbvh[hit_mbvh] - dist_bf[hit_mbvh]).abs().max().item()
        assert diff < 1e-3, f"Distance mismatch: max diff = {diff}"


# ==============================================================================
# Test 2: No missing triangles (small mesh)
# ==============================================================================

def test_mbvh_reaches_all_triangles():
    """Every triangle of a tetrahedron must be hittable."""
    torch.manual_seed(0)

    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=torch.long)

    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    # Aim rays at the centroid of each triangle from the outside
    tri_centroids = verts[faces].mean(dim=1)   # [4, 3]
    tri_verts = verts[faces]                   # [4, 3, 3]
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    normals = torch.cross(e1, e2, dim=-1)
    normals = normals / normals.norm(dim=-1, keepdim=True)

    origins = tri_centroids + normals * 1.0
    dirs = -normals

    dist, face_ids, _ = inter.intersect(origins, dirs)

    for i in range(4):
        assert face_ids[i].item() >= 0, f"Triangle {i} was never hit"
        assert dist[i].item() < float("inf")
        assert dist[i].item() > 0


# ==============================================================================
# Test 3: Miss rays
# ==============================================================================

def test_mbvh_miss():
    """Rays that don't intersect anything return -1 / inf."""
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    # Rays pointing away
    origins = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    dirs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    dist, face_ids, _ = inter.intersect(origins, dirs)
    assert (face_ids == -1).all()
    assert (dist == float("inf")).all()


# ==============================================================================
# Test 4: Parallel rays
# ==============================================================================

def test_mbvh_parallel_rays():
    """A batch of parallel rays must all hit the front face of a box."""
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    # 10 rays from z = -3, all along +z, spread in x
    N = 10
    x = torch.linspace(-0.8, 0.8, N)
    origins = torch.stack([x, torch.zeros(N), torch.full((N,), -3.0)], dim=-1)
    dirs = torch.tensor([0.0, 0.0, 1.0]).expand(N, 3).contiguous()

    dist, face_ids, _ = inter.intersect(origins, dirs)

    # All rays should hit the front face (z = -1) at distance ~2
    assert (face_ids >= 0).all()
    expected_dist = 2.0
    diff = (dist - expected_dist).abs().max().item()
    assert diff < 1e-4, f"Distance diff = {diff}"


# ==============================================================================
# Test 5: Zero-direction rays (numerical safety)
# ==============================================================================

def test_mbvh_zero_direction():
    """Zero-length directions must not crash."""
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    origins = torch.tensor([[0.0, 0.0, -2.0]])
    dirs = torch.tensor([[0.0, 0.0, 0.0]])   # zero direction

    # Should not crash; behavior is implementation-defined
    dist, face_ids, _ = inter.intersect(origins, dirs)
    assert dist.shape == (1,)
    assert face_ids.shape == (1,)


# ==============================================================================
# Test 6: Many-ray stress test
# ==============================================================================

def test_mbvh_10k_rays():
    """10k random rays against a sphere — exercise traversal stability."""
    torch.manual_seed(123)
    sphere = trimesh.creation.icosphere(subdivisions=3)
    verts = torch.tensor(sphere.vertices, dtype=torch.float32)
    faces = torch.tensor(sphere.faces, dtype=torch.long)
    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    N = 10000
    origins = torch.randn(N, 3) * 3.0
    dirs = torch.randn(N, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    dist, face_ids, _ = inter.intersect(origins, dirs)

    # No NaNs
    assert not torch.isnan(dist).any()
    assert not torch.isnan(face_ids.float()).any()

    # All hits must have valid face IDs
    hit_mask = dist < float("inf")
    assert (face_ids[hit_mask] >= 0).all()
    assert (face_ids[hit_mask] < faces.shape[0]).all()


# ==============================================================================
# Test 7: allowed_face_mask filtering
# ==============================================================================

def test_mbvh_allowed_face_mask():
    """allowed_face_mask should exclude disallowed triangles."""
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    # Single ray hitting the front face
    origins = torch.tensor([[0.0, 0.0, -3.0]])
    dirs = torch.tensor([[0.0, 0.0, 1.0]])

    # All faces allowed
    dist_all, face_all, _ = inter.intersect(origins, dirs)
    assert face_all[0].item() >= 0

    # Disallow the face that would be hit
    mask = torch.ones(faces.shape[0], dtype=torch.bool)
    mask[face_all[0].item()] = False

    dist_filt, face_filt, _ = inter.intersect(
        origins, dirs, allowed_face_mask=mask)
    # The original hit face is now excluded; either a farther face is hit
    # or the ray misses entirely.
    if face_filt[0].item() >= 0:
        assert face_filt[0].item() != face_all[0].item()


# ==============================================================================
# Test 8: prev_face_ids exclusion
# ==============================================================================

def test_mbvh_prev_face_exclusion():
    """prev_face_ids should prevent self-intersection."""
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    inter = BVHMeshIntersector(verts, faces, torch.device("cpu"))

    origins = torch.tensor([[0.0, 0.0, -3.0]])
    dirs = torch.tensor([[0.0, 0.0, 1.0]])

    # First hit
    dist1, face1, _ = inter.intersect(origins, dirs)
    assert face1[0].item() >= 0

    # Now start from the hit point and exclude that face
    hit_pt = origins[0] + dirs[0] * dist1[0]
    origins2 = hit_pt.unsqueeze(0) + dirs * 1e-5
    prev_faces = face1.clone()

    dist2, face2, _ = inter.intersect(
        origins2, dirs, prev_face_ids=prev_faces)

    # The excluded face must not be returned as the hit
    assert face2[0].item() != face1[0].item()


# ==============================================================================
# Run all tests
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))