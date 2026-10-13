"""Physical ray tracer for PhyRay.

All interface physics is derived from (n, k, roughness):
    - Complex Fresnel coefficients (handles absorbing media + TIR phase)
    - Beer-Lambert volumetric absorption (in GRIN propagation)
    - Beckmann-Kirchhoff rough-surface scattering
    - Complex Jones-Mueller polarization tracking (preserves TIR phase)
    - Polarization-weighted reflectance (uses incident Stokes)
"""
import math
import torch

from .bvh import BVHMeshIntersector
from .materials import (
    build_material,
    complex_fresnel,
    fresnel_reflectance,
    fresnel_transmittance_flow,
    beer_lambert,
    roughness_specular_factor,
    roughness_diffuse_fraction,
)
from .fields import BaseRefractiveField, build_field
from .polarization import (
    _choose_perpendicular,
    rotate_stokes_reference,
    apply_jones_mueller,
    randomize_stokes_diffuse,
    stokes_to_pol_vector,
    degree_of_polarization,
)


class PhysicalRayTracer:
    """GPU-accelerated ray tracer with physically-based material model.

    Materials are defined by (n, k, roughness).
    All interface reflection/transmission and volumetric absorption are
    computed from these parameters automatically.
    """

    def __init__(self, vertices, faces, face_materials, materials_cfg, device):
        self.device = device
        self.intersector = BVHMeshIntersector(vertices, faces, device)
        self.face_materials = face_materials.to(dtype=torch.long, device=device)
        self.materials_cfg = materials_cfg
        self.num_faces = faces.shape[0]

        # Medium-to-face masks
        self.medium_to_faces_mask = {}
        all_medium_ids = set()
        for f_idx in range(self.num_faces):
            all_medium_ids.add(self.face_materials[f_idx, 0].item())
            all_medium_ids.add(self.face_materials[f_idx, 1].item())
        for med_id in all_medium_ids:
            mask = ((self.face_materials[:, 0] == med_id) |
                    (self.face_materials[:, 1] == med_id)).to(device)
            self.medium_to_faces_mask[med_id] = mask

        # Build materials and fields
        self.materials = {}
        self.fields = {}
        self.is_grin = {}

        for mat_id, cfg in materials_cfg.items():
            self.materials[mat_id] = build_material(cfg)
            field_cfg = dict(cfg)
            field_cfg["material"] = self.materials[mat_id]
            self.fields[mat_id] = build_field(field_cfg)
            has_fn = any(k in cfg for k in ["n_fn", "delta_n_fn", "values"])
            self.is_grin[mat_id] = has_fn

    def trace(self, ray_dict, max_steps=200, ds_init=0.02, ds_min=1e-3,
              ds_max=0.1, rk_tol=1e-4, verbose=False,
              rr_threshold=0.0):
        """
        Trace rays through the scene.

        Args:
            ...
            rr_threshold: if > 0, enable Russian roulette on the Monte
                Carlo energy weight when it drops below this threshold.

        Returns:
            dict with (in addition to previous keys):
                "energy": [N, S+1] Monte Carlo path weight per ray per step.
                "energy_conserved": True only if:
                    (a) 0 <= energy <= 1
                    (b) energy monotone non-increasing
                    (c) inactive rays have zero weight
        """
        # ---- Initialize ----
        curr_origins = ray_dict["origins"].clone()
        curr_dirs = ray_dict["dirs"].clone()
        curr_wls = ray_dict["wavelengths"].clone()
        curr_stokes = ray_dict["stokes"].clone()
        curr_s_ref = ray_dict["s_ref"].clone()
        curr_p_ref = ray_dict["p_ref"].clone()

        N = curr_origins.shape[0]
        curr_medium_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        prev_hit_faces = torch.full((N,), -1, dtype=torch.long, device=self.device)
        active_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # ★ Monte Carlo path weight
        curr_weight = torch.ones(N, device=self.device)

        ds_per_ray = torch.full((N,), float(ds_init), device=self.device)

        # History
        hist_pos = [curr_origins.clone()]
        hist_stokes = [curr_stokes.clone()]
        hist_active = [active_mask.clone()]
        hist_refl = [torch.zeros((N, 1), device=self.device)]
        hist_trans = [torch.zeros((N, 1), device=self.device)]
        hist_abs = [torch.zeros((N, 1), device=self.device)]
        hist_n1 = [torch.ones((N, 1), device=self.device)]
        hist_n2 = [torch.ones((N, 1), device=self.device)]
        hist_k1 = [torch.zeros((N, 1), device=self.device)]
        hist_k2 = [torch.zeros((N, 1), device=self.device)]
        hist_pol_deg = [0.5 * torch.atan2(curr_stokes[:, 2:3],
                                          curr_stokes[:, 1:2]) * (180.0 / math.pi)]
        hist_energy = [curr_weight.clone()]
        hist_rk_err = [torch.zeros(N, device=self.device)]
        hist_ds = [ds_per_ray.clone()]
        hist_dop = [degree_of_polarization(curr_stokes).clone()]

        step = 0
        while step < max_steps:
            if not active_mask.any():
                break

            # ---------- 1. Intersection ----------
            hit_dist = torch.full((N,), float('inf'), device=self.device)
            hit_face_ids = torch.full((N,), -1, dtype=torch.long, device=self.device)
            hit_normals = torch.zeros((N, 3), device=self.device)

            unique_mediums = torch.unique(curr_medium_ids[active_mask])
            for med_id in unique_mediums:
                med_val = med_id.item()
                ray_mask = active_mask & (curr_medium_ids == med_id)
                if not ray_mask.any():
                    continue
                allowed_faces = self.medium_to_faces_mask.get(med_val, None)
                dist_m, face_ids_m, normals_m = self.intersector.intersect(
                    curr_origins[ray_mask], curr_dirs[ray_mask],
                    prev_face_ids=prev_hit_faces[ray_mask],
                    allowed_face_mask=allowed_faces)
                hit_dist[ray_mask] = dist_m
                hit_face_ids[ray_mask] = face_ids_m
                hit_normals[ray_mask] = normals_m

            # ---------- 2. Buffers ----------
            next_origins = curr_origins.clone()
            next_dirs = curr_dirs.clone()
            next_stokes = curr_stokes.clone()
            next_s_ref = curr_s_ref.clone()
            next_p_ref = curr_p_ref.clone()
            next_weight = curr_weight.clone()

            step_refl = torch.zeros((N, 1), device=self.device)
            step_trans = torch.zeros((N, 1), device=self.device)
            step_abs = torch.zeros((N, 1), device=self.device)
            step_n1 = torch.ones((N, 1), device=self.device)
            step_n2 = torch.ones((N, 1), device=self.device)
            step_k1 = torch.zeros((N, 1), device=self.device)
            step_k2 = torch.zeros((N, 1), device=self.device)
            step_rk_err = torch.zeros(N, device=self.device)

            # ---------- 3. GRIN stepping + Beer-Lambert ----
            is_in_grin = torch.zeros(N, dtype=torch.bool, device=self.device)
            for mid, is_grin in self.is_grin.items():
                if is_grin:
                    is_in_grin |= (curr_medium_ids == mid)
            grin_mask = active_mask & is_in_grin & (hit_dist > ds_per_ray)

            if grin_mask.any():
                pol_vec_all = stokes_to_pol_vector(curr_stokes, curr_s_ref, curr_p_ref)
                dop_all = degree_of_polarization(curr_stokes)

                grin_meds = torch.unique(curr_medium_ids[grin_mask])
                for g_med_id in grin_meds:
                    g_val = g_med_id.item()
                    sub_mask = grin_mask & (curr_medium_ids == g_med_id)
                    field = self.fields[g_val]

                    field.set_context(pol_vec_all[sub_mask], dop_all[sub_mask])

                    h_sub = ds_per_ray[sub_mask]

                    r_new, d_new, h_new, err_new, accepted = self._adaptive_step(
                        curr_origins[sub_mask], curr_dirs[sub_mask], field,
                        curr_wls[sub_mask], h_sub, rk_tol, ds_min, ds_max)

                    s_old = curr_s_ref[sub_mask]
                    s_proj = torch.sum(s_old * d_new, dim=-1, keepdim=True)
                    s_candidate = s_old - s_proj * d_new
                    s_candidate = s_candidate / (
                            torch.norm(s_candidate, dim=-1, keepdim=True) + 1e-12)
                    stokes_candidate = rotate_stokes_reference(
                        curr_stokes[sub_mask], s_old, s_candidate, d_new)

                    acc_3d = accepted.unsqueeze(-1)
                    next_origins[sub_mask] = torch.where(
                        acc_3d, r_new, curr_origins[sub_mask])
                    next_dirs[sub_mask] = torch.where(
                        acc_3d, d_new, curr_dirs[sub_mask])
                    next_s_ref[sub_mask] = torch.where(
                        acc_3d, s_candidate, curr_s_ref[sub_mask])
                    next_p_ref[sub_mask] = torch.cross(
                        next_dirs[sub_mask], next_s_ref[sub_mask], dim=-1)
                    next_stokes[sub_mask] = torch.where(
                        acc_3d, stokes_candidate, curr_stokes[sub_mask])

                    # ★ Beer-Lambert volumetric absorption
                    # 这里用介质 (n, k) 算出每步的透过率，并同时更新
                    # 光线的蒙特卡洛权重。
                    _, k_medium = field.get_complex_index(
                        curr_origins[sub_mask], curr_wls[sub_mask])
                    atten = beer_lambert(k_medium, h_sub.unsqueeze(-1),
                                         curr_wls[sub_mask])
                    # 正常值 >= eps；用 clamp 保护
                    atten = atten.clamp(0.0, 1.0)
                    atten_flat = torch.where(
                        acc_3d, atten, torch.ones_like(atten))
                    next_weight[sub_mask] = next_weight[sub_mask] * atten_flat.squeeze(-1)

                    step_rk_err[sub_mask] = err_new
                    ds_per_ray[sub_mask] = h_new

                    field.clear_context()

                    if verbose and step % 5 == 0:
                        print(f"  [GRIN step={step} med={g_val}] "
                              f"ds=[{h_new.min().item():.5f}, "
                              f"{h_new.max().item():.5f}], "
                              f"acc={accepted.float().mean().item():.1%}")

            # ---------- 4. Surface interaction ----------
            mesh_hit_mask = active_mask & (~grin_mask) & (hit_face_ids >= 0)
            if mesh_hit_mask.any():
                hit_pts = curr_origins[mesh_hit_mask] + \
                          hit_dist[mesh_hit_mask].unsqueeze(-1) * \
                          curr_dirs[mesh_hit_mask]
                next_origins[mesh_hit_mask] = hit_pts

                normals = hit_normals[mesh_hit_mask]
                dirs_in = curr_dirs[mesh_hit_mask]
                stokes_in = curr_stokes[mesh_hit_mask]
                s_ref_in = curr_s_ref[mesh_hit_mask]

                cos_theta_i_raw = -torch.sum(dirs_in * normals, dim=-1,
                                             keepdim=True)
                is_front = cos_theta_i_raw > 0.0
                oriented_normals = torch.where(is_front, normals, -normals)
                cos_theta_i = torch.clamp(torch.abs(cos_theta_i_raw), 0.0, 1.0)

                faces_hit = hit_face_ids[mesh_hit_mask]
                mat_front = self.face_materials[faces_hit, 0]
                mat_back = self.face_materials[faces_hit, 1]
                n1_ids = torch.where(is_front.squeeze(-1), mat_front, mat_back)
                n2_ids = torch.where(is_front.squeeze(-1), mat_back, mat_front)

                wls_hit = curr_wls[mesh_hit_mask]
                cos_hit = cos_theta_i

                # Incident-plane s basis
                cross_plane = torch.cross(dirs_in, oriented_normals, dim=-1)
                norm_plane = torch.norm(cross_plane, dim=-1, keepdim=True)
                degenerate = norm_plane.squeeze(-1) < 1e-8
                s_plane = torch.where(degenerate.unsqueeze(-1), s_ref_in,
                                      cross_plane / (norm_plane + 1e-12))

                stokes_plane = rotate_stokes_reference(
                    stokes_in, s_ref_in, s_plane, dirs_in)
                S1_plane = stokes_plane[:, 1:2]

                # Materials
                n1_vals = torch.zeros_like(wls_hit)
                k1_vals = torch.zeros_like(wls_hit)
                n2_vals = torch.zeros_like(wls_hit)
                k2_vals = torch.zeros_like(wls_hit)
                rough1 = torch.zeros_like(wls_hit)
                rough2 = torch.zeros_like(wls_hit)

                all_mats = torch.unique(torch.cat([n1_ids, n2_ids]))
                for m_id in all_mats:
                    m_val = m_id.item()
                    mat = self.materials[m_val]

                    m1 = (n1_ids == m_val)
                    if m1.any():
                        n1_vals[m1], k1_vals[m1] = mat.get_complex_index(wls_hit[m1])
                        rough1[m1] = mat.get_roughness(wls_hit[m1])
                    m2 = (n2_ids == m_val)
                    if m2.any():
                        n2_vals[m2], k2_vals[m2] = mat.get_complex_index(wls_hit[m2])
                        rough2[m2] = mat.get_roughness(wls_hit[m2])

                n1_vals = torch.clamp(n1_vals, min=0.01)
                n2_vals = torch.clamp(n2_vals, min=0.01)
                k1_vals = torch.clamp(k1_vals, min=0.0)
                k2_vals = torch.clamp(k2_vals, min=0.0)
                rough1 = torch.clamp(rough1, min=0.0)
                rough2 = torch.clamp(rough2, min=0.0)

                # Complex Fresnel
                r_s_cplx, r_p_cplx, t_s_cplx, t_p_cplx = complex_fresnel(
                    n1_vals, k1_vals, n2_vals, k2_vals, cos_hit)

                abs_R_s = torch.abs(r_s_cplx) ** 2
                abs_R_p = torch.abs(r_p_cplx) ** 2

                # Polarization-weighted Fresnel
                w_s = 0.5 * (1.0 + S1_plane)
                w_p = 0.5 * (1.0 - S1_plane)
                R_eff = w_s * abs_R_s + w_p * abs_R_p

                # Transmittance
                T_s, T_p, T_avg = fresnel_transmittance_flow(
                    n1_vals, k1_vals, n2_vals, k2_vals, cos_hit)

                # Roughness
                wavelength_m = wls_hit * 1e-9
                rough_avg = 0.5 * (rough1 + rough2)
                spec_factor = roughness_specular_factor(
                    rough_avg, wavelength_m, cos_hit)
                diffuse_frac = roughness_diffuse_fraction(
                    rough_avg, wavelength_m, cos_hit)

                # Probability split
                P_specular = R_eff * spec_factor
                P_diffuse = R_eff * diffuse_frac
                P_refract = T_avg

                total = P_specular + P_diffuse + P_refract
                total_safe = torch.clamp(total, min=1e-12)
                P_specular = P_specular / total_safe
                P_diffuse = P_diffuse / total_safe
                P_refract = P_refract / total_safe
                P_absorb = torch.zeros_like(P_specular)

                step_refl[mesh_hit_mask] = P_specular + P_diffuse
                step_trans[mesh_hit_mask] = P_refract
                step_abs[mesh_hit_mask] = P_absorb
                step_n1[mesh_hit_mask] = n1_vals
                step_n2[mesh_hit_mask] = n2_vals
                step_k1[mesh_hit_mask] = k1_vals
                step_k2[mesh_hit_mask] = k2_vals

                # Directions
                eta = n1_vals / (n2_vals + 1e-12)
                sin2_t = torch.clamp(1.0 - cos_hit ** 2, min=0.0)
                sin2_tt = (eta ** 2) * sin2_t
                is_tir = sin2_tt > 1.0
                cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_tt, min=0.0))

                reflect_dirs = dirs_in + 2.0 * cos_hit * oriented_normals
                reflect_dirs = reflect_dirs / torch.norm(
                    reflect_dirs, dim=-1, keepdim=True)
                refract_dirs = eta * dirs_in + (
                        eta * cos_hit - cos_theta_t) * oriented_normals
                refract_dirs = refract_dirs / torch.norm(
                    refract_dirs, dim=-1, keepdim=True)

                # Diffuse sampling
                num_hits = hit_pts.shape[0]
                r_sq = torch.rand(num_hits, 1, device=self.device)
                sin_theta_d = torch.sqrt(r_sq)
                cos_theta_d = torch.sqrt(1.0 - r_sq)
                phi_d = 2.0 * math.pi * torch.rand(num_hits, 1, device=self.device)
                ref_vecs = torch.zeros_like(oriented_normals)
                ref_vecs[:, 2] = 1.0
                ref_vecs[torch.abs(oriented_normals[:, 2]) > 0.999] = \
                    torch.tensor([0.0, 1.0, 0.0], device=self.device)
                local_u = torch.cross(ref_vecs, oriented_normals, dim=-1)
                local_u = local_u / torch.norm(local_u, dim=-1, keepdim=True)
                local_v = torch.cross(oriented_normals, local_u, dim=-1)
                diffuse_dirs = (sin_theta_d * torch.cos(phi_d) * local_u +
                                sin_theta_d * torch.sin(phi_d) * local_v +
                                cos_theta_d * oriented_normals)
                diffuse_dirs = diffuse_dirs / torch.norm(
                    diffuse_dirs, dim=-1, keepdim=True)

                # Monte Carlo event selection
                rand_val = torch.rand_like(P_specular)
                is_diff = rand_val < P_diffuse
                is_spec = (~is_diff) & (rand_val < (P_diffuse + P_specular))
                is_refr = ~(is_diff | is_spec)

                # ★ ★ ★ Update Monte Carlo path weight ★ ★ ★
                branch_prob = (
                        P_diffuse * is_diff.float() +
                        P_specular * is_spec.float() +
                        P_refract * is_refr.float()
                )
                next_weight[mesh_hit_mask] = next_weight[mesh_hit_mask] * branch_prob.squeeze(-1)

                # ---- Russian roulette (optional) ----
                if rr_threshold > 0.0:
                    needs_rr = next_weight[mesh_hit_mask] < rr_threshold
                    surv_prob = torch.clamp(
                        next_weight[mesh_hit_mask] / rr_threshold, 0.0, 1.0)
                    surv = torch.rand_like(next_weight[mesh_hit_mask]) < surv_prob
                    boosted = torch.where(surv, torch.full_like(
                        next_weight[mesh_hit_mask], rr_threshold),
                                          next_weight[mesh_hit_mask])
                    next_weight[mesh_hit_mask] = torch.where(needs_rr, boosted,
                                                             next_weight[mesh_hit_mask])
                    rr_kill = needs_rr & (~surv)

                # Complex Jones-Mueller polarization
                stokes_refl_plane = apply_jones_mueller(
                    stokes_plane, r_s_cplx, r_p_cplx)
                stokes_refr_plane = apply_jones_mueller(
                    stokes_plane, t_s_cplx, t_p_cplx)
                stokes_diff_plane = randomize_stokes_diffuse(stokes_plane)

                stokes_refl_new = rotate_stokes_reference(
                    stokes_refl_plane, s_plane, s_plane, reflect_dirs)
                stokes_refr_new = rotate_stokes_reference(
                    stokes_refr_plane, s_plane, s_plane, refract_dirs)

                new_d = torch.where(is_diff, diffuse_dirs,
                                    torch.where(is_spec, reflect_dirs,
                                                refract_dirs))
                s_diff_default = _choose_perpendicular(diffuse_dirs)
                new_s_ref = torch.where(is_diff, s_diff_default,
                                        torch.where(is_spec, s_plane, s_plane))
                new_stokes = torch.where(is_diff, stokes_diff_plane,
                                         torch.where(is_spec, stokes_refl_new,
                                                     stokes_refr_new))

                next_dirs[mesh_hit_mask] = new_d
                next_s_ref[mesh_hit_mask] = new_s_ref
                next_p_ref[mesh_hit_mask] = torch.cross(new_d, new_s_ref, dim=-1)
                next_stokes[mesh_hit_mask] = new_stokes

                is_reflected = is_diff | is_spec
                curr_medium_ids[mesh_hit_mask] = torch.where(
                    is_reflected.squeeze(-1), n1_ids, n2_ids)
                prev_hit_faces[mesh_hit_mask] = faces_hit

                # Kill RR victims
                if rr_threshold > 0.0:
                    if rr_kill.any():
                        rr_idx = torch.where(mesh_hit_mask)[0][rr_kill]
                        active_mask[rr_idx] = False

            # ---------- 5. Escape ----------
            no_hit_mask = active_mask & (~grin_mask) & (hit_face_ids < 0)
            if no_hit_mask.any():
                next_origins[no_hit_mask] = curr_origins[no_hit_mask] + \
                                            curr_dirs[no_hit_mask] * 1.0
                active_mask[no_hit_mask] = False

            # ---------- 6. Update ----------
            curr_origins = next_origins
            curr_dirs = next_dirs
            curr_stokes = next_stokes
            curr_s_ref = next_s_ref
            curr_p_ref = next_p_ref
            curr_weight = next_weight

            # ---------- 7. Record ----------
            hist_pos.append(curr_origins.clone())
            hist_stokes.append(curr_stokes.clone())
            hist_active.append(active_mask.clone())
            hist_refl.append(step_refl)
            hist_trans.append(step_trans)
            hist_abs.append(step_abs)
            hist_n1.append(step_n1)
            hist_n2.append(step_n2)
            hist_k1.append(step_k1)
            hist_k2.append(step_k2)
            hist_pol_deg.append(
                0.5 * torch.atan2(curr_stokes[:, 2:3], curr_stokes[:, 1:2])
                * (180.0 / math.pi))
            hist_energy.append(curr_weight.clone())
            hist_rk_err.append(step_rk_err)
            hist_ds.append(ds_per_ray.clone())
            hist_dop.append(degree_of_polarization(curr_stokes).clone())

            step += 1

        # =============================================================
        # ★ Strict energy conservation check
        # =============================================================
        energy_hist = torch.stack(hist_energy, dim=1)  # [N, S+1]

        in_range = bool(torch.all(
            (energy_hist >= -1e-6) & (energy_hist <= 1.0 + 1e-6)).item())

        if energy_hist.shape[1] >= 2:
            diffs = energy_hist[:, 1:] - energy_hist[:, :-1]
            monotonic = bool(torch.all(diffs <= 1e-6).item())
        else:
            monotonic = True

        final_active = active_mask
        final_energy = energy_hist[:, -1]
        inactive_zero = bool(torch.all(
            (final_active) | (final_energy.abs() < 1e-6)).item())

        energy_ok = in_range and monotonic and inactive_zero

        return {
            "positions": torch.stack(hist_pos, dim=1),
            "stokes": torch.stack(hist_stokes, dim=1),
            "active": torch.stack(hist_active, dim=1),
            "wavelengths": curr_wls,
            "reflectivity": torch.stack(hist_refl, dim=1),
            "transmissivity": torch.stack(hist_trans, dim=1),
            "absorptivity": torch.stack(hist_abs, dim=1),
            "n1": torch.stack(hist_n1, dim=1),
            "n2": torch.stack(hist_n2, dim=1),
            "k1": torch.stack(hist_k1, dim=1),
            "k2": torch.stack(hist_k2, dim=1),
            "polarization_deg": torch.stack(hist_pol_deg, dim=1),
            "dop": torch.stack(hist_dop, dim=1),
            "energy": energy_hist,
            "rk_error": torch.stack(hist_rk_err, dim=1),
            "ds_history": torch.stack(hist_ds, dim=1),
            "energy_conserved": energy_ok,
        }

    def _adaptive_step(self, r, d, field, wavelength, h_per_ray,
                       tol, h_min, h_max):
        """Per-ray adaptive RK4 with anti-deadlock guard."""

        def rk4_step(rr, dd, hh):
            n1, _ = field.get_complex_index(rr, wavelength)
            g1 = field.get_gradient(rr, wavelength)
            p = n1 * dd
            dr1 = p / n1
            dp1 = g1

            r2 = rr + 0.5 * hh.unsqueeze(-1) * dr1
            n2, _ = field.get_complex_index(r2, wavelength)
            g2 = field.get_gradient(r2, wavelength)
            p2 = p + 0.5 * hh.unsqueeze(-1) * dp1
            dr2 = p2 / n2
            dp2 = g2

            r3 = rr + 0.5 * hh.unsqueeze(-1) * dr2
            n3, _ = field.get_complex_index(r3, wavelength)
            g3 = field.get_gradient(r3, wavelength)
            p3 = p + 0.5 * hh.unsqueeze(-1) * dp2
            dr3 = p3 / n3
            dp3 = g3

            r4 = rr + hh.unsqueeze(-1) * dr3
            n4, _ = field.get_complex_index(r4, wavelength)
            g4 = field.get_gradient(r4, wavelength)
            p4 = p + hh.unsqueeze(-1) * dp3
            dr4 = p4 / n4
            dp4 = g4

            r_next = rr + (hh.unsqueeze(-1) / 6.0) * (
                dr1 + 2 * dr2 + 2 * dr3 + dr4)
            p_next = p + (hh.unsqueeze(-1) / 6.0) * (
                dp1 + 2 * dp2 + 2 * dp3 + dp4)
            n_next, _ = field.get_complex_index(r_next, wavelength)
            d_next = p_next / n_next
            d_next = d_next / torch.norm(d_next, dim=-1, keepdim=True)
            return r_next, d_next

        r1, d1 = rk4_step(r, d, h_per_ray)
        h_half = h_per_ray * 0.5
        r_mid, d_mid = rk4_step(r, d, h_half)
        r2, d2 = rk4_step(r_mid, d_mid, h_half)

        err = torch.maximum(
            torch.norm(r1 - r2, dim=-1),
            torch.norm(d1 - d2, dim=-1))

        accepted = err < tol
        at_h_min = h_per_ray <= (h_min * 1.05)
        force_accept = (~accepted) & at_h_min
        accepted = accepted | force_accept

        acc_3d = accepted.unsqueeze(-1)
        r_next = torch.where(acc_3d, r2, r)
        d_next = torch.where(acc_3d, d2, d)

        h_max_t = torch.full_like(h_per_ray, float(h_max))
        h_min_t = torch.full_like(h_per_ray, float(h_min))
        new_h = torch.where(
            accepted,
            torch.minimum(h_per_ray * 1.5, h_max_t),
            torch.maximum(h_per_ray * 0.5, h_min_t))

        return r_next, d_next, new_h, err, accepted