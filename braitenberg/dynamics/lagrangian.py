"""Act II: Multi-source Lagrangian chaos, FTLE / LCS ridges, fractal basins, and Poincare maps.

When a second or third stimulus source enters the arena, SO(2) rotational symmetry
breaks and memoryless Braitenberg vehicles lift from a 2D planar phase portrait into
a genuine 3D nonlinear flow on SE(2) = R^2 x S^1 (or T^2 x S^1 with periodic edges).
Finite-Time Lyapunov Exponent (FTLE) ridges expose the hidden Lagrangian Coherent
Structures (separatrices / stable manifolds of saddle orbits) that partition initial
conditions into competing lamp capture or residence basins.

This module never imports pygame (see AGENTS.md).
"""

from __future__ import annotations

import copy
import math

import numpy as np

from braitenberg.definition import ENERGY_SENSOR, VehicleDefinition
from braitenberg.dynamics.phase_plane import (
    _can_vectorize_stateless,
    eval_response_curve,
    resolve_primary_source,
    wrap_pi,
)
from braitenberg.vehicle import Vehicle
from braitenberg.world import World, to_surface


def extract_primary_sources(
    vehicle_def: VehicleDefinition,
    world: World,
    time: float = 0.0,
) -> tuple[str | None, list[tuple[float, float]]]:
    """Return ``(primary_stimulus, [(sx, sy), ...])`` for active sources in ``world``."""
    stim_name, _ = resolve_primary_source(vehicle_def, world)
    if stim_name is not None and stim_name in world.stimuli:
        pts = [
            (float(s.position[0]), float(s.position[1]))
            for s in world.stimuli[stim_name].sources
            if s.active(time)
        ]
        if pts:
            return stim_name, pts
    for name, st in world.stimuli.items():
        pts = [(float(s.position[0]), float(s.position[1])) for s in st.sources if s.active(time)]
        if pts:
            return name, pts
    return stim_name, []


def _compile_fast_stepper(vehicle_def: VehicleDefinition, world: World):
    """Pre-compile a zero-overhead NumPy step kernel for stateless vehicles.

    Returns ``None`` if ``vehicle_def`` requires stateful ``Vehicle`` evaluation.
    """
    can_vec, topo_order = _can_vectorize_stateless(vehicle_def)
    if not can_vec:
        return None
    if any(s.field_of_view < 360.0 for s in vehicle_def.sensors):
        # Directional sensors supported too
        pass

    hw = vehicle_def.general.width / 2.0
    hl = vehicle_def.general.length / 2.0
    arena = world.arena
    is_wrap = arena.edges == "wrap"
    W, H = float(arena.width), float(arena.height)
    half_W, half_H = 0.5 * W, 0.5 * H

    # Pre-pack sensor specs
    sensor_specs = []
    for s in vehicle_def.sensors:
        if s.type == ENERGY_SENSOR:
            sensor_specs.append(("energy", s.name, s.response, vehicle_def.energy.initial if vehicle_def.energy else 0.0))
            continue
        st = world.stimuli.get(s.type)
        srcs = st.sources if st is not None else []
        src_data = [
            (float(src.position[0]), float(src.position[1]), float(src.intensity), float(src.falloff) ** 2, src.schedule)
            for src in srcs
        ]
        surf_data = (float(st.surfaces.intensity), float(st.surfaces.falloff) ** 2) if (st and st.surfaces) else None
        sensor_specs.append(
            (
                "world",
                s.name,
                s.response,
                float(s.position[0] * hw),
                float(s.position[1] * hl),
                float(s.field_of_view),
                math.radians(s.direction),
                src_data,
                surf_data,
            )
        )

    obs_boxes = [o.box() for o in world.obstacles]
    unit_specs = [
        (u.name, float(u.threshold), [(c.source, float(c.sign)) for c in u.inputs if c.active])
        for k in topo_order
        for u in (vehicle_def.units[k],)
    ]
    act_specs = [
        (
            float(a.base_speed),
            float(a.gain),
            float(a.max_speed),
            float(a.position[0] * hw),
            [(c.source, float(c.sign)) for c in a.inputs if c.active],
        )
        for a in vehicle_def.actuators
    ]
    n_act = len(act_specs)
    sxx = sum(spec[3] * spec[3] for spec in act_specs)
    veh_radius = max(hw, hl)

    def step_fn(
        ux: np.ndarray,
        uy: np.ndarray,
        heading: np.ndarray,
        t_now: float,
        dt: float,
        noise_vec: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = np.cos(heading)
        s_ang = np.sin(heading)
        shape = ux.shape
        signals: dict[str, np.ndarray] = {}

        # Current arena positions for sensor lookup
        if is_wrap:
            wx = ux % W
            wy = uy % H
        else:
            wx = ux
            wy = uy

        for spec in sensor_specs:
            if spec[0] == "energy":
                _, sname, resp, init_e = spec
                signals[sname] = eval_response_curve(resp, np.full(shape, init_e, dtype=float))
                continue
            _, sname, resp, tx, ty, fov, dir_rad, src_data, surf_data = spec
            px = wx + ty * c + tx * s_ang
            py = wy + ty * s_ang - tx * c
            raw = np.zeros(shape, dtype=float)
            for sx0, sy0, inten, f2, sched in src_data:
                if sched is not None and not sched.active(t_now):
                    continue
                dx = px - sx0
                dy = py - sy0
                if is_wrap:
                    dx = (dx + half_W) % W - half_W
                    dy = (dy + half_H) % H - half_H
                d2 = dx * dx + dy * dy
                contrib = inten / (1.0 + d2 / f2)
                if fov < 360.0:
                    look = heading + dir_rad
                    off = (np.arctan2(-dy, -dx) - look + math.pi) % (2.0 * math.pi) - math.pi
                    mask = (np.abs(off) <= math.radians(0.5 * fov)) | (d2 == 0.0)
                    contrib = np.where(mask, contrib, 0.0)
                raw = raw + contrib

            if surf_data is not None and obs_boxes:
                s_inten, s_f2 = surf_data
                for bx, by, bhw, bhl, bturn in obs_boxes:
                    dx = px - bx
                    dy = py - by
                    if is_wrap:
                        dx = (dx + half_W) % W - half_W
                        dy = (dy + half_H) % H - half_H
                    _, _, dist = to_surface(dx, dy, bhw, bhl, bturn)
                    raw = raw + s_inten / (1.0 + (dist * dist) / s_f2)

            signals[sname] = eval_response_curve(resp, raw)

        for uname, thresh, u_inputs in unit_specs:
            tot = np.zeros(shape, dtype=float)
            for src_sig, sign in u_inputs:
                tot = tot + sign * signals[src_sig]
            signals[uname] = np.where(tot >= thresh, 1.0, 0.0)

        v_sum = np.zeros(shape, dtype=float)
        om_num = np.zeros(shape, dtype=float)
        for base_sp, gain, max_sp, ax_pos, a_inputs in act_specs:
            tot = np.zeros(shape, dtype=float)
            for src_sig, sign in a_inputs:
                tot = tot + sign * signals[src_sig]
            ms = np.clip(base_sp + gain * tot, 0.0, max_sp)
            v_sum = v_sum + ms
            om_num = om_num + ms * ax_pos

        speed = v_sum / n_act
        turn_rate = om_num / sxx if sxx > 0.0 else np.zeros(shape, dtype=float)

        if noise_vec is not None:
            heading[:] = heading + turn_rate * dt + noise_vec
        else:
            heading[:] = heading + turn_rate * dt

        step_dx = speed * np.cos(heading) * dt
        step_dy = speed * np.sin(heading) * dt
        ux[:] = ux + step_dx
        uy[:] = uy + step_dy

        if not is_wrap:
            # Bounce boundary rule matching Arena.apply_edges
            over_x_lo = ux < 0.0
            over_x_hi = ux > W
            if np.any(over_x_lo | over_x_hi):
                ux[:] = np.where(over_x_lo, -ux, np.where(over_x_hi, 2.0 * W - ux, ux))
                heading[:] = np.where(over_x_lo | over_x_hi, math.pi - heading, heading)
            over_y_lo = uy < 0.0
            over_y_hi = uy > H
            if np.any(over_y_lo | over_y_hi):
                uy[:] = np.where(over_y_lo, -uy, np.where(over_y_hi, 2.0 * H - uy, uy))
                heading[:] = np.where(over_y_lo | over_y_hi, -heading, heading)

        if obs_boxes:
            cur_wx = ux % W if is_wrap else ux
            cur_wy = uy % H if is_wrap else uy
            for bx, by, bhw, bhl, bturn in obs_boxes:
                dx = cur_wx - bx
                dy = cur_wy - by
                if is_wrap:
                    dx = (dx + half_W) % W - half_W
                    dy = (dy + half_H) % H - half_H
                gx, gy, dist = to_surface(dx, dy, bhw, bhl, bturn)
                hit = (dist > 0.0) & (dist < veh_radius)
                if np.any(hit):
                    push = np.where(hit, (veh_radius - dist) / np.maximum(dist, 1e-9), 0.0)
                    ux[:] = ux - gx * push
                    uy[:] = uy - gy * push

        return speed, turn_rate

    return step_fn


def _compute_shannon_and_box_entropy(basin: np.ndarray, box_size: int = 8) -> tuple[float, float, float]:
    """Compute global Shannon basin entropy, Sanjuan box basin entropy, and boundary fraction."""
    flat = basin.ravel()
    _, counts = np.unique(flat, return_counts=True)
    probs = counts / float(flat.size)
    shannon = float(-np.sum(probs * np.log(np.maximum(probs, 1e-15))))
    if abs(shannon) < 1e-15:
        shannon = 0.0

    # Boundary fraction: fraction of interior cells differing from a 4-neighbor
    diff_x = basin[:, 1:] != basin[:, :-1]
    diff_y = basin[1:, :] != basin[:-1, :]
    bnd_mask = np.zeros_like(basin, dtype=bool)
    bnd_mask[:, :-1] |= diff_x
    bnd_mask[:, 1:] |= diff_x
    bnd_mask[:-1, :] |= diff_y
    bnd_mask[1:, :] |= diff_y
    boundary_fraction = float(np.mean(bnd_mask))

    # Local box basin entropy S_b (Daza & Sanjuan 2016)
    ny, nx = basin.shape
    by = max(1, ny // box_size)
    bx = max(1, nx // box_size)
    box_ent_sum = 0.0
    n_boxes = 0
    for iy in range(by):
        y0 = iy * box_size
        y1 = ny if iy == by - 1 else (iy + 1) * box_size
        for ix in range(bx):
            x0 = ix * box_size
            x1 = nx if ix == bx - 1 else (ix + 1) * box_size
            patch = basin[y0:y1, x0:x1].ravel()
            if patch.size == 0:
                continue
            _, p_counts = np.unique(patch, return_counts=True)
            if len(p_counts) > 1:
                p = p_counts / float(patch.size)
                box_ent_sum += float(-np.sum(p * np.log(p)))
            n_boxes += 1
    box_entropy = box_ent_sum / max(n_boxes, 1)
    return shannon, box_entropy, boundary_fraction


def compute_ftle_and_basins(
    vehicle_def: VehicleDefinition,
    world: World,
    nx: int = 120,
    ny: int = 120,
    horizon: float = 12.0,
    dt: float = 0.05,
    heading_mode: str = "toward_centroid",
    fixed_heading: float = 0.0,
    capture_radius: float = 5.0,
) -> dict:
    """Integrate a 2D spatial grid of noise-free vehicles to extract FTLE ridges and basins.

    Integrates unwrapped coordinates ``(ux, uy)`` while applying periodic sensor lookups
    when ``world.arena.edges == "wrap"`` so periodic boundaries never create fake FTLE spikes.
    """
    W, H = float(world.arena.width), float(world.arena.height)
    dx_grid = W / nx
    dy_grid = H / ny
    x = np.linspace(0.5 * dx_grid, W - 0.5 * dx_grid, nx)
    y = np.linspace(0.5 * dy_grid, H - 0.5 * dy_grid, ny)
    X0, Y0 = np.meshgrid(x, y, indexing="xy")

    _, sources = extract_primary_sources(vehicle_def, world)
    if sources:
        cx = float(np.mean([p[0] for p in sources]))
        cy = float(np.mean([p[1] for p in sources]))
    else:
        cx, cy = 0.5 * W, 0.5 * H

    if heading_mode == "toward_centroid":
        TH0 = np.arctan2(cy - Y0, cx - X0)
    elif heading_mode == "away_from_centroid":
        TH0 = np.arctan2(Y0 - cy, X0 - cx)
    elif heading_mode == "tangent":
        TH0 = np.arctan2(cy - Y0, cx - X0) - 0.5 * math.pi
    else:
        TH0 = np.full_like(X0, float(fixed_heading))

    n_steps = max(1, int(round(abs(horizon) / abs(dt))))
    step_dt = math.copysign(abs(dt), horizon)
    total_T = max(abs(n_steps * step_dt), 1e-9)

    ux = X0.ravel().copy()
    uy = Y0.ravel().copy()
    heading = TH0.ravel().copy()
    n_pts = ux.size

    k_sources = len(sources)
    src_arr = np.array(sources, dtype=float) if k_sources > 0 else np.zeros((0, 2), dtype=float)
    is_wrap = world.arena.edges == "wrap"
    half_W, half_H = 0.5 * W, 0.5 * H

    captured_basin = np.full(n_pts, -1, dtype=int)
    capture_time = np.full(n_pts, np.nan, dtype=float)

    # Track tail lamp residence for orbital attractors (e.g. Vehicle 4a Orbiter) that circle
    # outside a small core capture_radius without colliding with the lamp center.
    tail_start = int(0.6 * n_steps)
    tail_counts = np.zeros((n_pts, max(k_sources, 1)), dtype=int)
    tail_dist_sum = np.zeros((n_pts, max(k_sources, 1)), dtype=float)
    first_zone_time = np.full((n_pts, max(k_sources, 1)), np.nan, dtype=float)
    tail_samples = 0

    orbit_zone_radius = max(float(capture_radius) * 4.5, 25.0)
    if k_sources >= 2:
        dists = [
            math.hypot(sources[i][0] - sources[j][0], sources[i][1] - sources[j][1])
            for i in range(k_sources)
            for j in range(i + 1, k_sources)
        ]
        orbit_zone_radius = min(orbit_zone_radius, 0.58 * min(dists))

    fast_stepper = _compile_fast_stepper(vehicle_def, world)

    if fast_stepper is not None:
        for step in range(n_steps):
            t_now = step * step_dt
            fast_stepper(ux, uy, heading, t_now, step_dt, None)

            if k_sources > 0:
                wx = ux % W if is_wrap else ux
                wy = uy % H if is_wrap else uy
                dx_s = wx[:, None] - src_arr[None, :, 0]
                dy_s = wy[:, None] - src_arr[None, :, 1]
                if is_wrap:
                    dx_s = (dx_s + half_W) % W - half_W
                    dy_s = (dy_s + half_H) % H - half_H
                d_all = np.hypot(dx_s, dy_s)
                nearest_k = np.argmin(d_all, axis=1)
                nearest_d = d_all[np.arange(n_pts), nearest_k]

                newly_hit = (captured_basin < 0) & (nearest_d <= capture_radius)
                if np.any(newly_hit):
                    captured_basin[newly_hit] = nearest_k[newly_hit]
                    capture_time[newly_hit] = (step + 1) * abs(step_dt)

                in_zone = d_all <= orbit_zone_radius
                first_mask = in_zone & np.isnan(first_zone_time)
                if np.any(first_mask):
                    first_zone_time[first_mask] = (step + 1) * abs(step_dt)

                if step >= tail_start:
                    tail_counts[np.arange(n_pts), nearest_k] += 1
                    tail_dist_sum += d_all
                    tail_samples += 1
    else:
        # Stateful fallback loop
        clean_def = copy.deepcopy(vehicle_def)
        clean_def.general.noise_std = 0.0
        rng = np.random.default_rng(0)
        vehs = [Vehicle(clean_def, float(ux[i]), float(uy[i]), float(heading[i]), dt=abs(step_dt), seed=0) for i in range(n_pts)]
        for step in range(n_steps):
            t_now = step * abs(step_dt)
            for i, veh in enumerate(vehs):
                prev_x, prev_y = veh.x, veh.y
                readings = {
                    name: float(world.stimulus_at(st, px, py, t_now, veh.heading))
                    for name, st, px, py in veh.sensor_points()
                }
                veh.step(readings, abs(step_dt), rng)
                world.arena.apply_edges(veh)
                world.block(veh)
                ddx, ddy = world.arena.delta(veh.x - prev_x, veh.y - prev_y)
                ux[i] += ddx
                uy[i] += ddy

    # Assign orbital residence basins for particles not yet captured within core capture_radius
    if k_sources > 0 and tail_samples > 0:
        still_free = captured_basin < 0
        best_k = np.argmax(tail_counts, axis=1)
        dom_frac = tail_counts[np.arange(n_pts), best_k] / float(tail_samples)
        mean_d = tail_dist_sum[np.arange(n_pts), best_k] / float(tail_samples)
        orbital_captured = still_free & (dom_frac >= 0.78) & (mean_d <= orbit_zone_radius)
        if np.any(orbital_captured):
            captured_basin[orbital_captured] = best_k[orbital_captured]
            zt = first_zone_time[np.arange(n_pts), best_k]
            capture_time[orbital_captured] = np.where(
                np.isnan(zt[orbital_captured]), tail_start * abs(step_dt), zt[orbital_captured]
            )

    UX = ux.reshape((ny, nx))
    UY = uy.reshape((ny, nx))
    final_x = (UX % W) if is_wrap else UX.copy()
    final_y = (UY % H) if is_wrap else UY.copy()

    # Spatial deformation gradient D Phi_T from smooth unwrapped flow map (UX, UY)
    dux_dy, dux_dx = np.gradient(UX, y, x)
    duy_dy, duy_dx = np.gradient(UY, y, x)

    # Cauchy-Green tensor C = (D Phi_T)^T (D Phi_T) = [[c11, c12], [c12, c22]]
    c11 = dux_dx * dux_dx + duy_dx * duy_dx
    c22 = dux_dy * dux_dy + duy_dy * duy_dy
    c12 = dux_dx * dux_dy + duy_dx * duy_dy

    half_tr = 0.5 * (c11 + c22)
    half_diff = 0.5 * (c11 - c22)
    lam_max = half_tr + np.sqrt(half_diff * half_diff + c12 * c12)
    ftle = (1.0 / total_T) * np.log(np.sqrt(np.maximum(lam_max, 1e-12)))

    basin_2d = captured_basin.reshape((ny, nx))
    cap_time_2d = capture_time.reshape((ny, nx))
    shannon_ent, box_ent, bnd_frac = _compute_shannon_and_box_entropy(basin_2d)

    return {
        "x": x,
        "y": y,
        "ftle": ftle,
        "basin": basin_2d,
        "final_x": final_x,
        "final_y": final_y,
        "final_ux": UX,
        "final_uy": UY,
        "capture_time": cap_time_2d,
        "sources": sources,
        "basin_entropy": shannon_ent,
        "box_basin_entropy": box_ent,
        "boundary_fraction": bnd_frac,
    }


def _initialize_ensemble(
    n_particles: int,
    world: World,
    sources: list[tuple[float, float]],
    init_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate initial ``(x0, y0, heading0)`` for swarm visualizations."""
    W, H = float(world.arena.width), float(world.arena.height)
    if sources:
        cx = float(np.mean([p[0] for p in sources]))
        cy = float(np.mean([p[1] for p in sources]))
    else:
        cx, cy = 0.5 * W, 0.5 * H

    if init_mode in ("rings", "annulus"):
        # Dual concentric rings inside and outside typical preferred radius (ideal for Orbiter!)
        sx0, sy0 = sources[0] if sources else (cx, cy)
        n_in = n_particles // 2
        n_out = n_particles - n_in
        ang_in = np.linspace(0.0, 2.0 * math.pi, n_in, endpoint=False) + rng.uniform(-0.05, 0.05, n_in)
        ang_out = np.linspace(0.0, 2.0 * math.pi, n_out, endpoint=False) + rng.uniform(-0.05, 0.05, n_out)
        r_in = rng.uniform(5.0, 11.5, n_in)
        r_out = rng.uniform(28.0, min(42.0, 0.46 * min(W, H)), n_out)
        x0 = np.concatenate([sx0 + r_in * np.cos(ang_in), sx0 + r_out * np.cos(ang_out)]) % W
        y0 = np.concatenate([sy0 + r_in * np.sin(ang_in), sy0 + r_out * np.sin(ang_out)]) % H
        th0 = np.concatenate([ang_in + 0.55 * math.pi, ang_out + 0.55 * math.pi])
        return x0, y0, np.asarray(wrap_pi(th0), dtype=float)

    if init_mode in ("grid", "uniform"):
        nx = max(2, int(round(math.sqrt(n_particles * W / H))))
        ny = max(2, int(math.ceil(n_particles / nx)))
        gx = np.linspace(0.06 * W, 0.94 * W, nx)
        gy = np.linspace(0.06 * H, 0.94 * H, ny)
        GX, GY = np.meshgrid(gx, gy, indexing="xy")
        x0 = GX.ravel()[:n_particles] + rng.normal(0.0, 0.25, n_particles)
        y0 = GY.ravel()[:n_particles] + rng.normal(0.0, 0.25, n_particles)
        x0 = np.clip(x0, 0.5, W - 0.5)
        y0 = np.clip(y0, 0.5, H - 0.5)
        th0 = np.arctan2(cy - y0, cx - x0)
        return x0, y0, th0

    if init_mode in ("separatrix", "fan"):
        # Compact packet straddling the saddle midpoint between sources
        x0 = cx + rng.normal(0.0, 3.5, n_particles)
        y0 = cy + np.linspace(-0.32 * H, 0.32 * H, n_particles)
        th0 = np.arctan2(cy - y0, cx - x0) + rng.uniform(-0.35, 0.35, n_particles)
        return x0 % W, y0 % H, np.asarray(wrap_pi(th0), dtype=float)

    # Default "circle": multi-layered ring around centroid
    angles = np.linspace(0.0, 2.0 * math.pi, n_particles, endpoint=False)
    radii = 0.32 * min(W, H) * (0.65 + 0.55 * ((np.arange(n_particles) % 5) / 4.0))
    x0 = (cx + radii * np.cos(angles)) % W
    y0 = (cy + radii * np.sin(angles)) % H
    th0 = np.arctan2(cy - y0, cx - x0) + 0.25 * np.sin(3.0 * angles)
    return x0, y0, np.asarray(wrap_pi(th0), dtype=float)


def simulate_swarm_ensemble(
    vehicle_def: VehicleDefinition,
    world: World,
    n_particles: int = 400,
    steps: int = 600,
    dt: float = 0.05,
    seed: int = 0,
    init_mode: str = "circle",
    noise_scale: float = 0.0,
) -> dict:
    """Integrate an ensemble of ``n_particles`` vehicles over ``steps`` steps.

    Records both wrapped arena coordinates ``(x, y)`` and continuous unwrapped
    coordinates ``(ux, uy)`` so comet/trail renderers never draw lines across wrap edges.
    """
    rng = np.random.default_rng(seed)
    _, sources = extract_primary_sources(vehicle_def, world)
    x0, y0, th0 = _initialize_ensemble(n_particles, world, sources, init_mode, rng)

    W, H = float(world.arena.width), float(world.arena.height)
    is_wrap = world.arena.edges == "wrap"
    half_W, half_H = 0.5 * W, 0.5 * H

    t_arr = np.arange(steps, dtype=float) * dt
    x_hist = np.empty((steps, n_particles), dtype=float)
    y_hist = np.empty((steps, n_particles), dtype=float)
    ux_hist = np.empty((steps, n_particles), dtype=float)
    uy_hist = np.empty((steps, n_particles), dtype=float)
    th_hist = np.empty((steps, n_particles), dtype=float)
    sp_hist = np.empty((steps, n_particles), dtype=float)
    om_hist = np.empty((steps, n_particles), dtype=float)

    ux = x0.copy()
    uy = y0.copy()
    heading = th0.copy()

    eff_noise = float(noise_scale) * math.sqrt(dt)
    fast_stepper = _compile_fast_stepper(vehicle_def, world)

    if fast_stepper is not None:
        for step in range(steps):
            ux_pos = ux.copy()
            uy_pos = uy.copy()
            noise_vec = rng.normal(0.0, eff_noise, n_particles) if eff_noise > 0.0 else None
            sp, om = fast_stepper(ux, uy, heading, step * dt, dt, noise_vec)
            ux_hist[step] = ux_pos
            uy_hist[step] = uy_pos
            x_hist[step] = (ux_pos % W) if is_wrap else ux_pos
            y_hist[step] = (uy_pos % H) if is_wrap else uy_pos
            th_hist[step] = wrap_pi(heading)
            sp_hist[step] = sp
            om_hist[step] = om
    else:
        clean_def = copy.deepcopy(vehicle_def)
        clean_def.general.noise_std = float(noise_scale)
        vehs = [
            Vehicle(clean_def, float(x0[i]), float(y0[i]), float(th0[i]), dt=dt, seed=[seed, i])
            for i in range(n_particles)
        ]
        for step in range(steps):
            t_now = step * dt
            for i, veh in enumerate(vehs):
                prev_x, prev_y = veh.x, veh.y
                ux_hist[step, i] = ux[i]
                uy_hist[step, i] = uy[i]
                x_hist[step, i] = prev_x
                y_hist[step, i] = prev_y
                readings = {
                    name: float(world.stimulus_at(st, px, py, t_now, veh.heading))
                    for name, st, px, py in veh.sensor_points()
                }
                veh.step(readings, dt, rng)
                world.arena.apply_edges(veh)
                world.block(veh)
                ddx, ddy = world.arena.delta(veh.x - prev_x, veh.y - prev_y)
                ux[i] += ddx
                uy[i] += ddy
                th_hist[step, i] = float(wrap_pi(veh.heading))
                sp_hist[step, i] = veh.speed
                om_hist[step, i] = veh.turn_rate

    # Classify asymptotic basin ID per particle from the tail quarter of the run
    basin_id = np.full(n_particles, -1, dtype=int)
    if sources:
        src_arr = np.array(sources, dtype=float)
        tail = max(1, steps // 4)
        tx = x_hist[-tail:]
        ty = y_hist[-tail:]
        dx_s = tx[:, :, None] - src_arr[None, None, :, 0]
        dy_s = ty[:, :, None] - src_arr[None, None, :, 1]
        if is_wrap:
            dx_s = (dx_s + half_W) % W - half_W
            dy_s = (dy_s + half_H) % H - half_H
        d_tail = np.hypot(dx_s, dy_s)  # (tail, N, K)
        mean_d = d_tail.mean(axis=0)   # (N, K)
        best_k = np.argmin(mean_d, axis=1)
        best_dist = mean_d[np.arange(n_particles), best_k]
        basin_id = np.where(best_dist <= 30.0, best_k, -1)

    return {
        "t": t_arr,
        "x": x_hist,
        "y": y_hist,
        "ux": ux_hist,
        "uy": uy_hist,
        "heading": th_hist,
        "speed": sp_hist,
        "turn_rate": om_hist,
        "basin_id": basin_id,
        "sources": sources,
    }


def poincare_section(
    vehicle_def: VehicleDefinition,
    world: World,
    n_ic: int = 48,
    steps: int = 3000,
    dt: float = 0.05,
    axis: str = "x_mid",
    seed: int = 0,
) -> dict:
    """Record oriented crossings of a symmetry plane between lamps.

    Parameters
    ----------
    axis : ``"x_mid"`` or ``"y_mid"``
        ``"x_mid"`` slices vertically at the source-centroid x-coordinate (recording ``y``);
        ``"y_mid"`` slices horizontally at the source-centroid y-coordinate (recording ``x``).
    """
    W, H = float(world.arena.width), float(world.arena.height)
    is_wrap = world.arena.edges == "wrap"
    _, sources = extract_primary_sources(vehicle_def, world)
    if sources:
        cx = float(np.mean([p[0] for p in sources]))
        cy = float(np.mean([p[1] for p in sources]))
        ref_source = sources[0]
    else:
        cx, cy = 0.5 * W, 0.5 * H
        ref_source = (cx, cy)

    rng = np.random.default_rng(seed)
    x0, y0, th0 = _initialize_ensemble(n_ic, world, sources, "grid", rng)
    # Add diverse initial headings to illuminate full phase-space structure
    th0 = np.asarray(wrap_pi(th0 + np.linspace(-math.pi, math.pi, n_ic, endpoint=False)), dtype=float)

    ux = x0.copy()
    uy = y0.copy()
    heading = th0.copy()

    use_x = axis != "y_mid"
    sec_pos = cx if use_x else cy
    period_n = W if use_x else H
    period_t = H if use_x else W

    coords: list[float] = []
    psis: list[float] = []
    headings: list[float] = []
    ic_ids: list[int] = []
    directions: list[int] = []

    fast_stepper = _compile_fast_stepper(vehicle_def, world)
    if fast_stepper is None:
        # Fallback through simulate_swarm_ensemble
        ens = simulate_swarm_ensemble(
            vehicle_def, world, n_particles=n_ic, steps=steps, dt=dt, seed=seed, init_mode="grid", noise_scale=0.0
        )
        u_norm = ens["ux"] if use_x else ens["uy"]
        u_tran = ens["uy"] if use_x else ens["ux"]
        for s in range(steps - 1):
            prev_n, curr_n = u_norm[s], u_norm[s + 1]
            if is_wrap:
                s0 = (prev_n - sec_pos + 0.5 * period_n) % period_n - 0.5 * period_n
                s1 = s0 + (curr_n - prev_n)
            else:
                s0 = prev_n - sec_pos
                s1 = curr_n - sec_pos
            crossed = ((s0 < 0.0) & (s1 >= 0.0)) | ((s0 > 0.0) & (s1 <= 0.0))
            for idx in np.flatnonzero(crossed):
                denom = float(s1[idx] - s0[idx])
                alpha = float(-s0[idx] / denom) if abs(denom) > 1e-12 else 0.5
                tc = float(u_tran[s, idx] + alpha * (u_tran[s + 1, idx] - u_tran[s, idx]))
                if is_wrap:
                    tc %= period_t
                th_c = float(wrap_pi(ens["heading"][s, idx] + alpha * wrap_pi(ens["heading"][s + 1, idx] - ens["heading"][s, idx])))
                px = sec_pos if use_x else tc
                py = tc if use_x else sec_pos
                dx_r, dy_r = world.arena.delta(ref_source[0] - px, ref_source[1] - py)
                psi_c = float(wrap_pi(math.atan2(dy_r, dx_r) - th_c))
                coords.append(tc)
                psis.append(psi_c)
                headings.append(th_c)
                ic_ids.append(int(idx))
                directions.append(1 if denom > 0.0 else -1)
    else:
        for step in range(steps):
            prev_ux = ux.copy()
            prev_uy = uy.copy()
            prev_th = heading.copy()
            fast_stepper(ux, uy, heading, step * dt, dt, None)

            prev_n = prev_ux if use_x else prev_uy
            curr_n = ux if use_x else uy
            prev_t = prev_uy if use_x else prev_ux
            curr_t = uy if use_x else ux

            if is_wrap:
                s0 = (prev_n - sec_pos + 0.5 * period_n) % period_n - 0.5 * period_n
                s1 = s0 + (curr_n - prev_n)
            else:
                s0 = prev_n - sec_pos
                s1 = curr_n - sec_pos

            crossed = ((s0 < 0.0) & (s1 >= 0.0)) | ((s0 > 0.0) & (s1 <= 0.0))
            if np.any(crossed):
                for idx in np.flatnonzero(crossed):
                    denom = float(s1[idx] - s0[idx])
                    alpha = float(-s0[idx] / denom) if abs(denom) > 1e-12 else 0.5
                    tc = float(prev_t[idx] + alpha * (curr_t[idx] - prev_t[idx]))
                    if is_wrap:
                        tc %= period_t
                    dth = float(wrap_pi(heading[idx] - prev_th[idx]))
                    th_c = float(wrap_pi(prev_th[idx] + alpha * dth))
                    px = sec_pos if use_x else tc
                    py = tc if use_x else sec_pos
                    dx_r, dy_r = world.arena.delta(ref_source[0] - px, ref_source[1] - py)
                    psi_c = float(wrap_pi(math.atan2(dy_r, dx_r) - th_c))
                    coords.append(tc)
                    psis.append(psi_c)
                    headings.append(th_c)
                    ic_ids.append(int(idx))
                    directions.append(1 if denom > 0.0 else -1)

    return {
        "coord": np.asarray(coords, dtype=float),
        "psi": np.asarray(psis, dtype=float),
        "heading": np.asarray(headings, dtype=float),
        "ic_id": np.asarray(ic_ids, dtype=int),
        "direction": np.asarray(directions, dtype=int),
        "section_pos": float(sec_pos),
        "axis": axis,
    }
