"""Act I: SO(2) symmetry-reduced cylindrical polar phase plane for Braitenberg vehicles.

Around a single isotropic stimulus source at (sx, sy) on an unbounded (or locally
unwrapped) plane, rotational and translational symmetry reduce a memoryless vehicle's
3D configuration (x, y, theta) in SE(2) to a 2D autonomous dynamical system on the
cylinder (r, psi) in (0, inf) x [-pi, pi]:

    r   = hypot(sx - x, sy - y)
    psi = wrap_pi(atan2(sy - y, sx - x) - theta)

where psi = 0 means facing straight at the source, psi > 0 puts the source to the
vehicle's left, and psi < 0 puts the source to the vehicle's right. Along continuous
differential-drive kinematics (dx/dt = v cos(theta), dy/dt = v sin(theta),
dtheta/dt = omega), the exact polar ODEs are:

    dr/dt   = -v(r, psi) * cos(psi)
    dpsi/dt = (v(r, psi) / r) * sin(psi) - omega(r, psi)

This module never imports pygame (see AGENTS.md).
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np

from braitenberg.definition import ENERGY_SENSOR, Response, VehicleDefinition
from braitenberg.vehicle import Vehicle
from braitenberg.world import Arena, Source, World


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) in radians into [-pi, pi]."""
    arr = np.asarray(angle, dtype=float)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    # Map -pi to +pi only when input was congruent to +pi and positive, keeping [-pi, pi].
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def polar_coords(
    x: float | np.ndarray,
    y: float | np.ndarray,
    heading: float | np.ndarray,
    source_pos: tuple[float, float],
    arena: Arena | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arena coordinates (x, y, heading) to cylindrical polar (r, psi).

    Parameters
    ----------
    x, y, heading : float or array-like
        Vehicle position(s) and heading(s) in radians (0 = +x, CCW).
    source_pos : (sx, sy)
        Stimulus source coordinates in the arena.
    arena : Arena, optional
        If provided, uses ``arena.delta`` for shortest displacement on a torus.

    Returns
    -------
    r, psi : tuple[np.ndarray, np.ndarray]
        Distance to source ``r >= 0`` and body-frame source bearing ``psi`` in ``[-pi, pi]``.
    """
    sx, sy = float(source_pos[0]), float(source_pos[1])
    dx = np.subtract(sx, np.asarray(x, dtype=float))
    dy = np.subtract(sy, np.asarray(y, dtype=float))
    if arena is not None:
        dx, dy = arena.delta(dx, dy)
    r = np.asarray(np.hypot(dx, dy), dtype=float)
    phi = np.arctan2(dy, dx)
    psi = np.asarray((phi - np.asarray(heading, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)
    return r, psi


def eval_response_curve(response: Response, readings: np.ndarray) -> np.ndarray:
    """Vectorized sensor response supporting linear, peak/bell, step/threshold, and inverse."""
    rtype = response.type
    if rtype == "linear":
        return readings
    if rtype in ("peak", "bell"):
        return response.height * np.exp(-0.5 * ((readings - response.at) / response.width) ** 2)
    if rtype in ("step", "threshold"):
        return np.where(readings >= response.at, response.height, 0.0)
    if rtype == "inverse":
        scale = max(float(response.width), 1e-9)
        return response.height / (1.0 + np.maximum(readings, 0.0) / scale)
    return response.outputs(readings)


def resolve_primary_source(
    vehicle_def: VehicleDefinition,
    world: World,
    source_pos: tuple[float, float] | None = None,
    stimulus_name: str | None = None,
) -> tuple[str | None, tuple[float, float]]:
    """Select the primary stimulus name and source coordinates for polar analysis."""
    chosen_stim = stimulus_name
    if chosen_stim is None:
        for s in vehicle_def.sensors:
            if s.type != ENERGY_SENSOR and s.type in world.stimuli and world.stimuli[s.type].sources:
                chosen_stim = s.type
                break
    if chosen_stim is None:
        for name, st in world.stimuli.items():
            if st.sources:
                chosen_stim = name
                break
    if source_pos is not None:
        return chosen_stim, (float(source_pos[0]), float(source_pos[1]))
    if chosen_stim is not None and chosen_stim in world.stimuli and world.stimuli[chosen_stim].sources:
        pos = world.stimuli[chosen_stim].sources[0].position
        return chosen_stim, (float(pos[0]), float(pos[1]))
    return chosen_stim, (world.arena.width / 2.0, world.arena.height / 2.0)


def _sources_for_polar_sensor(
    sensor_type: str,
    world: World,
    source_pos: tuple[float, float],
    primary_stimulus: str | None,
) -> list[Source]:
    """Active sources co-located at ``source_pos`` for isotropic single-source reduction."""
    if sensor_type not in world.stimuli:
        return []
    active = [s for s in world.stimuli[sensor_type].sources if s.active(0.0)]
    if not active:
        return []
    sx, sy = source_pos
    colocated = [s for s in active if math.hypot(s.position[0] - sx, s.position[1] - sy) <= 1e-5]
    if colocated:
        return colocated
    if sensor_type == primary_stimulus or len(active) == 1:
        return [replace(active[0], position=source_pos)]
    return []


def _can_vectorize_stateless(vehicle_def: VehicleDefinition) -> tuple[bool, list[int]]:
    """Check whether ``vehicle_def`` can be evaluated statelessly in pure NumPy.

    Returns ``(True, topo_order)`` when there are no cameras, habituating sensors,
    threshold controls, or recurrent unit loops; otherwise ``(False, [])``.
    """
    if any(s.camera or s.habituation for s in vehicle_def.sensors):
        return False, []
    units = vehicle_def.units
    if not units:
        return True, []
    if any(u.is_control for u in units):
        return False, []
    unit_idx = {u.name: k for k, u in enumerate(units)}
    indegree = [0] * len(units)
    dependents: list[list[int]] = [[] for _ in units]
    for k, u in enumerate(units):
        for c in u.inputs:
            if not c.active:
                continue
            if c.source in unit_idx:
                src_k = unit_idx[c.source]
                if src_k == k:
                    return False, []
                indegree[k] += 1
                dependents[src_k].append(k)
    queue = [k for k, deg in enumerate(indegree) if deg == 0]
    order: list[int] = []
    while queue:
        curr = queue.pop(0)
        order.append(curr)
        for nxt in dependents[curr]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(units):
        return False, []
    return True, order


def evaluate_signals_to_kinematics(
    vehicle_def: VehicleDefinition,
    sensor_readings: dict[str, np.ndarray],
    shape: tuple[int, ...],
    topo_order: list[int] | None = None,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Map vectorized sensor readings of ``shape`` to ``(speed, turn_rate)``."""
    can_vec = topo_order is not None
    if not can_vec:
        can_vec, topo_order = _can_vectorize_stateless(vehicle_def)

    if can_vec:
        signals: dict[str, np.ndarray] = {}
        for s in vehicle_def.sensors:
            if s.type == ENERGY_SENSOR:
                val = vehicle_def.energy.initial if vehicle_def.energy else 0.0
                raw = np.full(shape, val, dtype=float)
            else:
                raw = sensor_readings.get(s.name, np.zeros(shape, dtype=float))
            signals[s.name] = eval_response_curve(s.response, raw)

        assert topo_order is not None
        for k in topo_order:
            u = vehicle_def.units[k]
            total = np.zeros(shape, dtype=float)
            for c in u.inputs:
                if c.active:
                    total = total + c.sign * signals[c.source]
            signals[u.name] = np.where(total >= u.threshold, 1.0, 0.0)

        hw = vehicle_def.general.width / 2.0
        motor_speeds: list[np.ndarray] = []
        xs: list[float] = []
        for a in vehicle_def.actuators:
            tot = np.zeros(shape, dtype=float)
            for c in a.inputs:
                if c.active:
                    tot = tot + c.sign * signals[c.source]
            ms = np.clip(a.base_speed + a.gain * tot, 0.0, a.max_speed)
            motor_speeds.append(ms)
            xs.append(a.position[0] * hw)

        speed = sum(motor_speeds) / len(motor_speeds)
        sxx = sum(x * x for x in xs)
        if sxx > 0.0:
            turn_rate = sum(ms * x for ms, x in zip(motor_speeds, xs)) / sxx
        else:
            turn_rate = np.zeros(shape, dtype=float)
        return np.asarray(speed, dtype=float), np.asarray(turn_rate, dtype=float)

    # Fallback for vehicles with stateful/recurrent units: evaluate noise-free Vehicle instances
    flat_size = int(np.prod(shape))
    flat_readings = {k: v.reshape(flat_size) for k, v in sensor_readings.items()}
    v_out = np.zeros(flat_size, dtype=float)
    om_out = np.zeros(flat_size, dtype=float)
    clean_def = copy.deepcopy(vehicle_def)
    clean_def.general.noise_std = 0.0
    n_settle = max(2, len(clean_def.units) + 2)
    dummy_rng = np.random.default_rng(0)

    for idx in range(flat_size):
        veh = Vehicle(clean_def, 0.0, 0.0, 0.0, dt=dt, seed=0)
        step_r = {k: float(arr[idx]) for k, arr in flat_readings.items()}
        for _ in range(n_settle):
            veh.think(step_r)
        # Extract kinematics without adding noise or mutating position
        actuators = clean_def.actuators
        speeds = [veh.motor_speeds[a.name] for a in actuators]
        xs = [a.position[0] * veh.half_width for a in actuators]
        v_out[idx] = sum(speeds) / len(speeds)
        sxx = sum(x * x for x in xs)
        om_out[idx] = sum(s * x for s, x in zip(speeds, xs)) / sxx if sxx > 0 else 0.0

    return v_out.reshape(shape), om_out.reshape(shape)


def _polar_kinematics_on_arrays(
    vehicle_def: VehicleDefinition,
    world: World,
    R: np.ndarray,
    PSI: np.ndarray,
    source_pos: tuple[float, float],
    stimulus_name: str | None,
    topo_order: list[int] | None = None,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic ``(speed, turn_rate)`` at arbitrary ``(R, PSI)`` arrays."""
    hw = vehicle_def.general.width / 2.0
    hl = vehicle_def.general.length / 2.0
    cos_psi = np.cos(PSI)
    sin_psi = np.sin(PSI)
    shape = R.shape

    readings: dict[str, np.ndarray] = {}
    for s in vehicle_def.sensors:
        if s.type == ENERGY_SENSOR or s.camera:
            continue
        srcs = _sources_for_polar_sensor(s.type, world, source_pos, stimulus_name)
        if not srcs:
            readings[s.name] = np.zeros(shape, dtype=float)
            continue
        tx = s.position[0] * hw
        ty = s.position[1] * hl
        # Canonical placement: source at (0, 0), vehicle at (-R, 0), heading = -PSI.
        dx = -R + ty * cos_psi - tx * sin_psi
        dy = -ty * sin_psi - tx * cos_psi
        d2 = dx * dx + dy * dy

        val = np.zeros(shape, dtype=float)
        for src in srcs:
            contrib = src.intensity / (1.0 + d2 / (src.falloff * src.falloff))
            if s.field_of_view < 360.0:
                look_heading = -PSI + math.radians(s.direction)
                off = (np.arctan2(-dy, -dx) - look_heading + np.pi) % (2.0 * np.pi) - np.pi
                mask = (np.degrees(np.abs(off)) <= s.field_of_view / 2.0) | (d2 == 0.0)
                contrib = np.where(mask, contrib, 0.0)
            val = val + contrib
        readings[s.name] = val

    return evaluate_signals_to_kinematics(vehicle_def, readings, shape, topo_order=topo_order, dt=dt)


def polar_field(
    vehicle_def: VehicleDefinition,
    world: World,
    r_grid: np.ndarray | None = None,
    psi_grid: np.ndarray | None = None,
    source_pos: tuple[float, float] | None = None,
    stimulus_name: str | None = None,
    dt: float = 0.05,
) -> dict:
    """Compute the SO(2)-reduced cylindrical polar phase field (dr/dt, dpsi/dt).

    Returns
    -------
    dict
        ``{"r": (Nr,), "psi": (Npsi,), "dr": (Npsi, Nr), "dpsi": (Npsi, Nr),
           "speed": (Npsi, Nr), "turn_rate": (Npsi, Nr), "source_pos": (sx, sy)}``
    """
    r = np.linspace(1.5, 48.0, 96) if r_grid is None else np.asarray(r_grid, dtype=float).ravel()
    psi = (
        np.linspace(-np.pi, np.pi, 97)
        if psi_grid is None
        else np.asarray(psi_grid, dtype=float).ravel()
    )
    stim_name, src_pos = resolve_primary_source(vehicle_def, world, source_pos, stimulus_name)
    can_vec, topo_order = _can_vectorize_stateless(vehicle_def)
    order_arg = topo_order if can_vec else None

    R, PSI = np.meshgrid(r, psi, indexing="xy")
    speed, turn_rate = _polar_kinematics_on_arrays(
        vehicle_def, world, R, PSI, src_pos, stim_name, topo_order=order_arg, dt=dt
    )
    r_safe = np.maximum(R, 1e-9)
    dr = -speed * np.cos(PSI)
    dpsi = (speed / r_safe) * np.sin(PSI) - turn_rate

    return {
        "r": r,
        "psi": psi,
        "dr": dr,
        "dpsi": dpsi,
        "speed": speed,
        "turn_rate": turn_rate,
        "source_pos": src_pos,
        "stimulus_name": stim_name,
    }


def arena_kinematics_field(
    vehicle_def: VehicleDefinition,
    world: World,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    heading: float = 0.0,
    heading_mode: str = "fixed",
    time: float = 0.0,
    dt: float = 0.05,
) -> dict:
    """Evaluate instantaneous (vx, vy, speed, turn_rate) over a 2D arena grid.

    Useful for real-time HUD overlays and studio streamline backdrops in multi-source worlds.
    ``heading_mode`` may be ``"fixed"``, ``"toward_primary"``, or ``"tangent_primary"``.
    """
    x = (
        np.linspace(2.0, world.arena.width - 2.0, 26)
        if x_grid is None
        else np.asarray(x_grid, dtype=float).ravel()
    )
    y = (
        np.linspace(2.0, world.arena.height - 2.0, 26)
        if y_grid is None
        else np.asarray(y_grid, dtype=float).ravel()
    )
    X, Y = np.meshgrid(x, y, indexing="xy")
    shape = X.shape

    _, src_pos = resolve_primary_source(vehicle_def, world)
    if heading_mode == "toward_primary":
        dx_p, dy_p = world.arena.delta(src_pos[0] - X, src_pos[1] - Y)
        TH = np.arctan2(dy_p, dx_p)
    elif heading_mode == "tangent_primary":
        dx_p, dy_p = world.arena.delta(src_pos[0] - X, src_pos[1] - Y)
        TH = np.arctan2(dy_p, dx_p) - 0.5 * np.pi
    else:
        TH = np.full(shape, float(heading), dtype=float)

    c = np.cos(TH)
    s = np.sin(TH)
    hw = vehicle_def.general.width / 2.0
    hl = vehicle_def.general.length / 2.0

    readings: dict[str, np.ndarray] = {}
    for sen in vehicle_def.sensors:
        if sen.type == ENERGY_SENSOR or sen.camera:
            continue
        if sen.type not in world.stimuli:
            readings[sen.name] = np.zeros(shape, dtype=float)
            continue
        tx, ty = sen.position[0] * hw, sen.position[1] * hl
        sx_arr = X + ty * c + tx * s
        sy_arr = Y + ty * s - tx * c
        readings[sen.name] = np.asarray(world.stimulus_at(sen.type, sx_arr, sy_arr, time=time), dtype=float)

    can_vec, topo_order = _can_vectorize_stateless(vehicle_def)
    speed, turn_rate = evaluate_signals_to_kinematics(
        vehicle_def, readings, shape, topo_order=topo_order if can_vec else None, dt=dt
    )
    vx = speed * c
    vy = speed * s
    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "turn_rate": turn_rate,
        "heading": TH,
    }


def classify_jacobian_2d(J: np.ndarray, tol: float = 1e-5) -> tuple[str, list[complex]]:
    """Classify a 2x2 equilibrium Jacobian into Strogatz's six canonical linear types."""
    eigs = np.linalg.eigvals(J)
    eigs_list = [complex(eigs[0]), complex(eigs[1])]
    im = abs(eigs[0].imag)
    re0, re1 = float(eigs[0].real), float(eigs[1].real)

    if im > tol:
        if re0 < -tol:
            return "stable_spiral", eigs_list
        if re0 > tol:
            return "unstable_spiral", eigs_list
        return "center", eigs_list

    if re0 * re1 < -(tol * tol):
        return "saddle", eigs_list
    if max(re0, re1) < -tol:
        return "stable_node", eigs_list
    if min(re0, re1) > tol:
        return "unstable_node", eigs_list
    return "center", eigs_list


def _integrate_single_source_trajectory(
    vehicle_def: VehicleDefinition,
    world: World,
    source_pos: tuple[float, float],
    stimulus_name: str | None,
    r0: float,
    psi0: float,
    steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate one noise-free Vehicle in unbounded single-source coordinates."""
    clean_def = copy.deepcopy(vehicle_def)
    clean_def.general.noise_std = 0.0
    sx, sy = source_pos
    # Place at angle phi = pi (so vector vehicle->source is (+r0, 0), phi_vs = 0 => heading = -psi0)
    x0 = sx - r0
    y0 = sy
    heading0 = -psi0
    veh = Vehicle(clean_def, x0, y0, heading0, dt=dt, seed=0)
    rng = np.random.default_rng(0)

    sensor_sources = {
        s.name: _sources_for_polar_sensor(s.type, world, source_pos, stimulus_name)
        for s in clean_def.sensors
        if s.type != ENERGY_SENSOR and not s.camera
    }

    xs = np.empty(steps, dtype=float)
    ys = np.empty(steps, dtype=float)
    rs = np.empty(steps, dtype=float)
    psis = np.empty(steps, dtype=float)
    speeds = np.empty(steps, dtype=float)

    for i in range(steps):
        readings: dict[str, float] = {}
        for name, _stype, px, py in veh.sensor_points():
            val = 0.0
            for src in sensor_sources.get(name, ()):
                dx = px - sx
                dy = py - sy
                d2 = dx * dx + dy * dy
                val += src.intensity / (1.0 + d2 / (src.falloff * src.falloff))
            readings[name] = val
        veh.step(readings, dt, rng)
        xs[i] = veh.x
        ys[i] = veh.y
        dx_vs = sx - veh.x
        dy_vs = sy - veh.y
        rs[i] = math.hypot(dx_vs, dy_vs)
        psis[i] = (math.atan2(dy_vs, dx_vs) - veh.heading + math.pi) % (2.0 * math.pi) - math.pi
        speeds[i] = veh.speed

    return xs, ys, rs, psis, speeds


def find_fixed_points_and_cycles(
    vehicle_def: VehicleDefinition,
    world: World,
    r_bounds: tuple[float, float] = (1.5, 48.0),
    dt: float = 0.05,
) -> dict:
    """Locate polar equilibria (dr=0, dpsi=0), classify their Jacobians, and detect limit cycles.

    Returns
    -------
    dict
        ``{"fixed_points": list[dict], "limit_cycle": dict | None}``
    """
    r_min, r_max = float(r_bounds[0]), float(r_bounds[1])
    stim_name, src_pos = resolve_primary_source(vehicle_def, world)
    can_vec, topo_order = _can_vectorize_stateless(vehicle_def)
    order_arg = topo_order if can_vec else None

    def eval_rhs(r_arr: np.ndarray, psi_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v, om = _polar_kinematics_on_arrays(
            vehicle_def, world, r_arr, psi_arr, src_pos, stim_name, topo_order=order_arg, dt=dt
        )
        dr = -v * np.cos(psi_arr)
        dpsi = (v / np.maximum(r_arr, 1e-9)) * np.sin(psi_arr) - om
        return dr, dpsi, v, om

    def rhs_scalar(rv: float, pv: float) -> tuple[float, float, float, float]:
        dr, dpsi, v, om = eval_rhs(np.array([rv]), np.array([pv]))
        return float(dr[0]), float(dpsi[0]), float(v[0]), float(om[0])

    candidates: list[tuple[float, float, bool]] = []  # (r*, psi*, is_deadzone_edge)

    # 1. Off-axis circular relative equilibria along cos(psi) = 0 (psi = +-pi/2) where v > 0
    rs_dense = np.linspace(r_min, r_max, 2400)
    for psi0 in (-0.5 * math.pi, 0.5 * math.pi):
        psi_vec = np.full_like(rs_dense, psi0)
        _dr, dpsi, v_vec, _om = eval_rhs(rs_dense, psi_vec)
        active_mask = v_vec > 1e-5
        sign_change = np.where(
            active_mask[:-1] & active_mask[1:] & (np.sign(dpsi[:-1]) * np.sign(dpsi[1:]) <= 0)
        )[0]
        for idx in sign_change:
            a, b = float(rs_dense[idx]), float(rs_dense[idx + 1])
            fa, fb = float(dpsi[idx]), float(dpsi[idx + 1])
            if abs(fa) < 1e-12:
                r_star = a
            elif abs(fb) < 1e-12:
                r_star = b
            else:
                for _ in range(36):
                    mid = 0.5 * (a + b)
                    fm = rhs_scalar(mid, psi0)[1]
                    if fa * fm <= 0.0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                r_star = 0.5 * (a + b)
            candidates.append((r_star, psi0, False))

    # 2. Invariant radial manifolds psi = 0 and psi = pi (where sin(psi) = 0)
    for psi0 in (0.0, math.pi):
        psi_vec = np.full_like(rs_dense, psi0)
        dr, dpsi, v_vec, _om = eval_rhs(rs_dense, psi_vec)
        # Detect motor-clipping stopping boundary (e.g. Vehicle 3a Love / 3b Explorer)
        stopped = (v_vec <= 1e-7) & (np.abs(dpsi) <= 1e-5)
        transitions = np.where(stopped[:-1] != stopped[1:])[0]
        for idx in transitions:
            a, b = float(rs_dense[idx]), float(rs_dense[idx + 1])
            sa = bool(stopped[idx])
            for _ in range(36):
                mid = 0.5 * (a + b)
                _, dp_m, vm, _ = rhs_scalar(mid, psi0)
                sm = (vm <= 1e-7) and (abs(dp_m) <= 1e-5)
                if sm == sa:
                    a = mid
                else:
                    b = mid
            candidates.append((0.5 * (a + b), psi0, True))

    # 3. General 2D off-axis search for any asymmetric equilibria outside dead zones
    rg = np.linspace(r_min, r_max, 90)
    pg = np.linspace(-math.pi, math.pi, 91)[:-1]
    RG, PG = np.meshgrid(rg, pg, indexing="xy")
    DR, DPSI, VG, _ = eval_rhs(RG, PG)
    MAG2 = DR * DR + DPSI * DPSI
    padded = np.pad(MAG2, ((1, 1), (1, 1)), mode="wrap")
    padded[:, 0] = np.inf
    padded[:, -1] = np.inf
    is_local_min = (
        (VG > 1e-4)
        & (MAG2 < 1e-2)
        & (MAG2 <= padded[1:-1, :-2])
        & (MAG2 <= padded[1:-1, 2:])
        & (MAG2 <= padded[:-2, 1:-1])
        & (MAG2 <= padded[2:, 1:-1])
    )
    for iy, ix in np.argwhere(is_local_min):
        rv, pv = float(RG[iy, ix]), float(PG[iy, ix])
        # Damped Newton polish
        for _ in range(20):
            f0, g0, vv, _ = rhs_scalar(rv, pv)
            if f0 * f0 + g0 * g0 < 1e-18 or vv <= 1e-6:
                break
            eps = 1e-5
            f_r, g_r, _, _ = rhs_scalar(min(r_max, rv + eps), pv)
            f_p, g_p, _, _ = rhs_scalar(rv, pv + eps)
            J = np.array([[(f_r - f0) / eps, (f_p - f0) / eps], [(g_r - g0) / eps, (g_p - g0) / eps]])
            try:
                delta = np.linalg.solve(J, -np.array([f0, g0]))
            except np.linalg.LinAlgError:
                break
            rv = float(np.clip(rv + 0.7 * delta[0], r_min, r_max))
            pv = float(wrap_pi(pv + 0.7 * delta[1]))
        f_end, g_end, v_end, _ = rhs_scalar(rv, pv)
        if math.hypot(f_end, g_end) < 1e-5 and v_end > 1e-5:
            candidates.append((rv, pv, False))

    # Deduplicate and classify fixed points
    fixed_points: list[dict] = []
    for r_star, psi_star, is_edge in candidates:
        psi_canon = math.pi if abs(abs(psi_star) - math.pi) < 1e-4 else float(wrap_pi(psi_star))
        if any(
            abs(fp["r"] - r_star) < 0.15 and abs( float(wrap_pi(fp["psi"] - psi_canon)) ) < 0.08
            for fp in fixed_points
        ):
            continue

        eps_r = 1e-3 if is_edge else 1e-4
        eps_p = 1e-4
        if is_edge:
            # One-sided derivative into the active region where v > 0
            _, _, v_plus, _ = rhs_scalar(min(r_max, r_star + 2.0 * eps_r), psi_canon)
            r_probe = r_star + eps_r if v_plus > 1e-6 else max(r_min, r_star - eps_r)
            f0, g0, _, _ = rhs_scalar(r_star, psi_canon)
            f1, g1, _, _ = rhs_scalar(r_probe, psi_canon)
            step_r = r_probe - r_star
            df_dr = (f1 - f0) / step_r
            dg_dr = (g1 - g0) / step_r
            fp_plus, gp_plus, _, _ = rhs_scalar(r_probe, psi_canon + eps_p)
            fp_minus, gp_minus, _, _ = rhs_scalar(r_probe, psi_canon - eps_p)
            df_dp = (fp_plus - fp_minus) / (2.0 * eps_p)
            dg_dp = (gp_plus - gp_minus) / (2.0 * eps_p)
        else:
            r_hi = min(r_max + 1.0, r_star + eps_r)
            r_lo = max(0.2, r_star - eps_r)
            f_hi, g_hi, _, _ = rhs_scalar(r_hi, psi_canon)
            f_lo, g_lo, _, _ = rhs_scalar(r_lo, psi_canon)
            df_dr = (f_hi - f_lo) / (r_hi - r_lo)
            dg_dr = (g_hi - g_lo) / (r_hi - r_lo)
            fp_plus, gp_plus, _, _ = rhs_scalar(r_star, psi_canon + eps_p)
            fp_minus, gp_minus, _, _ = rhs_scalar(r_star, psi_canon - eps_p)
            df_dp = (fp_plus - fp_minus) / (2.0 * eps_p)
            dg_dp = (gp_plus - gp_minus) / (2.0 * eps_p)

        J = np.array([[df_dr, df_dp], [dg_dr, dg_dp]], dtype=float)
        cls_name, eigs = classify_jacobian_2d(J)
        _, _, v_star, om_star = rhs_scalar(r_star, psi_canon)
        stability = (
            "stable"
            if cls_name in ("stable_node", "stable_spiral")
            else ("saddle" if cls_name == "saddle" else "unstable")
        )
        fixed_points.append(
            {
                "r": float(r_star),
                "psi": float(psi_canon),
                "classification": cls_name,
                "stability": stability,
                "eigenvalues": eigs,
                "jacobian": J.tolist(),
                "speed": float(v_star),
                "turn_rate": float(om_star),
            }
        )

    # Detect attracting orbital limit cycle around the source in noiseless simulation
    limit_cycle: dict | None = None
    r_mid = 0.5 * (r_min + r_max)
    probe_ics = [
        (0.45 * r_mid, 0.45 * math.pi),
        (0.85 * r_mid, 0.35 * math.pi),
        (1.25 * r_mid, 0.35 * math.pi),
    ]
    sim_steps = 3200
    sx, sy = src_pos
    for r0, psi0 in probe_ics:
        xs, ys, rs, psis, speeds = _integrate_single_source_trajectory(
            vehicle_def, world, src_pos, stim_name, r0, psi0, sim_steps, dt
        )
        half = sim_steps // 2
        r_tail = rs[half:]
        v_tail = speeds[half:]
        if r_tail.min() < 1.2 or r_tail.max() > 1.8 * r_max or v_tail.mean() < 1e-2:
            continue
        phis = np.unwrap(np.arctan2(ys[half:] - sy, xs[half:] - sx))
        total_winding = abs(float(phis[-1] - phis[0]))
        if total_winding < 4.0 * math.pi:
            continue
        # Check convergence (steady amplitude over last two quarters)
        q3 = rs[half : half + (sim_steps - half) // 2]
        q4 = rs[half + (sim_steps - half) // 2 :]
        if abs(float(q4.mean() - q3.mean())) > 0.5:
            continue

        # Extract one full orbital revolution |phi - phi_start| in [0, 2*pi]
        dphi = np.abs(phis - phis[0])
        one_rev_idx = int(np.searchsorted(dphi, 2.0 * math.pi))
        one_rev_idx = max(8, min(one_rev_idx + 1, len(r_tail)))
        period = float(2.0 * math.pi / abs((phis[one_rev_idx - 1] - phis[0]) / ((one_rev_idx - 1) * dt)))

        limit_cycle = {
            "r_orbit": r_tail[:one_rev_idx].copy(),
            "psi_orbit": psis[half : half + one_rev_idx].copy(),
            "x_orbit": xs[half : half + one_rev_idx].copy(),
            "y_orbit": ys[half : half + one_rev_idx].copy(),
            "r_mean": float(q4.mean()),
            "r_min": float(q4.min()),
            "r_max": float(q4.max()),
            "period": period,
            "speed_mean": float(v_tail.mean()),
        }
        break

    return {
        "fixed_points": fixed_points,
        "limit_cycle": limit_cycle,
    }


def _apply_sweep_parameter(
    vehicle_def: VehicleDefinition,
    world: World,
    param: str,
    val: float,
) -> tuple[VehicleDefinition, World]:
    """Return non-mutating copies of ``(vehicle_def, world)`` with ``param = val``."""
    vdef = copy.deepcopy(vehicle_def)
    vdef.general.noise_std = 0.0
    w = copy.deepcopy(world)

    if param in ("peak_at", "at"):
        vdef.sensors = [
            replace(s, response=replace(s.response, at=float(val)))
            if s.response.type in ("peak", "bell", "step", "threshold")
            else s
            for s in vdef.sensors
        ]
    elif param in ("peak_width", "width"):
        vdef.sensors = [
            replace(s, response=replace(s.response, width=max(float(val), 1e-3)))
            if s.response.type in ("peak", "bell", "inverse")
            else s
            for s in vdef.sensors
        ]
    elif param == "base_speed":
        vdef.actuators = [replace(a, base_speed=max(float(val), 0.0)) for a in vdef.actuators]
    elif param == "gain":
        vdef.actuators = [replace(a, gain=float(val)) for a in vdef.actuators]
    elif param in ("intensity", "lamp_intensity"):
        for st in w.stimuli.values():
            st.sources = [replace(src, intensity=float(val)) for src in st.sources]
    elif param == "falloff":
        for st in w.stimuli.values():
            st.sources = [replace(src, falloff=max(float(val), 1e-3)) for src in st.sources]
    else:
        raise ValueError(f"Unsupported bifurcation sweep parameter: {param!r}")
    return vdef, w


def bifurcation_sweep(
    vehicle_def: VehicleDefinition,
    world: World,
    param: str = "peak_at",
    values: np.ndarray | list[float] | None = None,
    steps: int = 2500,
    dt: float = 0.05,
) -> dict:
    """Sweep a control parameter and characterize asymptotic radial attractors.

    Classifies each parameter slice into ``"limit_cycle"``, ``"capture"``, or ``"escape"``.
    Vectorizes across a probe ring of initial conditions per parameter value for speed.
    """
    if values is None:
        defaults = {
            "peak_at": np.linspace(1.2, 11.2, 32),
            "at": np.linspace(1.2, 11.2, 32),
            "peak_width": np.linspace(0.35, 4.5, 28),
            "width": np.linspace(0.35, 4.5, 28),
            "base_speed": np.linspace(0.2, 8.5, 28),
            "gain": np.linspace(0.5, 11.0, 28),
            "intensity": np.linspace(2.5, 22.0, 28),
            "lamp_intensity": np.linspace(2.5, 22.0, 28),
            "falloff": np.linspace(5.0, 28.0, 28),
        }
        vals = defaults.get(param, np.linspace(1.0, 10.0, 25))
    else:
        vals = np.asarray(values, dtype=float).ravel()

    n_vals = len(vals)
    r_min_out = np.empty(n_vals, dtype=float)
    r_max_out = np.empty(n_vals, dtype=float)
    r_mean_out = np.empty(n_vals, dtype=float)
    attractor_types: list[str] = []

    for i, val in enumerate(vals):
        vdef_i, world_i = _apply_sweep_parameter(vehicle_def, world, param, float(val))
        stim_name, (sx, sy) = resolve_primary_source(vdef_i, world_i)
        srcs = _sources_for_polar_sensor(stim_name, world_i, (sx, sy), stim_name) if stim_name else []

        # Include a smart initial radius near the theoretical peak reading if applicable
        r_probes = [5.0, 10.0, 16.0, 24.0, 34.0]
        if srcs and vdef_i.sensors and vdef_i.sensors[0].response.type in ("peak", "bell"):
            at_val = vdef_i.sensors[0].response.at
            if 0.05 < at_val < srcs[0].intensity:
                r_est = srcs[0].falloff * math.sqrt(srcs[0].intensity / at_val - 1.0)
                if 2.0 <= r_est <= 80.0:
                    r_probes.append(r_est)

        probe_r = np.array(r_probes + [8.0], dtype=float)
        probe_psi = np.array([0.42 * math.pi] * len(r_probes) + [0.05], dtype=float)
        m = len(probe_r)

        px = sx - probe_r
        py = np.full(m, sy, dtype=float)
        heading = -probe_psi
        hw = vdef_i.general.width / 2.0
        hl = vdef_i.general.length / 2.0
        can_vec, topo_order = _can_vectorize_stateless(vdef_i)
        order_arg = topo_order if can_vec else None
        sensor_srcs = {
            s.name: _sources_for_polar_sensor(s.type, world_i, (sx, sy), stim_name)
            for s in vdef_i.sensors
            if s.type != ENERGY_SENSOR and not s.camera
        }

        tail_len = max(100, steps // 4)
        tail_start = steps - tail_len
        r_hist = np.empty((tail_len, m), dtype=float)
        phi_hist = np.empty((tail_len, m), dtype=float)
        v_hist = np.empty((tail_len, m), dtype=float)

        for step in range(steps):
            c = np.cos(heading)
            s_ang = np.sin(heading)
            readings: dict[str, np.ndarray] = {}
            for sen in vdef_i.sensors:
                if sen.type == ENERGY_SENSOR or sen.camera:
                    continue
                tx, ty = sen.position[0] * hw, sen.position[1] * hl
                spx = px + ty * c + tx * s_ang - sx
                spy = py + ty * s_ang - tx * c - sy
                d2 = spx * spx + spy * spy
                acc = np.zeros(m, dtype=float)
                for src in sensor_srcs.get(sen.name, ()):
                    acc = acc + src.intensity / (1.0 + d2 / (src.falloff * src.falloff))
                readings[sen.name] = acc

            v_step, om_step = evaluate_signals_to_kinematics(
                vdef_i, readings, (m,), topo_order=order_arg, dt=dt
            )
            heading = heading + om_step * dt
            px = px + v_step * np.cos(heading) * dt
            py = py + v_step * np.sin(heading) * dt

            if step >= tail_start:
                k = step - tail_start
                dx = px - sx
                dy = py - sy
                r_hist[k] = np.hypot(dx, dy)
                phi_hist[k] = np.arctan2(dy, dx)
                v_hist[k] = v_step

        # Classify best bounded attractor among the probes
        chosen_type = "escape"
        chosen_idx = 0
        for p in range(m):
            rp = r_hist[:, p]
            vp = v_hist[:, p]
            unwrapped_phi = np.unwrap(phi_hist[:, p])
            winding = abs(float(unwrapped_phi[-1] - unwrapped_phi[0]))
            dr_net = float(rp[-1] - rp[0])

            if rp.max() < 95.0 and abs(dr_net) < 3.0:
                if vp.mean() < 0.05 or rp.min() < 1.2:
                    chosen_type = "capture"
                    chosen_idx = p
                    break
                if winding > 1.2 * math.pi:
                    chosen_type = "limit_cycle"
                    chosen_idx = p
                    break
                chosen_type = "capture"
                chosen_idx = p

        rp_sel = r_hist[:, chosen_idx]
        r_min_out[i] = float(rp_sel.min())
        r_max_out[i] = float(rp_sel.max())
        r_mean_out[i] = float(rp_sel.mean())
        attractor_types.append(chosen_type)

    return {
        "param_name": param,
        "values": vals,
        "r_min": r_min_out,
        "r_max": r_max_out,
        "r_mean": r_mean_out,
        "attractor_type": attractor_types,
    }
