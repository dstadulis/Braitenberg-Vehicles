"""Acts III & IV: Slow-fast hysteresis, synaptic bifurcations, and homeostatic criticality.

Vehicles 5 through 12 expand phase-space dimensionality beyond SE(2) via internal
state variables:
- Act III (Vehicles 5, 7, 11): Bistable self-excitatory latches (Moth's irreversible
  fold jump), slow-fast energy-latch relaxation oscillations (Forager), and
  piecewise-smooth synaptic weight bifurcations across ACTIVE_STRENGTH = 0.7
  (Wary Moth's Mnemotrix wire and Cautious Driver's Ergotrix wire).
- Act IV (Vehicle 12): Homeostatic threshold control (the caudate) stabilizing an
  8-neuron excitatory ring at the edge of chaos between quiescence and epileptic
  flooding, characterized via the Strogatz/Kuramoto circular order parameter Z(t).

This module never imports pygame (see AGENTS.md).
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from braitenberg.definition import ACTIVE_STRENGTH, ENERGY_SENSOR, VehicleDefinition, load_definition
from braitenberg.dynamics.phase_plane import polar_coords, resolve_primary_source
from braitenberg.scenario import Placement, Scenario, SimulationSettings, load_scenario
from braitenberg.sim import Simulation
from braitenberg.world import load_world

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_ring_order_parameter(activities: np.ndarray) -> dict:
    """Compute the circular complex ring order parameter Z(t) = (1/N) sum_k a_k e^{i 2 pi k / N}.

    Parameters
    ----------
    activities : array-like
        Either a 1D vector ``(N,)`` for a single instant or a 2D array ``(T, N)`` over time.

    Returns
    -------
    dict
        ``{"Z": complex or (T,), "R": float or (T,), "Phi": float or (T,),
           "total_activity": float or (T,), "coherence": float or (T,)}``
    """
    arr = np.asarray(activities, dtype=float)
    is_1d = arr.ndim == 1
    mat = arr[None, :] if is_1d else arr
    n_ring = mat.shape[1]
    if n_ring == 0:
        z = np.zeros(mat.shape[0], dtype=complex)
        r = np.zeros(mat.shape[0], dtype=float)
        phi = np.zeros(mat.shape[0], dtype=float)
        tot = np.zeros(mat.shape[0], dtype=float)
        coh = np.zeros(mat.shape[0], dtype=float)
    else:
        angles = 2.0 * np.pi * np.arange(n_ring, dtype=float) / float(n_ring)
        phasors = np.exp(1j * angles)
        raw_sum = mat @ phasors
        z = raw_sum / float(n_ring)
        r = np.abs(z)
        phi = np.angle(z)
        tot = np.sum(mat, axis=1)
        coh = np.where(tot > 1e-12, np.abs(raw_sum) / np.maximum(tot, 1e-12), 0.0)

    if is_1d:
        return {
            "Z": complex(z[0]),
            "R": float(r[0]),
            "Phi": float(phi[0]),
            "total_activity": float(tot[0]),
            "coherence": float(coh[0]),
        }
    return {
        "Z": z,
        "R": r,
        "Phi": phi,
        "total_activity": tot,
        "coherence": coh,
    }


def record_episode(
    scenario: Scenario | str | Path,
    steps: int = 2000,
    vehicle_index: int = 0,
) -> dict:
    """Step ``Simulation(scenario)`` headlessly and record full dynamical & neural traces."""
    sc = load_scenario(scenario) if isinstance(scenario, (str, Path)) else scenario
    sim = Simulation(sc)
    veh = sim.vehicles[vehicle_index]
    d = veh.definition
    arena = sim.world.arena

    stim_name, src_pos = resolve_primary_source(d, sim.world)
    all_primary_sources: list[tuple[float, float]] = []
    if stim_name is not None and stim_name in sim.world.stimuli:
        all_primary_sources = [
            (float(s.position[0]), float(s.position[1])) for s in sim.world.stimuli[stim_name].sources
        ]

    t_arr = np.empty(steps, dtype=float)
    x_arr = np.empty(steps, dtype=float)
    y_arr = np.empty(steps, dtype=float)
    ux_arr = np.empty(steps, dtype=float)
    uy_arr = np.empty(steps, dtype=float)
    heading_arr = np.empty(steps, dtype=float)
    speed_arr = np.empty(steps, dtype=float)
    turn_arr = np.empty(steps, dtype=float)
    energy_arr = np.full(steps, np.nan, dtype=float)
    r_primary_arr = np.empty(steps, dtype=float)
    r_nearest_arr = np.empty(steps, dtype=float)
    psi_primary_arr = np.empty(steps, dtype=float)

    unit_traces = {u.name: np.empty(steps, dtype=float) for u in d.units}
    eff_thresh_traces = {u.name: np.empty(steps, dtype=float) for u in d.units}

    # Map learning keys (unit_name, input_idx) -> unique "src->dst" label
    learning_keys: list[tuple[tuple[str, int], str]] = []
    seen_labels: set[str] = set()
    for u in d.units:
        for j, c in enumerate(u.inputs):
            if c.learning is not None:
                base_lbl = f"{c.source}->{u.name}"
                lbl = base_lbl if base_lbl not in seen_labels else f"{base_lbl}[{j}]"
                seen_labels.add(lbl)
                learning_keys.append(((u.name, j), lbl))
    strength_traces = {lbl: np.empty(steps, dtype=float) for _, lbl in learning_keys}

    cur_ux = float(veh.x)
    cur_uy = float(veh.y)

    for step in range(steps):
        prev_x, prev_y = veh.x, veh.y
        sim.step()
        ddx, ddy = arena.delta(veh.x - prev_x, veh.y - prev_y)
        cur_ux += float(ddx)
        cur_uy += float(ddy)

        t_arr[step] = sim.time
        x_arr[step] = veh.x
        y_arr[step] = veh.y
        ux_arr[step] = cur_ux
        uy_arr[step] = cur_uy
        heading_arr[step] = veh.heading
        speed_arr[step] = veh.speed
        turn_arr[step] = veh.turn_rate
        if veh.energy is not None:
            energy_arr[step] = veh.energy

        rp, psip = polar_coords(veh.x, veh.y, veh.heading, src_pos, arena=arena)
        r_primary_arr[step] = float(rp)
        psi_primary_arr[step] = float(psip)

        if all_primary_sources:
            dists = [
                float(polar_coords(veh.x, veh.y, veh.heading, sp, arena=arena)[0])
                for sp in all_primary_sources
            ]
            r_nearest_arr[step] = min(dists)
        else:
            r_nearest_arr[step] = float(rp)

        for k_u, u in enumerate(d.units):
            unit_traces[u.name][step] = veh.unit_outputs[u.name]
            eff_thresh_traces[u.name][step] = float(veh.effective_thresholds[k_u])

        for key_tuple, lbl in learning_keys:
            strength_traces[lbl][step] = float(veh.strengths[key_tuple])

    return {
        "t": t_arr,
        "x": x_arr,
        "y": y_arr,
        "ux": ux_arr,
        "uy": uy_arr,
        "heading": heading_arr,
        "speed": speed_arr,
        "turn_rate": turn_arr,
        "energy": energy_arr,
        "units": unit_traces,
        "strengths": strength_traces,
        "effective_thresholds": eff_thresh_traces,
        "r_primary": r_primary_arr,
        "r_nearest": r_nearest_arr,
        "psi_primary": psi_primary_arr,
        "source_pos": src_pos,
        "active_strength_threshold": float(ACTIVE_STRENGTH),
    }


def characterize_hysteresis_and_learning() -> dict:
    """Run canonical Act III scenarios and return structured dynamical phase-space datasets.

    Covers:
    - Vehicle 5 Moth: irreversible fold/latch catastrophe jump (pre-burn attraction vs post-burn flight).
    - Vehicle 5 Forager: slow-fast relaxation oscillation in (r, energy, full).
    - Vehicle 7 Wary Moth: continuous Mnemotrix plasticity crossing ACTIVE_STRENGTH = 0.7.
    - Vehicle 11 Cautious Driver: discrete-event Ergotrix plasticity crossing ACTIVE_STRENGTH = 0.7.
    """
    scenarios_dir = REPO_ROOT / "scenarios"

    # 1. Vehicle 5 Moth: irreversible fold / latch catastrophe
    moth_sc = load_scenario(scenarios_dir / "vehicle5-moth.yaml")
    moth_ep = record_episode(moth_sc, steps=2200)
    burned = moth_ep["units"]["burned"]
    pain = moth_ep["units"]["pain"]
    burn_indices = np.flatnonzero(burned >= 0.5)
    burn_step = int(burn_indices[0]) if burn_indices.size > 0 else -1
    r_moth = moth_ep["r_nearest"]
    dt_moth = float(moth_sc.simulation.dt)
    radial_vel = np.gradient(r_moth, dt_moth)

    moth_data = {
        "episode": moth_ep,
        "t": moth_ep["t"],
        "r": r_moth,
        "speed": moth_ep["speed"],
        "turn_rate": moth_ep["turn_rate"],
        "radial_velocity": radial_vel,
        "pain": pain,
        "burned": burned,
        "burn_step": burn_step,
        "burn_time": float(moth_ep["t"][burn_step]) if burn_step >= 0 else None,
        "burn_radius": float(r_moth[burn_step]) if burn_step >= 0 else None,
        "pre_burn_mask": burned < 0.5,
        "post_burn_mask": burned >= 0.5,
    }

    # 2. Vehicle 5 Forager: slow-fast relaxation oscillation across energy [30, 90]
    forager_sc = load_scenario(scenarios_dir / "vehicle5-forager.yaml")
    forager_sc = replace(
        forager_sc,
        simulation=SimulationSettings(dt=0.05, seed=forager_sc.simulation.seed),
    )
    forager_ep = record_episode(forager_sc, steps=3600)
    full_u = forager_ep["units"]["full"]
    seeking_u = forager_ep["units"]["seeking"]
    hungry_u = forager_ep["units"]["hungry"]
    charged_u = forager_ep["units"]["charged"]
    r_forager = forager_ep["r_primary"]

    phases: list[str] = []
    for s_idx in range(len(full_u)):
        if full_u[s_idx] >= 0.5:
            phases.append("exploring")
        elif seeking_u[s_idx] >= 0.5 and r_forager[s_idx] <= 18.0:
            phases.append("recharging")
        elif seeking_u[s_idx] >= 0.5:
            phases.append("seeking")
        else:
            phases.append("searching")
    switch_steps = np.flatnonzero(np.diff(full_u) != 0) + 1

    forager_data = {
        "episode": forager_ep,
        "t": forager_ep["t"],
        "r": r_forager,
        "energy": forager_ep["energy"],
        "speed": forager_ep["speed"],
        "turn_rate": forager_ep["turn_rate"],
        "hungry": hungry_u,
        "charged": charged_u,
        "full": full_u,
        "seeking": seeking_u,
        "cycle_phases": phases,
        "switch_steps": switch_steps,
        "energy_thresholds": (30.0, 90.0),
    }

    # 3. Vehicle 7 Wary Moth: Mnemotrix learning & w_c = 0.7 bifurcation
    wm_sc = load_scenario(scenarios_dir / "vehicle7-wary-moth.yaml")
    wm_ep = record_episode(wm_sc, steps=3200)
    arena_wm = wm_sc.world.arena
    red_pos = wm_sc.world.stimuli["red"].sources[0].position
    white_pos = wm_sc.world.stimuli["white"].sources[0].position
    r_red, _ = polar_coords(wm_ep["x"], wm_ep["y"], wm_ep["heading"], red_pos, arena=arena_wm)
    r_white, _ = polar_coords(wm_ep["x"], wm_ep["y"], wm_ep["heading"], white_pos, arena=arena_wm)
    w_red = wm_ep["strengths"]["sees_red->danger"]
    w_white = wm_ep["strengths"]["sees_white->danger"]
    wm_cross = np.flatnonzero(w_red > ACTIVE_STRENGTH)
    wm_bif_step = int(wm_cross[0]) if wm_cross.size > 0 else None

    wary_moth_data = {
        "episode": wm_ep,
        "t": wm_ep["t"],
        "r": r_red,
        "r_red": r_red,
        "r_white": r_white,
        "speed": wm_ep["speed"],
        "w_red": w_red,
        "w_white": w_white,
        "pain": wm_ep["units"]["pain"],
        "danger": wm_ep["units"]["danger"],
        "avoid": wm_ep["units"]["avoid"],
        "critical_weight": float(ACTIVE_STRENGTH),
        "bifurcation_step": wm_bif_step,
        "bifurcation_time": float(wm_ep["t"][wm_bif_step]) if wm_bif_step is not None else None,
    }

    # 4. Vehicle 11 Cautious Driver: Ergotrix temporal association & w_c = 0.7 bifurcation
    cd_sc = load_scenario(scenarios_dir / "vehicle11-cautious-driver.yaml")
    cd_ep = record_episode(cd_sc, steps=4500)
    w_yr = cd_ep["strengths"]["sees_yellow->expects_red"]
    cd_cross = np.flatnonzero(w_yr > ACTIVE_STRENGTH)
    cd_bif_step = int(cd_cross[0]) if cd_cross.size > 0 else None

    cautious_driver_data = {
        "episode": cd_ep,
        "t": cd_ep["t"],
        "x": cd_ep["x"],
        "ux": cd_ep["ux"],
        "r": cd_ep["r_primary"],
        "speed": cd_ep["speed"],
        "w_yellow_red": w_yr,
        "all_strengths": cd_ep["strengths"],
        "stop": cd_ep["units"]["stop"],
        "expects_red": cd_ep["units"]["expects_red"],
        "sees_yellow": cd_ep["units"]["sees_yellow"],
        "sees_red": cd_ep["units"]["sees_red"],
        "critical_weight": float(ACTIVE_STRENGTH),
        "bifurcation_step": cd_bif_step,
        "bifurcation_time": float(cd_ep["t"][cd_bif_step]) if cd_bif_step is not None else None,
    }

    return {
        "moth": moth_data,
        "forager": forager_data,
        "wary_moth": wary_moth_data,
        "cautious_driver": cautious_driver_data,
        "critical_weight": float(ACTIVE_STRENGTH),
    }


def _extract_thought_transitions_and_cycle(
    idea_matrix: np.ndarray,
    warmup_steps: int = 50,
) -> tuple[np.ndarray, float, list[int], list[int]]:
    """Extract idea-to-idea 8x8 transition matrix, sequence entropy, and tail cycle."""
    n_ideas = idea_matrix.shape[1]
    mat = idea_matrix[warmup_steps:] if len(idea_matrix) > warmup_steps else idea_matrix
    counts = np.zeros((n_ideas, n_ideas), dtype=float)

    # Track bout sequence of active ideas when 1..3 ideas are active (ignoring silent/flood steps)
    sequence: list[int] = []
    prev_active: tuple[int, ...] = ()
    for row in mat:
        on_tuple = tuple(int(k) for k in np.flatnonzero(row >= 0.5))
        if not on_tuple or len(on_tuple) >= n_ideas:
            prev_active = on_tuple
            continue
        if on_tuple != prev_active:
            if prev_active and len(prev_active) < n_ideas:
                for i_src in prev_active:
                    for j_dst in on_tuple:
                        if i_src != j_dst or len(on_tuple) == 1:
                            counts[i_src, j_dst] += 1.0
            # Representative primary idea for scalar sequence tracking
            rep = on_tuple[0]
            if not sequence or sequence[-1] != rep:
                sequence.append(rep)
            prev_active = on_tuple

    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )

    # Sequence entropy over bigram transitions in the compressed thought sequence
    seq_entropy = 0.0
    if len(sequence) >= 2:
        bigrams: dict[tuple[int, int], int] = {}
        for a, b in zip(sequence[:-1], sequence[1:]):
            bigrams[(a, b)] = bigrams.get((a, b), 0) + 1
        total_bg = float(len(sequence) - 1)
        probs = np.array([v / total_bg for v in bigrams.values()], dtype=float)
        seq_entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-15))))
        if abs(seq_entropy) < 1e-15:
            seq_entropy = 0.0

    # Detect exact periodic attractor cycle at the tail of ``sequence``
    cycle: list[int] = []
    seq_len = len(sequence)
    if seq_len >= 4:
        max_period = min(16, seq_len // 2)
        for p in range(1, max_period + 1):
            if sequence[-p:] == sequence[-2 * p : -p]:
                cycle = list(sequence[-p:])
                break
        if not cycle:
            cycle = sequence[-min(8, seq_len) :]

    return transition_matrix, seq_entropy, sequence, cycle


def _classify_criticality_phase(mean_activity: float, max_activity: float, seq_entropy: float) -> str:
    """Classify a Vehicle 12 run into quiescent, critical_trains, or epileptic_flood."""
    if mean_activity < 0.08 or max_activity < 0.5:
        return "quiescent"
    if mean_activity > 3.2:
        return "epileptic_flood"
    return "critical_trains"


def _analyze_single_daydreamer_scenario(scenario: Scenario, steps: int) -> dict:
    """Run one Vehicle 12 scenario and compute full ring order parameter & sequence metrics."""
    ep = record_episode(scenario, steps=steps)
    idea_names = [f"idea_{k}" for k in range(8) if f"idea_{k}" in ep["units"]]
    idea_matrix = (
        np.column_stack([ep["units"][name] for name in idea_names])
        if idea_names
        else np.zeros((steps, 8), dtype=float)
    )
    ring = compute_ring_order_parameter(idea_matrix)
    warmup = min(50, steps // 5)
    tmat, seq_ent, seq, cycle = _extract_thought_transitions_and_cycle(idea_matrix, warmup_steps=warmup)
    tail_act = ring["total_activity"][warmup:]
    mean_act = float(np.mean(tail_act)) if tail_act.size else 0.0
    max_act = float(np.max(tail_act)) if tail_act.size else 0.0
    phase = _classify_criticality_phase(mean_act, max_act, seq_ent)

    eff_mat = (
        np.column_stack([ep["effective_thresholds"][name] for name in idea_names])
        if idea_names
        else np.zeros((steps, 8), dtype=float)
    )
    caudate_offset = ep["units"].get("caudate", np.zeros(steps, dtype=float))

    return {
        "episode": ep,
        "t": ep["t"],
        "x": ep["x"],
        "y": ep["y"],
        "ux": ep["ux"],
        "uy": ep["uy"],
        "heading": ep["heading"],
        "speed": ep["speed"],
        "turn_rate": ep["turn_rate"],
        "idea_matrix": idea_matrix,
        "Z": ring["Z"],
        "R": ring["R"],
        "Phi": ring["Phi"],
        "coherence": ring["coherence"],
        "total_activity": ring["total_activity"],
        "caudate_offset": caudate_offset,
        "effective_threshold_mean": np.mean(eff_mat, axis=1),
        "mean_activity": mean_act,
        "sequence_entropy": seq_ent,
        "transition_matrix": tmat,
        "thought_sequence": seq,
        "attractor_cycle": cycle,
        "phase": phase,
    }


def _make_swept_daydreamer_definition(
    base_def: VehicleDefinition,
    rate: float,
    target: float,
) -> VehicleDefinition:
    """Return a non-mutated copy of ``base_def`` with caudate ``rate`` and ``target`` replaced."""
    new_units = [
        replace(u, rate=float(rate), target=float(target)) if u.is_control else u
        for u in base_def.units
    ]
    return replace(base_def, units=new_units)


def analyze_daydreamer_criticality(
    rates: np.ndarray | list[float] | None = None,
    targets: np.ndarray | list[float] | None = None,
    steps: int = 1000,
    seed: int = 0,
) -> dict:
    """Analyze homeostatic criticality in Vehicle 12 across presets and caudate (rate, target) grid.

    Maps the three dynamical regimes:
    - ``"quiescent"`` (<A> ~ 0)
    - ``"critical_trains"`` (0 < <A> <= 3.2, structured itinerant thought cycles)
    - ``"epileptic_flood"`` (<A> -> 8, runaway saturation)
    """
    vehicles_dir = REPO_ROOT / "vehicles"
    scenarios_dir = REPO_ROOT / "scenarios"
    worlds_dir = REPO_ROOT / "worlds"

    base_def = load_definition(vehicles_dir / "vehicle12-daydreamer.yaml")
    open_field = load_world(worlds_dir / "open-field.yaml")

    rate_vals = (
        np.array([0.04, 0.12, 0.25, 0.5, 1.0, 2.0], dtype=float)
        if rates is None
        else np.asarray(rates, dtype=float).ravel()
    )
    target_vals = (
        np.array([0.1, 0.5, 1.0, 1.8, 3.2, 5.5, 7.5], dtype=float)
        if targets is None
        else np.asarray(targets, dtype=float).ravel()
    )

    # 1. Canonical presets (daydreamer, silent, flooded, distracted)
    preset_files = {
        "daydreamer": scenarios_dir / "vehicle12-daydreamer.yaml",
        "silent": scenarios_dir / "vehicle12-daydreamer-silent.yaml",
        "flooded": scenarios_dir / "vehicle12-daydreamer-flooded.yaml",
        "distracted": scenarios_dir / "vehicle12-distracted.yaml",
    }
    presets: dict[str, dict] = {}
    for pname, spath in preset_files.items():
        sc = load_scenario(spath)
        sc = replace(sc, simulation=SimulationSettings(dt=sc.simulation.dt, seed=seed))
        presets[pname] = _analyze_single_daydreamer_scenario(sc, steps=steps)

    # 2. 2D parameter sweep over caudate (rate, target) using dataclasses.replace
    n_t = len(target_vals)
    n_r = len(rate_vals)
    mean_activity_grid = np.zeros((n_t, n_r), dtype=float)
    mean_R_grid = np.zeros((n_t, n_r), dtype=float)
    entropy_grid = np.zeros((n_t, n_r), dtype=float)
    phase_grid: list[list[str]] = []
    phase_code_grid = np.zeros((n_t, n_r), dtype=int)  # 0=quiescent, 1=critical_trains, 2=epileptic_flood
    code_map = {"quiescent": 0, "critical_trains": 1, "epileptic_flood": 2}

    sweep_steps = min(steps, 600)
    warmup = min(50, sweep_steps // 5)

    for it, tgt in enumerate(target_vals):
        row_phases: list[str] = []
        for ir, rt in enumerate(rate_vals):
            swept_def = _make_swept_daydreamer_definition(base_def, rate=float(rt), target=float(tgt))
            sc = Scenario(
                name=f"Daydreamer sweep rate={rt:g} target={tgt:g}",
                world=open_field,
                vehicles=[Placement(swept_def, (50.0, 50.0), 0.5 * np.pi, "Daydreamer")],
                simulation=SimulationSettings(dt=0.05, seed=seed),
            )
            sim = Simulation(sc)
            veh = sim.vehicles[0]
            idea_mat = np.empty((sweep_steps, 8), dtype=float)
            for s_idx in range(sweep_steps):
                sim.step()
                for k in range(8):
                    idea_mat[s_idx, k] = veh.unit_outputs[f"idea_{k}"]

            ring = compute_ring_order_parameter(idea_mat[warmup:])
            _, seq_ent, _, _ = _extract_thought_transitions_and_cycle(idea_mat, warmup_steps=warmup)
            m_act = float(np.mean(ring["total_activity"]))
            max_act = float(np.max(ring["total_activity"]))
            ph = _classify_criticality_phase(m_act, max_act, seq_ent)

            mean_activity_grid[it, ir] = m_act
            mean_R_grid[it, ir] = float(np.mean(ring["R"]))
            entropy_grid[it, ir] = seq_ent
            row_phases.append(ph)
            phase_code_grid[it, ir] = code_map[ph]
        phase_grid.append(row_phases)

    canonical = presets["daydreamer"]
    return {
        "rates": rate_vals,
        "targets": target_vals,
        "mean_activity_grid": mean_activity_grid,
        "mean_R_grid": mean_R_grid,
        "entropy_grid": entropy_grid,
        "phase_grid": phase_grid,
        "phase_code_grid": phase_code_grid,
        "presets": presets,
        # Canonical Daydreamer top-level series for direct ergonomics
        "t": canonical["t"],
        "ux": canonical["ux"],
        "uy": canonical["uy"],
        "x": canonical["x"],
        "y": canonical["y"],
        "Z": canonical["Z"],
        "R": canonical["R"],
        "Phi": canonical["Phi"],
        "coherence": canonical["coherence"],
        "idea_matrix": canonical["idea_matrix"],
        "total_activity": canonical["total_activity"],
        "caudate_offset": canonical["caudate_offset"],
        "effective_threshold_mean": canonical["effective_threshold_mean"],
        "transition_matrix": canonical["transition_matrix"],
        "sequence_entropy": canonical["sequence_entropy"],
        "attractor_cycle": canonical["attractor_cycle"],
        "phase": canonical["phase"],
    }
