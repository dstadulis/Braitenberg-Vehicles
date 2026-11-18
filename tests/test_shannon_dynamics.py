"""Tests for Axiom 1 isolation, swarm scenarios, and Claude Shannon ensemble dynamics."""

from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from braitenberg.definition import Connection, load_definition
from braitenberg.dynamics.shannon import (
    EXPERIMENTS,
    AxiomViolationError,
    ExperimentSpec,
    binary_entropy_bits,
    compute_shannon_snapshot,
    discrete_mi_bits,
    discrete_transfer_entropy_bits,
    lookup_wire_info,
    select_focal_vehicles,
    simulate_experiment_rollout,
    validate_single_experiment_axiom,
)
from braitenberg.scenario import load_scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Vehicle


EXPECTED_EXPERIMENT_IDS = (
    "v4a_orbiter",
    "v5_moth",
    "v5_forager",
    "v7_wary_moth",
    "v9_shy",
    "v12_daydreamer",
)


# ---------------------------------------------------------------------------
# 1. Axiom 1 — Strict Single-Experiment & Homogeneous Species Isolation
# ---------------------------------------------------------------------------


def test_canonical_experiments_registry_and_axiom_acceptance():
    assert tuple(EXPERIMENTS.keys()) == EXPECTED_EXPERIMENT_IDS
    for exp_id in EXPECTED_EXPERIMENT_IDS:
        spec = EXPERIMENTS[exp_id]
        assert isinstance(spec, ExperimentSpec)
        assert spec.experiment_id == exp_id
        assert len(spec.phase_axes) == 2

        # Accepted by ID, by ExperimentSpec, and as a 1-element list
        assert validate_single_experiment_axiom(exp_id) == spec
        assert validate_single_experiment_axiom(spec) == spec
        assert validate_single_experiment_axiom([spec]) == spec

        sc = load_scenario(spec.scenario_path)
        sim = Simulation(sc)
        focal = select_focal_vehicles(sim.vehicles)
        validated = validate_single_experiment_axiom(spec, vehicles=focal, scenario=sc)
        assert validated == spec


def test_axiom_rejects_multi_experiment_empty_and_cross_species_mixes():
    moth_def = load_definition("vehicles/vehicle5-moth.yaml")
    wary_def = load_definition("vehicles/vehicle7-wary-moth.yaml")
    orbiter_def = load_definition("vehicles/vehicle4a-orbiter.yaml")

    v_moth_0 = Vehicle(moth_def, 10.0, 20.0, 0.0, label="Moth #0")
    v_moth_1 = Vehicle(moth_def, 15.0, 25.0, 0.5, label="Moth #1")
    v_wary = Vehicle(wary_def, 20.0, 30.0, 0.0, label="Wary #0")
    v_orb = Vehicle(orbiter_def, 30.0, 30.0, 0.0, label="Orbiter #0")

    # Multiple specs / comma-joined / "all" / empty
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom(["v5_moth", "v7_wary_moth"])
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v5_moth,v7_wary_moth")
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("all")
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom([])
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom(None)

    # Empty focal vehicles
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v5_moth", vehicles=[])
    with pytest.raises(AxiomViolationError):
        compute_shannon_snapshot([])

    # Cross-species mixture (Moth + Wary Moth)
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v5_moth", vehicles=[v_moth_0, v_wary])
    with pytest.raises(AxiomViolationError):
        compute_shannon_snapshot([v_moth_0, v_wary])

    # Homogeneous vehicles passed to the WRONG ExperimentSpec
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v7_wary_moth", vehicles=[v_moth_0, v_moth_1])
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v4a_orbiter", scenario="scenarios/dynamics-moth-swarm.yaml")

    # Multiple scenarios rejected
    sc_moth = load_scenario("scenarios/dynamics-moth-swarm.yaml")
    sc_orb = load_scenario("scenarios/dynamics-orbiter-swarm.yaml")
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v5_moth", scenario=[sc_moth, sc_orb])

    # Subtle wire-topology mutation with same general.type & general.name must ALSO be rejected!
    mutated_moth = copy.deepcopy(moth_def)
    mutated_moth.units[0].inputs[0] = Connection(source="right_eye", sign=-1)
    v_mutated = Vehicle(mutated_moth, 12.0, 22.0, 0.0, label="Mutant Moth")
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v5_moth", vehicles=[v_moth_0, v_mutated])
    with pytest.raises(AxiomViolationError):
        compute_shannon_snapshot([v_orb, v_moth_0])


def test_shy_swarm_filters_environmental_chasers_and_walkers():
    sc = load_scenario("scenarios/dynamics-shy-swarm.yaml")
    sim = Simulation(sc)
    assert len(sim.vehicles) == 20  # 16 Shy + 2 Chasers + 2 Walkers

    # Raw unfiltered vehicles include Chaser & Walker and MUST violate Axiom 1
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom("v9_shy", vehicles=sim.vehicles)
    with pytest.raises(AxiomViolationError):
        compute_shannon_snapshot(sim.vehicles)

    # select_focal_vehicles isolates strictly the 16 Vehicle 9: Shy individuals
    focal = select_focal_vehicles(sim.vehicles)
    assert len(focal) == 16
    assert all(v.definition.general.type == "9" and v.definition.general.name == "Shy" for v in focal)
    assert focal[0].label == "Shy #0"
    assert validate_single_experiment_axiom("v9_shy", vehicles=focal, scenario=sc) == EXPERIMENTS["v9_shy"]


# ---------------------------------------------------------------------------
# 2. Six Runnable Multi-Vehicle Swarm Scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exp_id", "expected_focal_count", "expected_hero_label"),
    [
        ("v4a_orbiter", 24, "Orbiter #0"),
        ("v5_moth", 24, "Moth #0"),
        ("v5_forager", 24, "Forager #0"),
        ("v7_wary_moth", 24, "Wary Moth #0"),
        ("v9_shy", 16, "Shy #0"),
        ("v12_daydreamer", 24, "Daydreamer #0"),
    ],
)
def test_all_six_swarm_scenarios_load_and_step_deterministically(
    exp_id: str,
    expected_focal_count: int,
    expected_hero_label: str,
):
    spec = EXPERIMENTS[exp_id]
    sc1 = load_scenario(spec.scenario_path)
    sc2 = load_scenario(spec.scenario_path)

    sim1 = Simulation(sc1)
    sim2 = Simulation(sc2)

    focal1 = select_focal_vehicles(sim1.vehicles)
    assert len(focal1) == expected_focal_count
    assert focal1[0].label == expected_hero_label

    sim1.run(40)
    sim2.run(40)

    xs1 = np.array([v.x for v in sim1.vehicles])
    xs2 = np.array([v.x for v in sim2.vehicles])
    ys1 = np.array([v.y for v in sim1.vehicles])
    ys2 = np.array([v.y for v in sim2.vehicles])
    assert np.allclose(xs1, xs2, atol=1e-12)
    assert np.allclose(ys1, ys2, atol=1e-12)
    assert np.all(np.isfinite(xs1)) and np.all(np.isfinite(ys1))


# ---------------------------------------------------------------------------
# 3. Exact Claude Shannon Identities, Synergy, Exclusivity & Zero-Bit Bias
# ---------------------------------------------------------------------------


def _make_two_unit_vehicles(bit_pairs: list[tuple[int, int]]) -> list[Vehicle]:
    """Build a homogeneous N-vehicle test population with 2 binary units ('u_left', 'u_right')."""
    base_def = copy.deepcopy(load_definition("vehicles/vehicle5-moth.yaml"))
    u0 = copy.deepcopy(base_def.units[0])
    u1 = copy.deepcopy(base_def.units[1])
    u0.name = "u_left"
    u1.name = "u_right"
    base_def.units = [u0, u1]
    base_def.actuators = base_def.actuators[:2]
    for a in base_def.actuators:
        a.inputs = [Connection("u_left", 1), Connection("u_right", 1)]

    vehicles: list[Vehicle] = []
    for idx, (b0, b1) in enumerate(bit_pairs):
        v = Vehicle(base_def, float(idx), 10.0, 0.0, label=f"Synth #{idx}")
        v.unit_outputs["u_left"] = float(b0)
        v.unit_outputs["u_right"] = float(b1)
        vehicles.append(v)
    return vehicles


def test_anticorrelated_vs_correlated_swarms_distinguished_losslessly():
    # Anticorrelated / mutually exclusive swarm: 50% [1, 0], 50% [0, 1]
    anti_vehs = _make_two_unit_vehicles([(1, 0)] * 12 + [(0, 1)] * 12)
    # Correlated / co-activating swarm: 50% [1, 1], 50% [0, 0]
    corr_vehs = _make_two_unit_vehicles([(1, 1)] * 12 + [(0, 0)] * 12)
    # Independent 4-state swarm: 25% each of [0,0], [0,1], [1,0], [1,1]
    indep_vehs = _make_two_unit_vehicles([(0, 0)] * 6 + [(0, 1)] * 6 + [(1, 0)] * 6 + [(1, 1)] * 6)

    snap_anti = compute_shannon_snapshot(anti_vehs)
    snap_corr = compute_shannon_snapshot(corr_vehs)
    snap_indep = compute_shannon_snapshot(indep_vehs)

    # Marginal firing rates & marginal entropies are IDENTICAL (0.5 and 1.0 bit) in all three!
    for snap in (snap_anti, snap_corr, snap_indep):
        assert snap["part_firing_rate"]["u_left"] == pytest.approx(0.5, abs=1e-12)
        assert snap["part_firing_rate"]["u_right"] == pytest.approx(0.5, abs=1e-12)
        assert snap["part_entropy"]["u_left"] == pytest.approx(1.0, abs=1e-12)
        assert snap["part_entropy"]["u_right"] == pytest.approx(1.0, abs=1e-12)
        assert snap["marginal_entropy_sum_bits"] == pytest.approx(2.0, abs=1e-12)

    # Both anti and corr have H(W) = 1.0 bit and Watanabe synergy TC = 1.0 bit,
    # whereas independent has H(W) = 2.0 bits and TC = 0.0 bits!
    assert snap_anti["joint_entropy_bits"] == pytest.approx(1.0, abs=1e-12)
    assert snap_anti["total_correlation_bits"] == pytest.approx(1.0, abs=1e-12)
    assert snap_corr["joint_entropy_bits"] == pytest.approx(1.0, abs=1e-12)
    assert snap_corr["total_correlation_bits"] == pytest.approx(1.0, abs=1e-12)
    assert snap_indep["joint_entropy_bits"] == pytest.approx(2.0, abs=1e-12)
    assert snap_indep["total_correlation_bits"] == pytest.approx(0.0, abs=1e-12)

    # CRITICAL: part_codeword_arcs distinguishes exclusive [1,0]/[0,1] from co-active [1,1]/[0,0]!
    anti_left_cws = {cw for cw, _ in snap_anti["part_codeword_arcs"]["u_left"]}
    anti_right_cws = {cw for cw, _ in snap_anti["part_codeword_arcs"]["u_right"]}
    assert anti_left_cws.isdisjoint(anti_right_cws)

    corr_left_cws = {cw for cw, _ in snap_corr["part_codeword_arcs"]["u_left"]}
    corr_right_cws = {cw for cw, _ in snap_corr["part_codeword_arcs"]["u_right"]}
    assert corr_left_cws == corr_right_cws == {0}


def test_constant_bias_awake_unit_produces_exact_zero_bits():
    assert binary_entropy_bits(0.0) == 0.0
    assert binary_entropy_bits(1.0) == 0.0
    assert binary_entropy_bits(0.5) == pytest.approx(1.0, abs=1e-15)

    sim = Simulation(load_scenario("scenarios/dynamics-daydreamer-swarm.yaml"))
    sim.run(140)  # t = 7.0s: wake pulse has finished (wake=0) and awake=1 for all 24 vehicles
    snap = compute_shannon_snapshot(sim.vehicles)

    assert snap["part_firing_rate"]["awake"] == pytest.approx(1.0, abs=1e-12)
    assert snap["part_entropy"]["awake"] == 0.0
    assert snap["part_firing_rate"]["wake"] == pytest.approx(0.0, abs=1e-12)
    assert snap["part_entropy"]["wake"] == 0.0
    # Meanwhile, idea units diversify across the jittered swarm -> positive joint entropy!
    assert snap["joint_entropy_bits"] > 2.0


def test_transfer_entropy_and_learning_wire_gating_identities():
    # Exact causal 1-step delay test: D_t = S_{t-1}, while S_t is independent of S_{t-1}
    s_prev = np.array([0, 0, 1, 1] * 6, dtype=int)
    d_prev = np.array([0, 1, 0, 1] * 6, dtype=int)
    d_now = s_prev.copy()
    s_now = d_prev.copy()  # orthogonal to d_now = s_prev!

    assert discrete_mi_bits(s_now, d_now) == pytest.approx(0.0, abs=1e-12)
    assert discrete_transfer_entropy_bits(d_now, s_prev, d_prev) == pytest.approx(1.0, abs=1e-12)

    # Wary Moth synaptic gate test: at step 1 (w_red <= 0.7 everywhere), learning wire MI & TE == 0.0
    sim = Simulation(load_scenario("scenarios/dynamics-wary-moth-swarm.yaml"))
    sim.step()
    snap_early = compute_shannon_snapshot(sim.vehicles)
    red_key = ("sees_red", "danger", 1)
    white_key = ("sees_white", "danger", 1)
    assert snap_early["wire_active_fraction"][red_key] == 0.0
    assert snap_early["wire_mi_bits"][red_key] == 0.0
    assert snap_early["wire_te_bits"][red_key] == 0.0
    assert snap_early["barcodes"]["sees_red->danger"]["is_learning"] is True
    assert snap_early["barcodes"]["sees_red->danger"]["gate"] == pytest.approx(0.7)

    # Advance until half the swarm learns sees_red -> danger > 0.7 while sees_white stays <= 0.7
    prev = None
    for _ in range(320):
        prev = compute_shannon_snapshot(sim.vehicles, prev_snapshot=prev)
        sim.step()
    snap_late = compute_shannon_snapshot(sim.vehicles, prev_snapshot=prev)

    assert 0.15 < snap_late["wire_active_fraction"][red_key] < 0.85
    assert snap_late["wire_info_bits"][red_key] > 0.0
    assert snap_late["wire_active_fraction"][white_key] == 0.0
    assert snap_late["wire_info_bits"][white_key] == 0.0

    # Verify lookup_wire_info helper matches wire_info_bits
    for w in snap_late["hero_diagram"].wires:
        if w.source == "sees_red" and w.target == "danger":
            assert lookup_wire_info(snap_late, w) == pytest.approx(snap_late["wire_info_bits"][red_key])


@pytest.mark.parametrize("max_cw", [3, 6, 32])
def test_codeword_arc_probability_conservation_and_bounds_across_truncations(max_cw: int):
    sim = Simulation(load_scenario("scenarios/dynamics-wary-moth-swarm.yaml"))
    sim.run(260)
    snap = compute_shannon_snapshot(sim.vehicles, max_codewords=max_cw, barcode_bins=20)

    # Every part's active arc probability sum MUST equal empirical firing rate to 1e-12,
    # even when K_raw > max_codewords triggers the residual codeword bucket!
    for part_name, rate in snap["part_firing_rate"].items():
        arc_sum = sum(prob for _, prob in snap["part_codeword_arcs"][part_name])
        assert arc_sum == pytest.approx(rate, abs=1e-12), f"Arc mismatch on {part_name}"
        for cw_id, prob in snap["part_codeword_arcs"][part_name]:
            assert 0 <= cw_id <= max_cw
            assert 0.0 < prob <= 1.0

    # Shannon bounds
    for h_val in snap["part_entropy"].values():
        assert 0.0 <= h_val <= 1.0 + 1e-12
    assert snap["joint_entropy_bits"] >= 0.0
    assert snap["marginal_entropy_sum_bits"] >= 0.0
    assert snap["total_correlation_bits"] >= 0.0
    assert snap["spatial_mi_bits"] >= 0.0

    for key, info_b in snap["wire_info_bits"].items():
        assert snap["wire_mi_bits"][key] >= 0.0
        assert snap["wire_te_bits"][key] >= 0.0
        assert info_b == pytest.approx(max(snap["wire_mi_bits"][key], snap["wire_te_bits"][key]))

    # Barcodes sum to 1.0 and preserve [0, 1] range
    for bc_name, bc in snap["barcodes"].items():
        assert bc["hist"].shape == (20,)
        assert np.all(bc["hist"] >= 0.0)
        assert float(np.sum(bc["hist"])) == pytest.approx(1.0, abs=1e-12), f"Hist sum on {bc_name}"
        assert 0.0 <= bc["min"] <= bc["hero_val"] <= bc["max"] <= 1.0
        assert bc["entropy_bits"] >= 0.0

    # Sorted N x M microstate bit-raster lossless permutation check
    B = snap["binary_matrix"]
    rmat = snap["raster_matrix"]
    assert rmat.shape == B.shape
    assert set(np.unique(rmat)).issubset({0, 1})
    hero_row = snap["hero_sorted_row"]
    assert np.array_equal(rmat[hero_row], B[snap["hero_index"]])
    assert snap["raster_codeword_ids"][hero_row] == snap["vehicle_codeword_ids"][snap["hero_index"]]


# ---------------------------------------------------------------------------
# 4. Full Rollout Verification Across All 6 Canonical Experiments & Headless Rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exp_id", EXPECTED_EXPERIMENT_IDS)
def test_simulate_experiment_rollout_all_six_experiments(exp_id: str):
    roll = simulate_experiment_rollout(exp_id, steps=80, stride=4, seed=0)
    assert roll["experiment_id"] == exp_id
    assert roll["spec"] == EXPERIMENTS[exp_id]

    n_frames = roll["n_frames"]
    n_veh = roll["n_vehicles"]
    assert n_frames == 20
    assert roll["x"].shape == (n_frames, n_veh)
    assert roll["y"].shape == (n_frames, n_veh)
    assert roll["ux"].shape == (n_frames, n_veh)
    assert roll["uy"].shape == (n_frames, n_veh)
    assert roll["phase_x"].shape == (n_frames, n_veh)
    assert roll["phase_y"].shape == (n_frames, n_veh)
    assert roll["vehicle_codeword_ids"].shape == (n_frames, n_veh)
    assert len(roll["shannon_snapshots"]) == n_frames
    assert 0 <= roll["best_poster_frame"] < n_frames

    snap = roll["shannon_snapshots"][-1]
    if exp_id == "v4a_orbiter":
        assert snap["binary_matrix"].shape == (24, 4)
        assert snap["feature_names"] == ["left_eye_high", "right_eye_high", "turn_left", "turn_right"]
        assert max(snap["wire_mi_bits"].values()) > 0.0
        assert "ftle" in roll["phase_background"]
    elif exp_id == "v9_shy":
        assert n_veh == 16
        assert roll["env_targets"]["n_targets"] == 4
        assert roll["env_targets"]["x"].shape == (n_frames, 4)
        assert snap["grid_prob"]["retina"].shape == (4, 24)
        assert snap["grid_entropy"]["retina"].shape == (4, 24)
        assert snap["grid_prob"]["odd_left"].shape == (4, 4)
        assert snap["grid_entropy"]["odd_right"].shape == (4, 4)
    elif exp_id == "v12_daydreamer":
        assert roll["Z"].shape == (n_frames, 24)


def test_shannon_module_never_imports_pygame():
    assert "pygame" not in sys.modules
