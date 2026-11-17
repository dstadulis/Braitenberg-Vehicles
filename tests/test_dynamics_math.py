"""Tests for braitenberg.dynamics mathematical core (phase_plane, lagrangian, neuro)."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from braitenberg.definition import ACTIVE_STRENGTH, load_definition
from braitenberg.dynamics.lagrangian import (
    compute_ftle_and_basins,
    poincare_section,
    simulate_swarm_ensemble,
)
from braitenberg.dynamics.neuro import (
    analyze_daydreamer_criticality,
    characterize_hysteresis_and_learning,
    compute_ring_order_parameter,
    record_episode,
)
from braitenberg.dynamics.phase_plane import (
    arena_kinematics_field,
    bifurcation_sweep,
    find_fixed_points_and_cycles,
    polar_coords,
    polar_field,
    wrap_pi,
)
from braitenberg.scenario import Placement, Scenario, SimulationSettings, load_scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Vehicle
from braitenberg.world import Arena, Source, Stimulus, World, load_world

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dynamics_math_never_imports_pygame():
    """AGENTS.md architecture rule: math/engine modules must never import pygame."""
    code = (
        "import sys, braitenberg.dynamics.phase_plane, "
        "braitenberg.dynamics.lagrangian, braitenberg.dynamics.neuro; "
        "assert 'pygame' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_wrap_pi_and_polar_coords_conventions():
    assert math.isclose(wrap_pi(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(wrap_pi(2.5 * math.pi), 0.5 * math.pi, abs_tol=1e-12)
    assert -math.pi <= wrap_pi(-3.5 * math.pi) <= math.pi

    arr = wrap_pi(np.array([-3.0 * math.pi, 0.0, 2.5 * math.pi]))
    assert arr.shape == (3,)
    assert np.all((arr >= -math.pi) & (arr <= math.pi))

    # Source at (60, 50), vehicle at (40, 50):
    # heading = 0 -> facing straight at lamp => psi = 0
    r0, psi0 = polar_coords(40.0, 50.0, 0.0, (60.0, 50.0))
    assert math.isclose(float(r0), 20.0, rel_tol=1e-12)
    assert math.isclose(float(psi0), 0.0, abs_tol=1e-12)

    # Vehicle turns right (heading = -0.3) -> lamp is now to the vehicle's LEFT => psi > 0
    _, psi_left = polar_coords(40.0, 50.0, -0.3, (60.0, 50.0))
    assert math.isclose(float(psi_left), +0.3, abs_tol=1e-12)

    # Vehicle turns left (heading = +0.3) -> lamp is now to the vehicle's RIGHT => psi < 0
    _, psi_right = polar_coords(40.0, 50.0, +0.3, (60.0, 50.0))
    assert math.isclose(float(psi_right), -0.3, abs_tol=1e-12)

    # Periodic torus distance via arena.delta
    arena = Arena(100.0, 100.0, "wrap")
    r_wrap, psi_wrap = polar_coords(95.0, 50.0, 0.0, (5.0, 50.0), arena=arena)
    assert math.isclose(float(r_wrap), 10.0, abs_tol=1e-12)
    assert math.isclose(float(psi_wrap), 0.0, abs_tol=1e-12)


@pytest.mark.parametrize(
    "vehicle_file",
    [
        "vehicle2a-fear.yaml",
        "vehicle2b-aggression.yaml",
        "vehicle3a-love.yaml",
        "vehicle3b-explorer.yaml",
        "vehicle4a-orbiter.yaml",
        "vehicle4b-decider.yaml",
    ],
)
def test_polar_field_matches_noiseless_vehicle_step(vehicle_file: str):
    """Verify exact (speed, turn_rate, dr, dpsi) signs and values against headless Vehicle.step."""
    vdef = load_definition(REPO_ROOT / "vehicles" / vehicle_file)
    vdef.general.noise_std = 0.0
    world = load_world(REPO_ROOT / "worlds" / "one-light.yaml")
    sx, sy = world.stimuli["light"].sources[0].position

    r_vals = np.array([6.0, 12.0, 19.5, 30.0])
    psi_vals = np.array([-2.3, -0.9, 0.0, 0.7, 1.9])
    pf = polar_field(vdef, world, r_grid=r_vals, psi_grid=psi_vals, dt=1e-4)

    assert pf["dr"].shape == (len(psi_vals), len(r_vals))
    assert pf["dpsi"].shape == (len(psi_vals), len(r_vals))

    dt = 1e-4
    rng = np.random.default_rng(0)
    for iy, psi0 in enumerate(psi_vals):
        for ix, r0 in enumerate(r_vals):
            # Place at arbitrary rotated angle alpha around source to test SO(2) invariance too!
            alpha = 0.63
            x0 = sx - r0 * math.cos(alpha)
            y0 = sy - r0 * math.sin(alpha)
            heading0 = alpha - float(psi0)

            veh = Vehicle(vdef, x0, y0, heading0, dt=dt, seed=0)
            readings = {
                name: float(world.stimulus_at(st, px, py))
                for name, st, px, py in veh.sensor_points()
            }
            veh.step(readings, dt, rng)
            r1, psi1 = polar_coords(veh.x, veh.y, veh.heading, (sx, sy))

            fd_dr = (float(r1) - float(r0)) / dt
            fd_dpsi = float(wrap_pi(float(psi1) - float(psi0))) / dt

            assert math.isclose(veh.speed, pf["speed"][iy, ix], rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(veh.turn_rate, pf["turn_rate"][iy, ix], rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(fd_dr, pf["dr"][iy, ix], abs_tol=5e-3)
            assert math.isclose(fd_dpsi, pf["dpsi"][iy, ix], abs_tol=5e-3)


def test_fixed_points_and_orbiter_limit_cycle():
    world = load_world(REPO_ROOT / "worlds" / "one-light.yaml")

    # Vehicle 3a Love: stable node equilibrium facing the lamp at stopping boundary r ~ 8.26
    love_def = load_definition(REPO_ROOT / "vehicles" / "vehicle3a-love.yaml")
    love_res = find_fixed_points_and_cycles(love_def, world)
    assert love_res["limit_cycle"] is None
    stable_fps = [fp for fp in love_res["fixed_points"] if fp["classification"] == "stable_node"]
    assert any(abs(fp["psi"]) < 0.05 and 7.8 < fp["r"] < 8.8 for fp in stable_fps)

    # Vehicle 4a Orbiter: twin stable spirals in (r, psi) at psi = +-pi/2 and circular limit cycle in (x, y)
    orb_def = load_definition(REPO_ROOT / "vehicles" / "vehicle4a-orbiter.yaml")
    orb_res = find_fixed_points_and_cycles(orb_def, world)
    lc = orb_res["limit_cycle"]
    assert lc is not None
    assert 19.4 <= lc["r_mean"] <= 20.4
    assert 15.8 <= lc["period"] <= 17.1
    assert lc["r_orbit"].ndim == 1 and len(lc["r_orbit"]) >= 20
    assert lc["psi_orbit"].shape == lc["r_orbit"].shape

    spirals = [fp for fp in orb_res["fixed_points"] if fp["classification"] == "stable_spiral"]
    assert len(spirals) == 2
    for fp in spirals:
        assert 19.4 <= fp["r"] <= 20.4
        assert math.isclose(abs(fp["psi"]), 0.5 * math.pi, abs_tol=0.05)

    # Vehicle 2a Fear: no limit cycle (escapes)
    fear_def = load_definition(REPO_ROOT / "vehicles" / "vehicle2a-fear.yaml")
    fear_res = find_fixed_points_and_cycles(fear_def, world)
    assert fear_res["limit_cycle"] is None


def test_bifurcation_sweep_orbiter_and_non_mutation():
    orb_def = load_definition(REPO_ROOT / "vehicles" / "vehicle4a-orbiter.yaml")
    world = load_world(REPO_ROOT / "worlds" / "one-light.yaml")
    orig_at = orb_def.sensors[0].response.at
    orig_noise = orb_def.general.noise_std

    vals = np.array([3.5, 4.5, 6.0, 11.0])
    res = bifurcation_sweep(orb_def, world, param="peak_at", values=vals, steps=1800, dt=0.05)

    # Original objects must remain untouched
    assert orb_def.sensors[0].response.at == orig_at
    assert orb_def.general.noise_std == orig_noise

    assert res["param_name"] == "peak_at"
    assert res["r_mean"].shape == (4,)
    assert res["attractor_type"][0] == "limit_cycle"
    assert res["attractor_type"][1] == "limit_cycle"
    assert res["attractor_type"][2] == "limit_cycle"
    # Higher peak_at means closer to the lamp -> monotonically smaller orbit radius
    assert res["r_mean"][0] > res["r_mean"][1] > res["r_mean"][2]
    # Above maximum source intensity (10.0), peak is unreachable -> escape!
    assert res["attractor_type"][3] == "escape"


def test_arena_kinematics_field_shapes():
    orb_def = load_definition(REPO_ROOT / "vehicles" / "vehicle4a-orbiter.yaml")
    world = load_world(REPO_ROOT / "worlds" / "two-lights.yaml")
    ak = arena_kinematics_field(orb_def, world, heading_mode="toward_primary")
    assert ak["vx"].shape == (len(ak["y"]), len(ak["x"]))
    assert np.all(np.isfinite(ak["speed"]))


def test_ftle_speed_separatrix_and_periodic_unwrap_cleanliness():
    """Verify 120x120 FTLE runs in <2s and unwrapped gradients produce zero wrap-border seams."""
    orb_def = load_definition(REPO_ROOT / "vehicles" / "vehicle4a-orbiter.yaml")
    two_lights = load_world(REPO_ROOT / "worlds" / "two-lights.yaml")

    t0 = time.perf_counter()
    res = compute_ftle_and_basins(
        orb_def, two_lights, nx=120, ny=120, horizon=10.0, dt=0.05, heading_mode="toward_centroid"
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"120x120 FTLE took {elapsed:.2f}s (must be < 2.0s)"

    assert res["ftle"].shape == (120, 120)
    assert res["basin"].shape == (120, 120)
    assert np.all(np.isfinite(res["ftle"]))
    # Both lamp 0 and lamp 1 capture substantial basins, separated by positive FTLE ridges
    unique_basins = set(np.unique(res["basin"]).tolist())
    assert 0 in unique_basins and 1 in unique_basins
    assert res["basin_entropy"] > 0.3
    # Symmetry midline x = 50 has an elevated FTLE separatrix ridge
    mid_col = res["ftle"][:, 59:61].max()
    assert mid_col > float(np.median(res["ftle"]))

    # Explicit Periodic Seam Cleanliness Check:
    # In a stimulus-free wrapping arena, a constant-speed vehicle crosses x=100->0 along a curve
    # of initial conditions x0 = 100 - v*T. With unwrapped UX, D Phi_T = I everywhere so FTLE == 0!
    # A buggy wrapped gradient would spike to ~ln(100/dx)/T > 0.4 along the wrap crossing!
    dark_world = World(
        name="Dark torus",
        arena=Arena(100.0, 100.0, "wrap"),
        stimuli={"light": Stimulus("light", sources=[Source((50.0, 50.0), intensity=0.0, falloff=15.0)])},
    )
    seam_res = compute_ftle_and_basins(
        orb_def, dark_world, nx=80, ny=80, horizon=8.0, dt=0.05, heading_mode="fixed", fixed_heading=0.3
    )
    assert np.allclose(seam_res["ftle"], 0.0, atol=1e-9)


def test_swarm_ensemble_matches_simulation_and_unwraps_cleanly():
    vdef = load_definition(REPO_ROOT / "vehicles" / "vehicle2a-fear.yaml")
    world = load_world(REPO_ROOT / "worlds" / "two-lights.yaml")

    ens = simulate_swarm_ensemble(vdef, world, n_particles=60, steps=250, dt=0.05, seed=7, noise_scale=0.0)
    assert ens["t"].shape == (250,)
    assert ens["x"].shape == (250, 60)
    assert ens["ux"].shape == (250, 60)
    assert ens["basin_id"].shape == (60,)

    # Unwrapped step displacements can never jump by arena width (max speed <= 10 => step <= 0.5)
    step_jumps = np.hypot(np.diff(ens["ux"], axis=0), np.diff(ens["uy"], axis=0))
    assert float(step_jumps.max()) <= 0.6

    # Verify exact trajectory match of particle 0 against headless Simulation
    clean_def = copy.deepcopy(vdef)
    clean_def.general.noise_std = 0.0
    sc = Scenario(
        name="Single check",
        world=world,
        vehicles=[Placement(clean_def, (float(ens["x"][0, 0]), float(ens["y"][0, 0])), float(ens["heading"][0, 0]), "V0")],
        simulation=SimulationSettings(dt=0.05, seed=0),
    )
    sim = Simulation(sc)
    for k in range(1, 250):
        sim.step()
        assert math.isclose(sim.vehicles[0].x, float(ens["x"][k, 0]), abs_tol=1e-9)
        assert math.isclose(sim.vehicles[0].y, float(ens["y"][k, 0]), abs_tol=1e-9)


def test_poincare_section_crossings():
    orb_def = load_definition(REPO_ROOT / "vehicles" / "vehicle4a-orbiter.yaml")
    two_lights = load_world(REPO_ROOT / "worlds" / "two-lights.yaml")
    sec = poincare_section(orb_def, two_lights, n_ic=24, steps=800, dt=0.05, axis="x_mid", seed=0)

    assert len(sec["coord"]) > 0
    assert sec["coord"].shape == sec["psi"].shape == sec["heading"].shape == sec["ic_id"].shape == sec["direction"].shape
    assert np.all((sec["coord"] >= 0.0) & (sec["coord"] <= two_lights.arena.height))
    assert np.all((sec["psi"] >= -math.pi) & (sec["psi"] <= math.pi))
    assert set(np.unique(sec["direction"]).tolist()).issubset({-1, 1})


def test_act3_hysteresis_and_synaptic_bifurcations():
    act3 = characterize_hysteresis_and_learning()
    assert math.isclose(act3["critical_weight"], ACTIVE_STRENGTH)

    # 1. Moth irreversible fold jump
    moth = act3["moth"]
    bs = moth["burn_step"]
    assert bs > 100
    assert np.all(moth["burned"][:bs] == 0.0)
    assert np.all(moth["burned"][bs:] == 1.0)
    # Pre-burn approaches lamp (negative mean dr/dt); post-burn flees lamp (positive mean dr/dt)
    assert float(np.mean(moth["radial_velocity"][max(0, bs - 300) : bs])) < -0.5
    assert float(np.mean(moth["radial_velocity"][bs + 20 : bs + 350])) > 0.5

    # 2. Forager slow-fast relaxation oscillation
    forager = act3["forager"]
    assert float(np.nanmin(forager["energy"])) <= 30.5
    assert float(np.nanmax(forager["energy"])) >= 90.0
    assert len(forager["switch_steps"]) >= 2
    assert "exploring" in forager["cycle_phases"] and (
        "seeking" in forager["cycle_phases"] or "recharging" in forager["cycle_phases"]
    )

    # 3. Wary Moth Mnemotrix bifurcation across w_c = 0.7
    wm = act3["wary_moth"]
    assert wm["bifurcation_step"] is not None
    assert float(wm["w_red"][0]) == 0.0
    assert float(wm["w_red"].max()) > ACTIVE_STRENGTH
    assert float(wm["w_white"].max()) == 0.0

    # 4. Cautious Driver Ergotrix bifurcation across w_c = 0.7
    cd = act3["cautious_driver"]
    assert cd["bifurcation_step"] is not None
    assert float(cd["w_yellow_red"][0]) == 0.0
    assert float(cd["w_yellow_red"].max()) > ACTIVE_STRENGTH


def test_act4_ring_order_parameter_and_daydreamer_criticality():
    # Exact analytical checks on Z(t) = (1/8) sum_{k=0}^7 a_k e^{i 2 pi k / 8}
    silent_ring = compute_ring_order_parameter(np.zeros(8))
    assert silent_ring["R"] == 0.0 and silent_ring["total_activity"] == 0.0

    flood_ring = compute_ring_order_parameter(np.ones(8))
    assert math.isclose(flood_ring["R"], 0.0, abs_tol=1e-12)
    assert flood_ring["total_activity"] == 8.0

    one_hot = np.zeros(8)
    one_hot[2] = 1.0  # angle = 2 * pi * 2 / 8 = pi / 2
    pulse_ring = compute_ring_order_parameter(one_hot)
    assert math.isclose(pulse_ring["R"], 0.125, rel_tol=1e-12)
    assert math.isclose(pulse_ring["Phi"], 0.5 * math.pi, abs_tol=1e-12)
    assert math.isclose(pulse_ring["coherence"], 1.0, rel_tol=1e-12)

    # Full Vehicle 12 criticality analysis
    crit = analyze_daydreamer_criticality(
        rates=[0.04, 0.5, 1.5],
        targets=[0.1, 1.0, 6.5],
        steps=600,
        seed=0,
    )
    assert crit["presets"]["silent"]["phase"] == "quiescent"
    assert crit["presets"]["daydreamer"]["phase"] == "critical_trains"
    assert crit["presets"]["flooded"]["phase"] == "epileptic_flood"

    assert len(crit["attractor_cycle"]) >= 2
    assert crit["sequence_entropy"] > 0.2
    assert crit["transition_matrix"].shape == (8, 8)
    assert crit["R"].shape == (600,)

    swept_phases = {ph for row in crit["phase_grid"] for ph in row}
    assert swept_phases == {"quiescent", "critical_trains", "epileptic_flood"}


def test_record_episode_contract():
    ep = record_episode(load_scenario(REPO_ROOT / "scenarios" / "vehicle12-daydreamer.yaml"), steps=120)
    for key in ("t", "x", "y", "ux", "uy", "heading", "speed", "turn_rate", "energy", "r_primary"):
        assert ep[key].shape == (120,)
    assert "idea_0" in ep["units"]
    assert "idea_0" in ep["effective_thresholds"]


def test_stateful_vehicle_polar_and_swarm_fallbacks():
    """Verify fallback evaluation works cleanly for vehicles with recurrent/latch units."""
    moth_def = load_definition(REPO_ROOT / "vehicles" / "vehicle5-moth.yaml")
    one_light = load_world(REPO_ROOT / "worlds" / "one-light.yaml")
    pf = polar_field(moth_def, one_light, r_grid=np.array([15.0, 25.0]), psi_grid=np.array([-0.5, 0.0, 0.5]))
    assert pf["dr"].shape == (3, 2)
    assert np.all(np.isfinite(pf["dr"])) and np.all(np.isfinite(pf["dpsi"]))

    ens = simulate_swarm_ensemble(moth_def, one_light, n_particles=4, steps=30, dt=0.05, seed=0)
    assert ens["x"].shape == (30, 4)
    assert ens["ux"].shape == (30, 4)
    assert np.all(np.isfinite(ens["speed"]))

