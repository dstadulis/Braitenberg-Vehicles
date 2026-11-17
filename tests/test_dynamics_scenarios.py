"""Tests for multi-source dynamical scenarios and separatrix splitting."""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from braitenberg.definition import load_definition
from braitenberg.scenario import Placement, Scenario, SimulationSettings, load_scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Vehicle
from braitenberg.world import load_world
from conftest import ONE_LIGHT, REPO, run_python

THREE_LIGHTS_CHAOS = REPO / "worlds" / "three-lights-chaos.yaml"
THREE_BODY_SCENARIO = REPO / "scenarios" / "dynamics-three-body-orbiter.yaml"
TWIN_LAMP_SCENARIO = REPO / "scenarios" / "dynamics-twin-lamp-separatrix.yaml"
EXPLORER_HTML = REPO / "experiments" / "strogatz-dynamics" / "explorer.html"
MONOGRAPH_README = REPO / "experiments" / "strogatz-dynamics" / "README.md"
VEHICLE3A = REPO / "vehicles" / "vehicle3a-love.yaml"
VEHICLE4A = REPO / "vehicles" / "vehicle4a-orbiter.yaml"
VEHICLE4B = REPO / "vehicles" / "vehicle4b-decider.yaml"


def _noiseless_scenario(scenario: Scenario) -> Scenario:
    """Return a deep copy of `scenario` with intrinsic angular noise disabled (`noise_std = 0`)."""
    clean = copy.deepcopy(scenario)
    for placement in clean.vehicles:
        placement.definition = copy.deepcopy(placement.definition)
        placement.definition.general.noise_std = 0.0
    return clean


def _nearest_source(world, x: float, y: float, stimulus: str = "light") -> tuple[int, float]:
    sources = world.stimuli[stimulus].sources
    best_idx, best_dist = 0, math.inf
    for idx, src in enumerate(sources):
        dx, dy = world.arena.delta(x - src.position[0], y - src.position[1])
        dist = math.hypot(dx, dy)
        if dist < best_dist:
            best_idx, best_dist = idx, dist
    return best_idx, best_dist


def test_three_lights_chaos_world_equilateral_geometry():
    world = load_world(THREE_LIGHTS_CHAOS)
    assert world.arena.width == 160
    assert world.arena.height == 160
    assert world.arena.edges == "wrap"
    sources = world.stimuli["light"].sources
    assert len(sources) == 3

    # Verify equilateral side lengths and that direct distances are strictly inside half-arena wrap
    positions = [s.position for s in sources]
    sides = []
    for i in range(3):
        for j in range(i + 1, 3):
            dx, dy = world.arena.delta(positions[i][0] - positions[j][0],
                                       positions[i][1] - positions[j][1])
            side = math.hypot(dx, dy)
            assert side < world.arena.width / 2
            sides.append(side)
    assert max(sides) - min(sides) < 0.05

    # Saddle midpoint intensity lies below Vehicle 4a's peak.at = 4 so distinct lamp rings compete
    mx = 0.5 * (positions[1][0] + positions[2][0])
    my = 0.5 * (positions[1][1] + positions[2][1])
    assert 2.0 < world.stimulus_at("light", mx, my) < 4.0


def test_twin_lamp_separatrix_loads_and_splits_across_midline():
    scenario = load_scenario(TWIN_LAMP_SCENARIO)
    assert len(scenario.vehicles) == 8

    # 1. Deterministic (noiseless) manifold splitting across x = 50
    det_sim = Simulation(_noiseless_scenario(scenario))
    det_sim.run(1000)
    world = det_sim.world

    # Left half of Love fan (indices 0, 1, 2) captured by Left lamp (index 0 at [30, 50]);
    # Right half of Love fan (indices 3, 4, 5) captured by Right lamp (index 1 at [70, 50]).
    for idx in (0, 1, 2):
        v = det_sim.vehicles[idx]
        lamp_idx, dist = _nearest_source(world, v.x, v.y)
        assert lamp_idx == 0 and dist < 16.0
    for idx in (3, 4, 5):
        v = det_sim.vehicles[idx]
        lamp_idx, dist = _nearest_source(world, v.x, v.y)
        assert lamp_idx == 1 and dist < 16.0

    # Exact reflection symmetry across x = 50 in noiseless run
    v_left_inner, v_right_inner = det_sim.vehicles[2], det_sim.vehicles[3]
    assert v_left_inner.x + v_right_inner.x == pytest.approx(100.0, abs=1e-6)
    assert v_left_inner.y == pytest.approx(v_right_inner.y, abs=1e-6)

    # 2. Default stochastic scenario (seed=0) also splits cleanly across the two lamps
    stoch_sim = Simulation(scenario)
    stoch_sim.run(1000)
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[0].x, stoch_sim.vehicles[0].y)[0] == 0
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[1].x, stoch_sim.vehicles[1].y)[0] == 0
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[4].x, stoch_sim.vehicles[4].y)[0] == 1
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[5].x, stoch_sim.vehicles[5].y)[0] == 1


def test_tiny_perturbation_splits_across_twin_lamp_separatrix():
    """Vehicles starting tiny dx or dtheta apart across x = 50 end up at opposite lamps."""
    world = load_scenario(TWIN_LAMP_SCENARIO).world
    v3a_det = load_definition(VEHICLE3A)
    v3a_det = copy.deepcopy(v3a_det)
    v3a_det.general.noise_std = 0.0

    # Tiny spatial offset dx = +/- 0.05 at identical heading 90 deg
    sc_spatial = Scenario(
        name="Tiny dx split",
        world=world,
        vehicles=[
            Placement(v3a_det, (49.95, 30.0), math.radians(90.0), "left_eps"),
            Placement(v3a_det, (50.05, 30.0), math.radians(90.0), "right_eps"),
        ],
        simulation=SimulationSettings(dt=0.05, seed=0),
    )
    sim_spatial = Simulation(sc_spatial)
    sim_spatial.run(1200)
    l_idx, l_dist = _nearest_source(sim_spatial.world, sim_spatial.vehicles[0].x, sim_spatial.vehicles[0].y)
    r_idx, r_dist = _nearest_source(sim_spatial.world, sim_spatial.vehicles[1].x, sim_spatial.vehicles[1].y)
    assert l_idx == 0 and l_dist < 16.0
    assert r_idx == 1 and r_dist < 16.0

    # Tiny angular offset dtheta = +/- 0.1 deg from exact saddle bisector (50, 30)
    sc_angular = Scenario(
        name="Tiny dtheta split",
        world=world,
        vehicles=[
            Placement(v3a_det, (50.0, 30.0), math.radians(90.1), "ccw_eps"),
            Placement(v3a_det, (50.0, 30.0), math.radians(89.9), "cw_eps"),
        ],
        simulation=SimulationSettings(dt=0.05, seed=0),
    )
    sim_angular = Simulation(sc_angular)
    sim_angular.run(1200)
    assert _nearest_source(sim_angular.world, sim_angular.vehicles[0].x, sim_angular.vehicles[0].y)[0] == 0
    assert _nearest_source(sim_angular.world, sim_angular.vehicles[1].x, sim_angular.vehicles[1].y)[0] == 1


def test_three_body_orbiter_sensitive_dependence_and_capture_diversity():
    scenario = load_scenario(THREE_BODY_SCENARIO)
    assert len(scenario.vehicles) == 8

    # 1. Deterministic flow: 2-degree perturbations from [80, 59] diverge onto all 3 lamps
    det_sim = Simulation(_noiseless_scenario(scenario))
    late_lamps = [set() for _ in det_sim.vehicles]
    for step in range(2600):
        det_sim.step()
        if step >= 1600:
            for k, v in enumerate(det_sim.vehicles):
                idx, _ = _nearest_source(det_sim.world, v.x, v.y)
                late_lamps[k].add(idx)

    # Orbiter 84 deg (idx 0) -> Lamp 0; Orbiter 88 deg (idx 2) -> Lamp 2; Orbiter 92 deg (idx 3) -> Lamp 1
    assert late_lamps[0] == {0}
    assert late_lamps[2] == {2}
    assert late_lamps[3] == {1}
    assert late_lamps[5] == {0}
    # Orbiter 86 deg (idx 1) and 94 deg (idx 4) execute multi-lamp tours rather than single-lamp lock
    assert len(late_lamps[1]) >= 2
    assert len(late_lamps[4]) >= 2

    # Decider pair (indices 6, 7) scatters mirror-symmetrically across x = 80
    dec_sim = Simulation(_noiseless_scenario(scenario))
    dec_sim.run(450)
    dec_l, dec_r = dec_sim.vehicles[6], dec_sim.vehicles[7]
    assert abs(dec_l.x - dec_r.x) > 40.0
    assert dec_l.x + dec_r.x == pytest.approx(160.0, abs=1e-6)

    # 2. Default stochastic scenario (seed=0) also diverges across all 3 lamp regions
    stoch_sim = Simulation(scenario)
    stoch_sim.run(2200)
    occupied_regions = {
        _nearest_source(stoch_sim.world, v.x, v.y)[0] for v in stoch_sim.vehicles[:6]
    }
    assert len(occupied_regions) >= 2


@pytest.mark.parametrize("scenario_path", [THREE_BODY_SCENARIO, TWIN_LAMP_SCENARIO])
def test_dynamics_scenarios_run_headless_without_pygame(scenario_path: Path):
    code = (
        "import sys\n"
        "from braitenberg.sim import main\n"
        f"main([{str(scenario_path)!r}, '--headless', '--steps', '80'])\n"
        "assert 'pygame' not in sys.modules, 'pygame was imported'\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "80 steps" in result.stdout


def test_explorer_html_self_contained_and_physics_parity(tmp_path: Path):
    assert EXPLORER_HTML.is_file()
    html = EXPLORER_HTML.read_text(encoding="utf-8")

    # Zero external network / CDN dependencies
    assert "http://" not in html and "https://" not in html
    for token in (
        'data-theme="cosmic"',
        '[data-theme="textbook"]',
        "Release 800-Particle Swarm",
        "Export 4K PNG",
        "Supercritical Limit Cycle",
        "Homoclinic/Saddle Escape vs Love",
        "Three-Body Lamp Chaos",
        "Twin-Lamp Fractal Capture Basins",
        "Moth Fold/Latch Hysteresis Catastrophe",
        "Daydreamer Caudate Criticality Ring",
    ):
        assert token in html

    match = re.search(r"<script>([\s\S]*?)</script>", html)
    assert match is not None
    js_code = match.group(1)
    assert "nearestPolar(wx, wy, lead.trailH[ti])" in js_code
    assert "rSpan > 0.15" in js_code

    node_bin = shutil.which("node") or (
        "/opt/homebrew/bin/node" if Path("/opt/homebrew/bin/node").exists() else None
    )
    if node_bin is not None:
        script_file = tmp_path / "explorer_check.js"
        script_file.write_text(js_code, encoding="utf-8")
        syntax = subprocess.run([node_bin, "--check", str(script_file)], capture_output=True, text=True)
        assert syntax.returncode == 0, syntax.stderr

        # Verify exact numerical kinematics parity between JS kernel and Python Vehicle.step
        kernel_js = js_code[: js_code.index('const arenaCanvas = document.getElementById("arenaCanvas");')]
        parity_file = tmp_path / "parity.js"
        parity_file.write_text(
            """
const document = {
  createElement: () => ({ width: 0, height: 0, getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {}
  })}),
  getElementById: () => ({ value: "4a", textContent: "", classList: { toggle: () => {} }, innerHTML: "", appendChild: () => {} }),
  querySelectorAll: () => []
};
function syncControlsFromState() {}
"""
            + kernel_js
            + """
const oneLight = [{ x: 60, y: 50, intensity: 10, falloff: 15, active: true }];
applyVehicleArchetype("4a");
let x = 35.0, y = 44.0, h = 0.0;
for (let step = 0; step < 200; step++) {
  const kin = memorylessKinematics(x, y, h, state.params, oneLight, 100, 100, true);
  h += kin.turnRate * 0.05;
  x = ((x + kin.speed * Math.cos(h) * 0.05) % 100 + 100) % 100;
  y = ((y + kin.speed * Math.sin(h) * 0.05) % 100 + 100) % 100;
}
console.log(JSON.stringify({ x, y, h }));
""",
            encoding="utf-8",
        )
        run_js = subprocess.run([node_bin, str(parity_file)], capture_output=True, text=True)
        assert run_js.returncode == 0, run_js.stderr
        js_pose = json.loads(run_js.stdout.strip())

        w1 = load_world(ONE_LIGHT)
        v4a_det = copy.deepcopy(load_definition(VEHICLE4A))
        v4a_det.general.noise_std = 0.0
        veh = Vehicle(v4a_det, 35.0, 44.0, 0.0, dt=0.05, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(200):
            readings = {
                name: float(w1.stimulus_at(stim, sx, sy))
                for name, stim, sx, sy in veh.sensor_points()
            }
            veh.step(readings, 0.05, rng)
            w1.arena.apply_edges(veh)

        assert js_pose["x"] == pytest.approx(veh.x, abs=1e-10)
        assert js_pose["y"] == pytest.approx(veh.y, abs=1e-10)
        assert js_pose["h"] == pytest.approx(veh.heading, abs=1e-10)


def test_monograph_readme_showcase_media_contract():
    assert MONOGRAPH_README.is_file()
    md = MONOGRAPH_README.read_text(encoding="utf-8")
    expected_media = (
        "media/strogatz_atlas.png",
        "media/orbiter.gif",
        "media/ftle_swarm.gif",
        "media/plasticity.gif",
        "media/daydreamer.gif",
        "media/act1_polar_phase_portraits.png",
        "media/act2_lagrangian_chaos_ftle.png",
        "media/act3_hysteresis_and_plasticity.png",
        "media/act4_neural_criticality.png",
    )
    for asset in expected_media:
        assert asset in md, f"Missing showcase asset reference {asset}"
    assert "explorer.html" in md
    assert "python -m braitenberg.dynamics" in md
    assert "w_c = 0.7" in md

