"""Tests for dynamical scenarios, Tri-Panel explorer.html, and Strogatz x Shannon monograph."""

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

CANONICAL_EXPERIMENTS: tuple[tuple[str, str, str, str], ...] = (
    ("v9_shy", "dynamics-shy-swarm.yaml", "9", "Shy"),
    ("v5_moth", "dynamics-moth-swarm.yaml", "5", "Moth"),
    ("v5_forager", "dynamics-forager-swarm.yaml", "5", "Forager"),
    ("v7_wary_moth", "dynamics-wary-moth-swarm.yaml", "7", "Wary moth"),
    ("v12_daydreamer", "dynamics-daydreamer-swarm.yaml", "12", "Daydreamer"),
    ("v4a_orbiter", "dynamics-orbiter-swarm.yaml", "4a", "Orbiter"),
)

PHASE_PORTRAIT_EXPERIMENT_IDS: tuple[str, ...] = (
    "v2a_fear",
    "v3a_love",
    "v4a_orbiter",
    "v5_moth",
    "v5_forager",
    "v7_wary_moth",
    "v11_cautious_driver",
    "v12_daydreamer",
)

PHASE_ATLAS_POSTER_STEMS: tuple[str, ...] = (
    "strogatz_atlas",
    "act1_polar_phase_portraits",
    "act2_lagrangian_chaos_ftle",
    "act3_hysteresis_and_plasticity",
    "act4_neural_criticality",
)

PHASE_ATLAS_SCENES: tuple[str, ...] = (
    "orbiter",
    "ftle_swarm",
    "plasticity",
    "daydreamer",
)


def _node_binary() -> str | None:
    return shutil.which("node") or (
        "/opt/homebrew/bin/node" if Path("/opt/homebrew/bin/node").exists() else None
    )


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

    mx = 0.5 * (positions[1][0] + positions[2][0])
    my = 0.5 * (positions[1][1] + positions[2][1])
    assert 2.0 < world.stimulus_at("light", mx, my) < 4.0


def test_twin_lamp_separatrix_loads_and_splits_across_midline():
    scenario = load_scenario(TWIN_LAMP_SCENARIO)
    assert len(scenario.vehicles) == 8

    det_sim = Simulation(_noiseless_scenario(scenario))
    det_sim.run(1000)
    world = det_sim.world

    for idx in (0, 1, 2):
        v = det_sim.vehicles[idx]
        lamp_idx, dist = _nearest_source(world, v.x, v.y)
        assert lamp_idx == 0 and dist < 16.0
    for idx in (3, 4, 5):
        v = det_sim.vehicles[idx]
        lamp_idx, dist = _nearest_source(world, v.x, v.y)
        assert lamp_idx == 1 and dist < 16.0

    v_left_inner, v_right_inner = det_sim.vehicles[2], det_sim.vehicles[3]
    assert v_left_inner.x + v_right_inner.x == pytest.approx(100.0, abs=1e-6)
    assert v_left_inner.y == pytest.approx(v_right_inner.y, abs=1e-6)

    stoch_sim = Simulation(scenario)
    stoch_sim.run(1000)
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[0].x, stoch_sim.vehicles[0].y)[0] == 0
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[1].x, stoch_sim.vehicles[1].y)[0] == 0
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[4].x, stoch_sim.vehicles[4].y)[0] == 1
    assert _nearest_source(stoch_sim.world, stoch_sim.vehicles[5].x, stoch_sim.vehicles[5].y)[0] == 1


def test_tiny_perturbation_splits_across_twin_lamp_separatrix():
    world = load_scenario(TWIN_LAMP_SCENARIO).world
    v3a_det = copy.deepcopy(load_definition(VEHICLE3A))
    v3a_det.general.noise_std = 0.0

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

    det_sim = Simulation(_noiseless_scenario(scenario))
    late_lamps = [set() for _ in det_sim.vehicles]
    for step in range(2600):
        det_sim.step()
        if step >= 1600:
            for k, v in enumerate(det_sim.vehicles):
                idx, _ = _nearest_source(det_sim.world, v.x, v.y)
                late_lamps[k].add(idx)

    assert late_lamps[0] == {0}
    assert late_lamps[2] == {2}
    assert late_lamps[3] == {1}
    assert late_lamps[5] == {0}
    assert len(late_lamps[1]) >= 2
    assert len(late_lamps[4]) >= 2

    dec_sim = Simulation(_noiseless_scenario(scenario))
    dec_sim.run(450)
    dec_l, dec_r = dec_sim.vehicles[6], dec_sim.vehicles[7]
    assert abs(dec_l.x - dec_r.x) > 40.0
    assert dec_l.x + dec_r.x == pytest.approx(160.0, abs=1e-6)

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


def test_explorer_html_self_contained_tripanel_axiom_and_shannon_identities(tmp_path: Path):
    assert EXPLORER_HTML.is_file()
    html = EXPLORER_HTML.read_text(encoding="utf-8")

    # 1. Zero external network / CDN links and zero legacy cross-species mashup controls
    assert "http://" not in html and "https://" not in html
    assert "Quartet (2a–3b)" not in html
    assert "customParams" not in html

    # 2. Required Tri-Panel canvases, themes, neural views, and 6 canonical pure experiment IDs
    for token in (
        'data-theme="cosmic"',
        '[data-theme="textbook"]',
        'id="arenaCanvas"',
        'id="phaseCanvas"',
        'id="neuralCanvas"',
        'id="selNeuralView"',
        "Export 4K Tri-Panel PNG",
        "validateSingleExperimentAxiom",
        "AxiomViolationError",
        "v9_shy",
        "v5_moth",
        "v5_forager",
        "v7_wary_moth",
        "v12_daydreamer",
        "v4a_orbiter",
    ):
        assert token in html, f"Missing required token in explorer.html: {token}"

    match = re.search(r"<script>([\s\S]*?)</script>", html)
    assert match is not None
    js_code = match.group(1)

    node_bin = _node_binary()
    if node_bin is None:
        pytest.skip("node binary not installed")

    # 3. JS syntax validation (`node --check`)
    script_file = tmp_path / "explorer_check.js"
    script_file.write_text(js_code, encoding="utf-8")
    syntax = subprocess.run([node_bin, "--check", str(script_file)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr

    # 4. Full JS runtime test: Axiom 1 enforcement, Shannon identities, click-to-switch Hero,
    #    exact kinematics parity with Python Vehicle.step, AND full 3-canvas rendering execution!
    harness_file = tmp_path / "tripanel_runtime_verify.js"
    harness_file.write_text(
        """
function makeMockCtx() {
  return {
    save() {}, restore() {}, beginPath() {}, closePath() {},
    moveTo() {}, lineTo() {}, stroke() {}, fill() {},
    fillRect() {}, strokeRect() {}, roundRect() {}, arc() {},
    translate() {}, rotate() {}, setLineDash() {},
    fillText() {}, strokeText() {}, drawImage() {},
    createRadialGradient() { return { addColorStop() {} }; },
    createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData() {}
  };
}

const elements = {};
function getEl(id) {
  if (!elements[id]) {
    elements[id] = {
      id,
      width: 840,
      height: 840,
      value: "composite",
      textContent: "",
      innerHTML: "",
      style: {},
      listeners: {},
      classList: { toggle() {}, add() {}, remove() {} },
      appendChild() {},
      addEventListener(ev, fn) { this.listeners[ev] = fn; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 840, height: 840 }; },
      getContext() { return makeMockCtx(); },
      toDataURL() { return "data:image/png;base64,AA=="; },
      click() {}
    };
  }
  return elements[id];
}

const document = {
  createElement(tag) { return getEl("dyn_" + tag + "_" + Math.random()); },
  getElementById(id) { return getEl(id); },
  querySelectorAll() { return []; },
  documentElement: { setAttribute() {} }
};
const window = { addEventListener() {} };
let rafCount = 0;
function requestAnimationFrame() { rafCount++; }
"""
        + js_code
        + """
const report = { presetIds: PRESETS.map(p => p.id), presets: {} };

// Test every preset across simulation steps, all 4 neural views, and both themes!
for (let idx = 0; idx < PRESETS.length; idx++) {
  loadPreset(idx);
  for (let step = 0; step < 35; step++) stepSimulation();
  const pr = PRESETS[idx];
  const snap = state.shannonSnap;

  // Verify environmental Chaser/Walker boxes are strictly excluded from focal particles
  const anyEnvInParticles = state.particles.some(p => p.isEnvTarget || p.experimentId !== pr.id);
  let maxArcErr = 0.0;
  let minH = Infinity, maxH = -Infinity;
  for (const uName of pr.unitNames) {
    const h = snap.partEntropy[uName];
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
    const arcSum = snap.partCodewordArcs[uName].reduce((a, b) => a + b.prob, 0.0);
    const err = Math.abs(arcSum - snap.partFiringProb[uName]);
    if (err > maxArcErr) maxArcErr = err;
  }
  const minWireBits = Math.min(...snap.wireStats.map(w => Math.min(w.miBits, w.teBits)));

  // Exercise all 4 neural views & both themes on all 3 canvases + 4K exporter!
  for (const theme of ["cosmic", "textbook"]) {
    state.theme = theme;
    for (const nv of ["composite", "codebook", "entropy", "raster"]) {
      state.neuralView = nv;
      drawArena(actx, 840, 840);
      drawPhaseSpace(pctx, 840, 840);
      drawNeuralConnectome(nctx, 900, 840);
    }
  }
  elements["btnExport4K"].listeners["click"]();

  report.presets[pr.id] = {
    N: snap.N,
    M: snap.M,
    envTargetsCount: state.envTargets.length,
    anyEnvInParticles,
    minH,
    maxH,
    H_W: snap.jointEntropyBits,
    TC: snap.totalCorrelationBits,
    I_Phase_W: snap.spatialMiBits,
    minWireBits,
    maxArcErr,
    barcodesCount: snap.barcodes.length,
    rasterLen: snap.rasterMatrix.length,
    awakeProb: snap.partFiringProb.awake ?? null,
    awakeEntropy: snap.partEntropy.awake ?? null
  };
}

// Verify Hero Tracer switching
loadPreset(0);
selectHeroVehicle(5);
report.heroSwitchOk = (state.heroIndex === 5 && state.shannonSnap.heroIndex === 5);

// Verify AxiomViolationError rejects cross-species mashup AND leaked envTarget!
let rejectsCrossSpecies = false;
try {
  const alien = makeFocalParticle(PRESETS[1], 20, 20, 0, 99);
  validateSingleExperimentAxiom(PRESETS[0], [state.particles[0], alien]);
} catch (e) {
  rejectsCrossSpecies = e instanceof AxiomViolationError;
}
let rejectsEnvLeak = false;
try {
  validateSingleExperimentAxiom(PRESETS[0], [state.particles[0], state.envTargets[0]]);
} catch (e) {
  rejectsEnvLeak = e instanceof AxiomViolationError;
}
report.rejectsCrossSpecies = rejectsCrossSpecies;
report.rejectsEnvLeak = rejectsEnvLeak;

// Verify anticorrelated [1,0]/[0,1] vs correlated [1,1]/[0,0] ensembles are distinguished by Codebook Arcs!
const dummySpec = { id: "v4a_orbiter", focalSpecies: "Vehicle 4a: Orbiter", unitNames: ["u0", "u1"] };
const antiSwarm = [
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([1, 0]), levels: {} },
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([1, 0]), levels: {} },
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([0, 1]), levels: {} },
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([0, 1]), levels: {} }
];
const corrSwarm = [
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([1, 1]), levels: {} },
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([1, 1]), levels: {} },
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([0, 0]), levels: {} },
  { experimentId: "v4a_orbiter", speciesName: "Vehicle 4a: Orbiter", isEnvTarget: false, bits: new Uint8Array([0, 0]), levels: {} }
];
const snapAnti = computeShannonSnapshot(dummySpec, antiSwarm, null, null, 0);
const snapCorr = computeShannonSnapshot(dummySpec, corrSwarm, null, null, 0);
report.antiDisjointPalette = snapAnti.partCodewordArcs.u0[0].paletteIdx !== snapAnti.partCodewordArcs.u1[0].paletteIdx;
report.corrSharedPalette = snapCorr.partCodewordArcs.u0[0].paletteIdx === snapCorr.partCodewordArcs.u1[0].paletteIdx;
report.antiTC = snapAnti.totalCorrelationBits;
report.corrTC = snapCorr.totalCorrelationBits;

// Exact 200-step Vehicle 4a kinematics parity against Python Vehicle.step
const oneLight = [{ x: 60, y: 50, intensity: 10, falloff: 15, active: true }];
const v4Params = { wiring: "crossed", sign: 1, responseType: "peak", peakAt: 4.0, peakWidth: 1.5, gain: 6.0, baseSpeed: 2.0, maxSpeed: 12.0 };
let kx = 35.0, ky = 44.0, kh = 0.0;
for (let step = 0; step < 200; step++) {
  const kin = memorylessKinematics(kx, ky, kh, v4Params, oneLight, 100, 100, true);
  kh += kin.turnRate * 0.05;
  kx = ((kx + kin.speed * Math.cos(kh) * 0.05) % 100 + 100) % 100;
  ky = ((ky + kin.speed * Math.sin(kh) * 0.05) % 100 + 100) % 100;
}
report.kinematicsPose = { x: kx, y: ky, h: kh };

console.log(JSON.stringify(report));
""",
        encoding="utf-8",
    )

    run_js = subprocess.run([node_bin, str(harness_file)], capture_output=True, text=True)
    assert run_js.returncode == 0, run_js.stderr
    report = json.loads(run_js.stdout.strip())

    assert report["presetIds"] == [exp_id for exp_id, _, _, _ in CANONICAL_EXPERIMENTS]
    assert report["heroSwitchOk"] is True
    assert report["rejectsCrossSpecies"] is True
    assert report["rejectsEnvLeak"] is True
    assert report["antiDisjointPalette"] is True
    assert report["corrSharedPalette"] is True
    assert report["antiTC"] == pytest.approx(1.0, abs=1e-12)
    assert report["corrTC"] == pytest.approx(1.0, abs=1e-12)

    for exp_id, _, _, _ in CANONICAL_EXPERIMENTS:
        pr = report["presets"][exp_id]
        assert pr["anyEnvInParticles"] is False
        assert pr["minH"] >= -1e-12 and pr["maxH"] <= 1.0 + 1e-12
        assert pr["H_W"] >= 0.0
        assert pr["TC"] >= 0.0
        assert pr["I_Phase_W"] >= 0.0
        assert pr["minWireBits"] >= 0.0
        assert pr["maxArcErr"] < 1e-12
        assert pr["barcodesCount"] >= 4
        assert pr["rasterLen"] == pr["N"] * pr["M"]

    assert report["presets"]["v9_shy"]["envTargetsCount"] >= 2
    assert report["presets"]["v12_daydreamer"]["awakeProb"] == pytest.approx(1.0, abs=1e-12)
    assert report["presets"]["v12_daydreamer"]["awakeEntropy"] == pytest.approx(0.0, abs=1e-12)

    # Verify exact numerical kinematics parity against Python Vehicle.step
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

    js_pose = report["kinematicsPose"]
    assert js_pose["x"] == pytest.approx(veh.x, abs=1e-10)
    assert js_pose["y"] == pytest.approx(veh.y, abs=1e-10)
    assert js_pose["h"] == pytest.approx(veh.heading, abs=1e-10)


def test_monograph_readme_showcase_media_and_axiomatic_purity():
    assert MONOGRAPH_README.is_file()
    md = MONOGRAPH_README.read_text(encoding="utf-8")

    expected_media_targets: set[str] = set()

    def _assert_embedded(rel_path: str) -> None:
        expected_media_targets.add(rel_path)
        pattern = rf"!\[[^\]]+\]\({re.escape(rel_path)}\)"
        assert re.search(pattern, md), f"Expected embedded image ![(...)]({rel_path}) in README.md"

    def _assert_linked(rel_path: str) -> None:
        expected_media_targets.add(rel_path)
        assert f"]({rel_path})" in md, f"Expected markdown link to {rel_path} in README.md"

    # Series I — Pure Axiom-1 Deep Phase Portraits (media/phase_portraits/, 16 4K plates)
    for exp_id in PHASE_PORTRAIT_EXPERIMENT_IDS:
        _assert_embedded(f"media/phase_portraits/{exp_id}_phase.png")
        _assert_linked(f"media/phase_portraits/{exp_id}_phase_textbook.png")

    # Series II — Pure Axiom-1 Shannon x Strogatz Tri-Panels (media/, 18 plates & GIFs + MP4s)
    for exp_id, scenario_name, _, _ in CANONICAL_EXPERIMENTS:
        _assert_embedded(f"media/{exp_id}.gif")
        _assert_embedded(f"media/{exp_id}_tripanel.png")
        _assert_linked(f"media/{exp_id}_tripanel_textbook.png")
        assert f"media/mp4/{exp_id}_" in md, f"Missing social MP4 reference for {exp_id}"
        assert scenario_name in md, f"Missing swarm scenario link {scenario_name}"

    # Series III — Comparative Synthesis Phase-Space Atlas (media/phase_atlas/, 14 plates & GIFs + MP4s)
    for stem in PHASE_ATLAS_POSTER_STEMS:
        _assert_embedded(f"media/phase_atlas/{stem}.png")
        _assert_linked(f"media/phase_atlas/{stem}_textbook.png")
    for scene in PHASE_ATLAS_SCENES:
        _assert_embedded(f"media/phase_atlas/{scene}.gif")
        assert (
            f"media/mp4/phase_atlas/{scene}_" in md
        ), f"Missing phase_atlas MP4 reference for {scene}"

    assert len(expected_media_targets) == 48

    # Zero un-namespaced root comparative plates/GIFs (they belong strictly under media/phase_atlas/)
    for unnamespaced_legacy in (
        *(f"media/{stem}.png" for stem in PHASE_ATLAS_POSTER_STEMS),
        *(f"media/{stem}_textbook.png" for stem in PHASE_ATLAS_POSTER_STEMS),
        *(f"media/{scene}.gif" for scene in PHASE_ATLAS_SCENES),
    ):
        assert (
            unnamespaced_legacy not in md
        ), f"Un-namespaced atlas path {unnamespaced_legacy} must live under media/phase_atlas/"

    # Exact 1-to-1 media link hygiene & repository relative-link existence check
    all_md_targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", md)
    media_targets_in_md = {t for t in all_md_targets if t.startswith("media/")}
    assert media_targets_in_md == expected_media_targets

    for media_rel in sorted(expected_media_targets):
        media_file = (MONOGRAPH_README.parent / media_rel).resolve()
        assert media_file.is_file() and media_file.stat().st_size > 0, (
            f"Missing or empty showcase media asset on disk: {media_rel} -> {media_file}"
        )

    for target in set(all_md_targets) - media_targets_in_md:
        assert not target.startswith(("http://", "https://")), f"Unexpected external link {target}"
        clean_target = target.split("#", 1)[0]
        if clean_target:
            resolved = (MONOGRAPH_README.parent / clean_target).resolve()
            assert resolved.exists(), f"Broken relative link in README.md: {target} -> {resolved}"

    for landmark in (
        "Axiom 1",
        "Watanabe",
        "TC(\\mathbf{A})",
        "T_{S \\to D}",
        "awake",
        "0.00\\text{ bits}",
        "w_c = 0.7",
        "Ergotrix",
        "Kuramoto",
        "explorer.html",
        "python -m braitenberg.dynamics",
        "--series",
        "tripanels",
        "phase-portraits",
        "atlas",
        "`D`",
        "`N`",
    ):
        assert landmark in md, f"Missing theoretical or CLI landmark {landmark!r} in README.md"


@pytest.mark.parametrize("exp_id,fname,expected_type,expected_name", CANONICAL_EXPERIMENTS)
def test_six_swarm_scenarios_headless_and_focal_homogeneity(
    exp_id: str, fname: str, expected_type: str, expected_name: str
):
    from braitenberg.dynamics.shannon import (
        EXPERIMENTS,
        compute_shannon_snapshot,
        select_focal_vehicles,
        validate_single_experiment_axiom,
    )

    assert set(EXPERIMENTS.keys()) == {eid for eid, _, _, _ in CANONICAL_EXPERIMENTS}

    sc_path = REPO / "scenarios" / fname
    assert sc_path.is_file()
    scenario = load_scenario(sc_path)
    sim = Simulation(scenario)
    sim.run(15)

    focal = select_focal_vehicles(sim.vehicles)
    assert len(focal) >= 12, f"Expected swarm of >= 12 focal vehicles in {fname}"
    for v in focal:
        assert v.definition.general.type == expected_type
        assert v.definition.general.name == expected_name

    if exp_id == "v9_shy":
        # Environmental Chaser/Walker stimulus vehicles exist in world but are excluded from focal!
        assert len(sim.vehicles) > len(focal)

    spec = EXPERIMENTS[exp_id]
    validate_single_experiment_axiom(spec, vehicles=focal)
    snap = compute_shannon_snapshot(focal)
    assert snap["joint_entropy_bits"] >= 0.0
    assert snap["total_correlation_bits"] >= 0.0


def test_python_and_js_shannon_entropy_exact_parity(tmp_path: Path):
    """Python shannon.py and explorer.html JS compute identical H(W), sum H(A_k), and TC(A)."""
    node_bin = _node_binary()
    if node_bin is None:
        pytest.skip("node binary not installed")

    from braitenberg.dynamics.shannon import compute_shannon_snapshot, select_focal_vehicles

    sim = Simulation(load_scenario(REPO / "scenarios" / "dynamics-moth-swarm.yaml"))
    sim.run(90)
    focal = select_focal_vehicles(sim.vehicles)
    py_snap = compute_shannon_snapshot(focal)
    b_mat = np.asarray(py_snap["binary_matrix"], dtype=int).tolist()
    feat_names = list(py_snap["feature_names"])

    html = EXPLORER_HTML.read_text(encoding="utf-8")
    js_code = re.search(r"<script>([\s\S]*?)</script>", html).group(1)
    kernel_js = js_code[: js_code.index('const arenaCanvas = document.getElementById("arenaCanvas");')]

    parity_script = tmp_path / "shannon_py_js_parity.js"
    parity_script.write_text(
        """
const document = {
  createElement: () => ({ width: 0, height: 0, getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {}
  })})
};
"""
        + kernel_js
        + f"""
const B = {json.dumps(b_mat)};
const unitNames = {json.dumps(feat_names)};
const spec = {{ id: "v5_moth", focalSpecies: "Vehicle 5: Moth", unitNames }};
const particles = B.map(row => ({{
  experimentId: "v5_moth",
  speciesName: "Vehicle 5: Moth",
  isEnvTarget: false,
  bits: Uint8Array.from(row),
  levels: {{}}
}}));
const jsSnap = computeShannonSnapshot(spec, particles, null, null, 0);
console.log(JSON.stringify({{
  jointEntropyBits: jsSnap.jointEntropyBits,
  marginalEntropySumBits: jsSnap.marginalEntropySumBits,
  totalCorrelationBits: jsSnap.totalCorrelationBits
}}));
""",
        encoding="utf-8",
    )
    res = subprocess.run([node_bin, str(parity_script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    js_snap = json.loads(res.stdout.strip())

    assert js_snap["jointEntropyBits"] == pytest.approx(py_snap["joint_entropy_bits"], abs=1e-12)
    assert js_snap["marginalEntropySumBits"] == pytest.approx(py_snap["marginal_entropy_sum_bits"], abs=1e-12)
    assert js_snap["totalCorrelationBits"] == pytest.approx(py_snap["total_correlation_bits"], abs=1e-12)

