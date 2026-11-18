"""Tests for the real-time Strogatz complex-dynamics HUD overlay in braitenberg/render.py."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from braitenberg.scenario import Scenario, SimulationSettings, load_scenario
from braitenberg.world import load_world

if TYPE_CHECKING:
    import pygame
    import braitenberg.render as render_mod
    from braitenberg.render import DYNAMICS_MODES, NEURAL_VIEWS, PANEL_WIDTH, Renderer

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "v2": REPO / "scenarios" / "vehicle2-fear-and-aggression.yaml",
    "v4": REPO / "scenarios" / "vehicle4-orbiter-and-decider.yaml",
    "v5_moth": REPO / "scenarios" / "vehicle5-moth.yaml",
    "v5_forager": REPO / "scenarios" / "vehicle5-forager.yaml",
    "v7_wary": REPO / "scenarios" / "vehicle7-wary-moth.yaml",
    "v9_shy": REPO / "scenarios" / "vehicle9-shy.yaml",
    "v11_driver": REPO / "scenarios" / "vehicle11-cautious-driver.yaml",
    "v12_daydreamer": REPO / "scenarios" / "vehicle12-daydreamer.yaml",
    "v13_once_bitten": REPO / "scenarios" / "vehicle13-once-bitten.yaml",
}
SWARM_SCENARIOS = {
    "orbiter": REPO / "scenarios" / "dynamics-orbiter-swarm.yaml",
    "moth": REPO / "scenarios" / "dynamics-moth-swarm.yaml",
    "forager": REPO / "scenarios" / "dynamics-forager-swarm.yaml",
    "wary_moth": REPO / "scenarios" / "dynamics-wary-moth-swarm.yaml",
    "shy": REPO / "scenarios" / "dynamics-shy-swarm.yaml",
    "daydreamer": REPO / "scenarios" / "dynamics-daydreamer-swarm.yaml",
}


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    had_pygame = "pygame" in sys.modules
    import pygame as _pg
    import braitenberg.render as _rmod

    globals()["pygame"] = _pg
    globals()["render_mod"] = _rmod
    globals()["Renderer"] = _rmod.Renderer
    globals()["DYNAMICS_MODES"] = _rmod.DYNAMICS_MODES
    globals()["NEURAL_VIEWS"] = _rmod.NEURAL_VIEWS
    globals()["PANEL_WIDTH"] = _rmod.PANEL_WIDTH

    _pg.init()
    yield
    _pg.quit()
    if not had_pygame:
        for mod_name in list(sys.modules):
            if mod_name == "braitenberg.render" or mod_name == "pygame" or mod_name.startswith("pygame."):
                sys.modules.pop(mod_name, None)


def surface_rgb(surface: pygame.Surface) -> np.ndarray:
    """Return (W, H, 3) uint8 pixel array from a pygame Surface."""
    return pygame.surfarray.array3d(surface).copy()


def test_docstring_documents_d_key_and_default_mode_is_off():
    assert "D dynamics" in (render_mod.__doc__ or "")
    renderer = Renderer(load_scenario(SCENARIOS["v4"]))
    assert renderer.dynamics_mode == 0
    assert DYNAMICS_MODES[renderer.dynamics_mode] == "Off"

    # Optional dynamics_mode constructor argument wraps modulo 4.
    r2 = Renderer(load_scenario(SCENARIOS["v4"]), dynamics_mode=2)
    assert r2.dynamics_mode == 2
    r_wrap = Renderer(load_scenario(SCENARIOS["v4"]), dynamics_mode=7)
    assert r_wrap.dynamics_mode == 3


def test_d_key_cycles_through_all_four_dynamics_modes():
    renderer = Renderer(load_scenario(SCENARIOS["v2"]))
    assert renderer.dynamics_mode == 0
    for expected in (1, 2, 3, 0, 1):
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
        assert renderer.handle(ev) is True
        assert renderer.dynamics_mode == expected


@pytest.mark.parametrize(
    "scenario_key, steps",
    [
        ("v2", 60),
        ("v4", 200),
        ("v5_moth", 900),
        ("v5_forager", 1200),
        ("v7_wary", 2800),
        ("v12_daydreamer", 160),
        ("v11_driver", 600),
    ],
)
def test_all_four_modes_render_cleanly_across_canonical_scenarios(scenario_key: str, steps: int):
    scenario = load_scenario(SCENARIOS[scenario_key])
    renderer = Renderer(scenario)
    renderer.advance(steps)
    surface = pygame.Surface(renderer.window_size)

    frames: dict[int, np.ndarray] = {}
    for sel in range(len(renderer.sim.vehicles)):
        renderer.selected = sel
        renderer.flow_cache_key = None
        renderer.phase_bg_key = None
        for mode in range(4):
            renderer.dynamics_mode = mode
            renderer.draw(surface)
            if sel == 0:
                frames[mode] = surface_rgb(surface)

    aw = renderer.size[0]
    # Mode 1 (Flow Field) & Mode 2 (Phase Portrait) must visibly enrich the arena/HUD region
    # while leaving the right-hand brain diagram panel untouched.
    assert not np.array_equal(frames[0][:aw], frames[1][:aw])
    assert not np.array_equal(frames[0][:aw], frames[2][:aw])
    assert not np.array_equal(frames[1][:aw], frames[3][:aw])
    assert np.array_equal(frames[0][aw:], frames[1][aw:])
    assert np.array_equal(frames[0][aw:], frames[2][aw:])
    assert np.array_equal(frames[0][aw:], frames[3][aw:])
    assert renderer.window_size[0] == aw + PANEL_WIDTH


def test_mouse_dragging_light_source_invalidates_cache_and_deforms_field():
    scenario = load_scenario(SCENARIOS["v4"])
    renderer = Renderer(scenario, dynamics_mode=3)
    renderer.advance(80)
    surface = pygame.Surface(renderer.window_size)
    renderer.draw(surface)

    aw = renderer.size[0]
    before_arena = surface_rgb(surface)[:aw].copy()
    old_flow_key = renderer.flow_cache_key
    old_phase_key = renderer.phase_bg_key
    assert old_flow_key is not None
    assert old_phase_key is not None

    # Grab the lamp at (60, 50) and drag it to (35, 65).
    lamp_screen = renderer.to_screen(60.0, 50.0)
    new_screen = renderer.to_screen(35.0, 65.0)
    renderer.handle(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(round(lamp_screen[0]), round(lamp_screen[1]))
    ))
    assert renderer.dragging == ("light", 0)

    renderer.handle(pygame.event.Event(
        pygame.MOUSEMOTION, pos=(round(new_screen[0]), round(new_screen[1]))
    ))
    assert renderer.flow_cache_key is None
    assert renderer.phase_bg_key is None
    new_pos = renderer.sim.world.stimuli["light"].sources[0].position
    assert new_pos == pytest.approx((35.0, 65.0), abs=0.3)

    renderer.draw(surface)
    after_arena = surface_rgb(surface)[:aw]
    assert renderer.flow_cache_key is not None and renderer.flow_cache_key != old_flow_key
    assert renderer.phase_bg_key is not None and renderer.phase_bg_key != old_phase_key
    assert not np.array_equal(before_arena, after_arena)

    renderer.handle(pygame.event.Event(
        pygame.MOUSEBUTTONUP, button=1, pos=(round(new_screen[0]), round(new_screen[1]))
    ))
    assert renderer.dragging is None


def test_switching_selected_vehicle_flips_fear_vs_aggression_flow():
    scenario = load_scenario(SCENARIOS["v2"])
    renderer = Renderer(scenario, dynamics_mode=1)
    renderer.advance(20)
    surface = pygame.Surface(renderer.window_size)

    # Vehicle 0 is Fear (repulsive); Vehicle 1 is Aggression (attractive).
    renderer.draw(surface)
    fear_frame = surface_rgb(surface)[: renderer.size[0]].copy()

    v1 = renderer.sim.vehicles[1]
    v1_px = renderer.to_screen(v1.x, v1.y)
    renderer.handle(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(round(v1_px[0]), round(v1_px[1]))
    ))
    assert renderer.selected == 1
    assert renderer.flow_cache_key is None

    renderer.draw(surface)
    aggression_frame = surface_rgb(surface)[: renderer.size[0]]
    assert not np.array_equal(fear_frame, aggression_frame)


def test_phase_plane_integration_and_moth_wary_daydreamer_regimes():
    # 1. Vehicle 4a Orbiter uses core-math polar_field & detects r* ~= 19.9 limit cycle.
    r4 = Renderer(load_scenario(SCENARIOS["v4"]), dynamics_mode=3)
    r4.advance(120)
    surf4 = pygame.Surface(r4.window_size)
    r4.draw(surf4)
    assert r4.phase_bg_meta.get("used_core") is True
    rings = r4._detect_invariant_rings(r4.sim.vehicles[0], (60.0, 50.0), 1.0, [])
    assert any(kind == "orbit" and 18.0 <= r_val <= 22.0 for r_val, kind in rings)

    # 2. Vehicle 5 Moth reverses arena flow around lamps after getting burned.
    r5 = Renderer(load_scenario(SCENARIOS["v5_moth"]), dynamics_mode=1)
    surf5 = pygame.Surface(r5.window_size)
    r5.advance(50)
    assert r5.sim.vehicles[0].unit_outputs["burned"] == 0.0
    r5.draw(surf5)
    pre_burn = surface_rgb(surf5)[: r5.size[0]].copy()

    r5.advance(1150)
    assert r5.sim.vehicles[0].unit_outputs["burned"] == 1.0
    r5.draw(surf5)
    post_burn = surface_rgb(surf5)[: r5.size[0]]
    assert not np.array_equal(pre_burn, post_burn)

    # 3. Vehicle 7 Wary Moth crosses synaptic bifurcation threshold wc = 0.7.
    r7 = Renderer(load_scenario(SCENARIOS["v7_wary"]), dynamics_mode=2)
    r7.advance(2800)
    assert max(r7.sim.vehicles[0].strengths.values()) > 0.7
    r7.draw(pygame.Surface(r7.window_size))

    # 4. Vehicle 12 Daydreamer computes non-zero ring order parameter Z(t) on 8-idea ring.
    r12 = Renderer(load_scenario(SCENARIOS["v12_daydreamer"]), dynamics_mode=3)
    r12.advance(360)
    v12 = r12.sim.vehicles[0]
    _, _, z_mag, active_sum, ring_units = render_mod._ring_order_parameter(v12)
    assert len(ring_units) == 8
    max_z_hist = max(math.hypot(h[7], h[8]) for h in r12.phase_history[0])
    assert max_z_hist >= 0.12
    assert active_sum >= 1.0 and 0.10 <= z_mag <= 1.0


def test_empty_vehicle_scenario_renders_all_modes_without_error():
    world = load_world(REPO / "worlds" / "one-light.yaml")
    empty = Scenario(name="Empty", world=world, vehicles=[], simulation=SimulationSettings())
    renderer = Renderer(empty, dynamics_mode=0)
    surface = pygame.Surface(renderer.window_size)
    for mode in range(4):
        renderer.dynamics_mode = mode
        renderer.draw(surface)


def test_n_key_cycles_neural_views_and_docstring_mentions_n_key():
    assert "N neural" in (render_mod.__doc__ or "")
    assert DYNAMICS_MODES == ("Off", "Arena Flow", "Phase + Shannon Brain", "Full Tri-View")
    assert NEURAL_VIEWS == ("composite", "codebook", "entropy", "raster")

    renderer = Renderer(load_scenario(SWARM_SCENARIOS["moth"]), dynamics_mode=2)
    assert renderer.neural_view == "composite"

    for expected in ("codebook", "entropy", "raster", "composite", "codebook"):
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)
        assert renderer.handle(ev) is True
        assert renderer.neural_view == expected

    r_custom = Renderer(load_scenario(SWARM_SCENARIOS["moth"]), dynamics_mode=2, neural_view="entropy")
    assert r_custom.neural_view == "entropy"
    r_invalid = Renderer(load_scenario(SWARM_SCENARIOS["moth"]), neural_view="nonexistent")
    assert r_invalid.neural_view == "composite"


def test_axiom1_homogeneous_focal_filtering_excludes_chasers_walkers_and_bullies():
    # 1. Single Shy scenario: 1 Shy + 1 Chaser + 2 Walkers -> Shy focal count is strictly 1.
    r_shy1 = Renderer(load_scenario(SCENARIOS["v9_shy"]), dynamics_mode=3)
    r_shy1.advance(40)
    focal_idxs, focal_vehs, hero_idx = r_shy1._focal_group()
    assert focal_idxs == [0]
    assert len(focal_vehs) == 1
    assert hero_idx == 0
    assert focal_vehs[0].definition.general.name == "Shy"
    assert r_shy1._get_live_shannon_snapshot() is None

    # Selecting Walker #2 filters focal ensemble strictly to the 2 Walkers (excluding Shy & Chaser).
    r_shy1.selected = 2
    r_shy1._shannon_snap_key = None
    w_idxs, w_vehs, w_hero = r_shy1._focal_group()
    assert w_idxs == [2, 3]
    assert all(v.definition.general.name == "Walker" for v in w_vehs)
    assert w_hero == 0

    # 2. Multi-vehicle Shy Swarm: 16 Shy + environmental Chasers/Walkers -> focal is strictly 16 Shy!
    r_swarm = Renderer(load_scenario(SWARM_SCENARIOS["shy"]), dynamics_mode=3)
    r_swarm.advance(80)
    s_idxs, s_vehs, s_hero = r_swarm._focal_group()
    assert len(r_swarm.sim.vehicles) > 16
    assert len(s_vehs) == 16
    assert s_idxs == list(range(16))
    assert s_hero == 0
    assert all(
        v.definition.general.type == "9" and v.definition.general.name == "Shy"
        for v in s_vehs
    )
    snap = r_swarm._get_live_shannon_snapshot()
    assert snap is not None
    assert snap["n_vehicles"] == 16
    assert snap["raster_matrix"].shape[0] == 16

    # 3. Vehicle 13 Once Bitten: selecting focal vehicle excludes environmental bullies/targets.
    r_v13 = Renderer(load_scenario(SCENARIOS["v13_once_bitten"]), dynamics_mode=2)
    _, v13_focal, _ = r_v13._focal_group()
    hero_def = r_v13.sim.vehicles[0].definition
    assert all(
        v.definition.general.type == hero_def.general.type
        and v.definition.general.name == hero_def.general.name
        for v in v13_focal
    )


@pytest.mark.parametrize(
    "swarm_key, steps",
    [
        ("shy", 120),
        ("moth", 220),
        ("forager", 220),
        ("wary_moth", 520),
        ("daydreamer", 140),
        ("orbiter", 140),
    ],
)
def test_shannon_panel_and_neural_views_across_all_six_swarm_experiments(
    swarm_key: str, steps: int
):
    scenario = load_scenario(SWARM_SCENARIOS[swarm_key])
    renderer = Renderer(scenario, dynamics_mode=0)
    renderer.advance(steps)
    surface = pygame.Surface(renderer.window_size)
    aw = renderer.size[0]

    # Mode 0 (Off): baseline Braitenberg rendering, zero Shannon snapshot overhead.
    renderer.draw(surface)
    frame_off = surface_rgb(surface)
    assert renderer._shannon_snap is None

    # Mode 1 (Arena Flow): enriches arena, leaves right-hand panel identical to Mode 0.
    renderer.dynamics_mode = 1
    renderer.draw(surface)
    frame_flow = surface_rgb(surface)
    assert not np.array_equal(frame_off[:aw], frame_flow[:aw])
    assert np.array_equal(frame_off[aw:], frame_flow[aw:])

    # Mode 2 & Mode 3: activates live Shannon Ensemble Connectome on the right-hand panel
    # and hue-locked Phase Portrait in the arena!
    renderer.dynamics_mode = 3
    # Step >= 2 steps in Mode 3 and draw so 1-step-delayed Transfer Entropy prev_snapshot is active.
    renderer.advance(3)

    panel_by_view: dict[str, np.ndarray] = {}
    for view in NEURAL_VIEWS:
        renderer.neural_view = view
        renderer.draw(surface)
        full = surface_rgb(surface)
        panel_by_view[view] = full[aw:].copy()
        assert not np.array_equal(frame_off[aw:], panel_by_view[view])

    assert renderer._prev_shannon_snap is not None

    # Cycling N produces genuinely distinct neural connectome visualizations on the panel.
    assert not np.array_equal(panel_by_view["composite"], panel_by_view["codebook"])
    assert not np.array_equal(panel_by_view["composite"], panel_by_view["raster"])
    assert not np.array_equal(panel_by_view["codebook"], panel_by_view["raster"])

    # Returning to Mode 0 restores exact baseline rendering cleanly.
    renderer.dynamics_mode = 0
    renderer.draw(surface)
    frame_off_again = surface_rgb(surface)
    renderer_fresh = Renderer(scenario, dynamics_mode=0)
    renderer_fresh.advance(steps + 3)
    surf_fresh = pygame.Surface(renderer_fresh.window_size)
    renderer_fresh.draw(surf_fresh)
    assert np.array_equal(frame_off_again, surface_rgb(surf_fresh))


def test_click_to_switch_hero_updates_arena_phase_and_panel_while_preserving_ensemble_entropies():
    renderer = Renderer(load_scenario(SWARM_SCENARIOS["shy"]), dynamics_mode=3)
    renderer.advance(180)
    surface = pygame.Surface(renderer.window_size)
    aw = renderer.size[0]

    renderer.draw(surface)
    frame_hero0 = surface_rgb(surface)
    snap0 = renderer._get_live_shannon_snapshot()
    assert snap0 is not None
    assert snap0["hero_index"] == 0
    h_joint_0 = snap0["joint_entropy_bits"]
    tc_0 = snap0["total_correlation_bits"]
    arcs_0 = snap0["part_codeword_arcs"]

    # Pick an isolated Shy peer (k in 1..15) whose click point selects that exact Shy vehicle.
    chosen_k = None
    for cand_k in range(1, 16):
        vk = renderer.sim.vehicles[cand_k]
        px, py = renderer.to_screen(vk.x, vk.y)
        click_pt = (round(px), round(py))
        if renderer.source_at(click_pt) is None and renderer.vehicle_at(click_pt) == cand_k:
            chosen_k = cand_k
            break
    assert chosen_k is not None

    vk = renderer.sim.vehicles[chosen_k]
    px, py = renderer.to_screen(vk.x, vk.y)
    renderer.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(round(px), round(py))))
    assert renderer.selected == chosen_k

    renderer.draw(surface)
    frame_herok = surface_rgb(surface)
    snapk = renderer._get_live_shannon_snapshot()
    assert snapk is not None
    assert snapk["hero_index"] == chosen_k
    assert snapk["hero_label"] == vk.label

    # Both arena/phase-portrait and right-hand schematic/raster/barcode-cursors update immediately!
    assert not np.array_equal(frame_hero0[:aw], frame_herok[:aw])
    assert not np.array_equal(frame_hero0[aw:], frame_herok[aw:])

    # Macroscopic Claude Shannon ensemble invariants remain strictly unchanged when switching Hero!
    assert snapk["joint_entropy_bits"] == pytest.approx(h_joint_0, abs=1e-12)
    assert snapk["total_correlation_bits"] == pytest.approx(tc_0, abs=1e-12)
    assert snapk["part_codeword_arcs"] == arcs_0
