"""Tests for the real-time Strogatz complex-dynamics HUD overlay in braitenberg/render.py."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

import braitenberg.render as render_mod
from braitenberg.render import DYNAMICS_MODES, PANEL_WIDTH, Renderer
from braitenberg.scenario import Scenario, SimulationSettings, load_scenario
from braitenberg.world import load_world

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "v2": REPO / "scenarios" / "vehicle2-fear-and-aggression.yaml",
    "v4": REPO / "scenarios" / "vehicle4-orbiter-and-decider.yaml",
    "v5_moth": REPO / "scenarios" / "vehicle5-moth.yaml",
    "v5_forager": REPO / "scenarios" / "vehicle5-forager.yaml",
    "v7_wary": REPO / "scenarios" / "vehicle7-wary-moth.yaml",
    "v11_driver": REPO / "scenarios" / "vehicle11-cautious-driver.yaml",
    "v12_daydreamer": REPO / "scenarios" / "vehicle12-daydreamer.yaml",
}


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


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
