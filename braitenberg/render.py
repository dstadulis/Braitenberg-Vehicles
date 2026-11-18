"""Real-time pygame renderer. The only module that imports pygame.

It reads the simulation state and draws it; user interaction (dragging sources,
pausing, resetting) changes only the running simulation, never the files.

Keys: space pause/resume, right arrow step (when paused), R reset, T trail,
V camera views, D dynamics overlay, N neural view, Tab next stimulus, +/- simulation speed, Esc quit.
Drag sources with the mouse; click a vehicle to show it in the diagram panel (specs/vehicle-diagram.md).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import numpy as np
import pygame

from braitenberg.definition import ACTIVE_STRENGTH, ENERGY_SENSOR
from braitenberg.diagram import bundles, module_box, port_offset, wire_ends
from braitenberg.dynamics.neuro import compute_ring_order_parameter
from braitenberg.dynamics.phase_plane import (
    find_fixed_points_and_cycles,
    polar_coords,
    polar_field,
    wrap_pi,
)
from braitenberg.dynamics.shannon import (
    compute_shannon_snapshot,
    select_focal_indices,
    select_focal_vehicles,
)
from braitenberg.scenario import Scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Diagram

MAX_WINDOW_SIDE = 720
PANEL_WIDTH = 420
ROW = 20                  # height of one readout line in the panel
UNIT_RADIUS = 14
MAX_DIAGRAM_RADIUS = 150
MIN_DIAGRAM_RADIUS = 110  # the window grows taller, if needed, to leave the diagram this much
MIN_WINDOW_HEIGHT = 600   # so the panel fits even for a long, narrow arena
HEATMAP_CELL = 4          # pixels per heat map sample
TRAIL_LENGTH = 800        # recent positions kept per vehicle
SOURCE_RADIUS = 7
PANEL_PORT_GAP = 22      # pixels between two ports of a module's box, in the panel
PANEL_BOX_HEIGHT = 18
MIN_VEHICLE = 6           # pixels: the smallest half length a vehicle is drawn with
GRAB_RADIUS = 14
FPS = 60

# Heat map color stops, from no stimulus (dark blue) to the strongest (pale yellow).
HEAT_STOPS = np.array([
    [8, 12, 40], [40, 30, 110], [150, 40, 110], [230, 100, 50], [255, 230, 140],
], dtype=float)

BACKGROUND = (8, 12, 40)
VEHICLE_FILL = (160, 160, 160)   # gray, as cameras see a vehicle by default
VEHICLE_OUTLINE = (20, 20, 20)
OBSTACLE_FILL = (120, 124, 140)
SOURCE_FILL = (255, 255, 255)
# Sources of a stimulus named after a color are drawn in that color.
STIMULUS_COLORS = {"red": (235, 70, 60), "green": (70, 210, 100), "blue": (80, 140, 255),
                   "yellow": (250, 220, 70)}
TEXT = (235, 235, 235)
SELECTED_RING = (120, 220, 255)
CAMERA_VIEW = (120, 220, 255)
GRID_WIDTH = 0.6          # body units: the width of a camera's or an array's grid of cells
PANEL_BG = (22, 24, 34)
PANEL_BORDER = (60, 64, 84)
PANEL_DIM = (150, 155, 175)
BODY_OUTLINE = (180, 185, 205)
WIRE_OFF = (60, 64, 90)
WIRE_ON = (255, 230, 140)
WIRE_INACTIVE = (70, 74, 96)
LEARN_DIM = (40, 100, 70)      # Mnemotrix wires (Vehicle 7)
LEARN_ON = (90, 230, 140)
ERGO_DIM = (25, 95, 90)        # Ergotrix wires (Vehicle 11)
ERGO_ON = (60, 215, 200)
CONTROL_COLOR = (170, 130, 245)   # a threshold control and its arrows (Vehicle 12)
LEARNING_LABELS = {"mnemotrix": "M", "ergotrix": "E"}
STRENGTH_TICK = 0.7           # a connection is active above this strength

# Live Strogatz x Shannon complex-dynamics overlay (cycled with D and N keys).
DYNAMICS_MODES = ("Off", "Arena Flow", "Phase + Shannon Brain", "Full Tri-View")
NEURAL_VIEWS = ("composite", "codebook", "entropy", "raster")
CODEWORD_PALETTE = (
    (56, 189, 248),    # C0: bioluminescent cyan
    (251, 113, 133),   # C1: electric coral
    (251, 191, 36),    # C2: solar amber
    (52, 211, 153),    # C3: emerald
    (192, 132, 252),   # C4: violet
    (129, 140, 248),   # C5: soft indigo
)
CODEWORD_RESIDUAL = (148, 163, 184)
HUD_GLASS_BG = (10, 14, 36, 215)
HUD_HEADER_BG = (18, 26, 62, 232)
HUD_PLOT_BG = (6, 9, 26, 185)
HUD_BORDER_GLOW = (56, 189, 248, 72)
HUD_BORDER_CORE = (96, 212, 255, 210)
HUD_GRID_LINE = (50, 68, 120, 90)
HUD_AXIS_LINE = (115, 145, 205, 165)
FLOW_ATTRACT = (56, 189, 248)     # bioluminescent cyan (inward attraction / dpsi=0)
FLOW_ORBIT = (251, 191, 36)       # solar amber (limit cycles / critical ticks)
FLOW_REPEL = (251, 113, 133)      # electric coral (repulsion / dr=0 / sharp curvature)
FLOW_CRUISE = (90, 160, 230)      # calm cerulean (neutral cruise)
PHASE_EMERALD = (52, 211, 153)    # energy & active synaptic plasticity
PHASE_VIOLET = (192, 132, 252)    # neural ring & homeostatic threshold control
PHASE_HISTORY_LEN = 320


def codeword_color(cw_id: int) -> tuple[int, int, int]:
    """Map a Shannon joint codeword index to its distinct high-contrast RGB hue."""
    if 0 <= int(cw_id) < len(CODEWORD_PALETTE):
        return CODEWORD_PALETTE[int(cw_id)]
    return CODEWORD_RESIDUAL


def learning_colors(w) -> tuple[tuple, tuple]:
    """A learning wire's colors, inactive and fully active."""
    return (ERGO_DIM, ERGO_ON) if w.learning == "ergotrix" else (LEARN_DIM, LEARN_ON)


def heat_colors(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB through HEAT_STOPS."""
    pos = np.clip(values, 0, 1) * (len(HEAT_STOPS) - 1)
    i = np.minimum(pos.astype(int), len(HEAT_STOPS) - 2)
    frac = (pos - i)[..., None]
    return (HEAT_STOPS[i] * (1 - frac) + HEAT_STOPS[i + 1] * frac).astype(np.uint8)


class Renderer:
    def __init__(
        self, scenario: Scenario, dynamics_mode: int = 0, neural_view: str = "composite"
    ):
        self.original = scenario
        world = scenario.world
        arena = world.arena
        self.scale = MAX_WINDOW_SIDE / max(arena.width, arena.height)
        self.size = (round(arena.width * self.scale), round(arena.height * self.scale))
        # Tall enough for the panel of the vehicle with the most readout lines.
        # An array of units is one line.
        lines = max((len(p.definition.sensors) + len({u.array or u.name for u in p.definition.units})
                     + len(p.definition.actuators)
                     + sum(1 for part in p.definition.units for c in part.inputs if c.learning)
                     for p in scenario.vehicles), default=0)
        # A vehicle with cameras or arrays of units has many parts: give its diagram all the room.
        crowded = any(p.definition.arrays or any(s.camera for s in p.definition.sensors)
                      for p in scenario.vehicles)
        radius = MAX_DIAGRAM_RADIUS if crowded else MIN_DIAGRAM_RADIUS
        panel_height = 62 + 30 + 2 * radius + 24 + ROW * (lines + 2) + 18
        self.window_size = (self.size[0] + PANEL_WIDTH,
                            max(self.size[1], MIN_WINDOW_HEIGHT, panel_height))
        self.stimulus_names = list(world.stimuli)
        sensors = [s for p in scenario.vehicles for s in p.definition.sensors]
        first_sensor = next((s.type for s in sensors if s.type in world.stimuli), None)
        self.stimulus_index = (self.stimulus_names.index(first_sensor)
                               if first_sensor in self.stimulus_names else 0)
        self.show_trail = True
        self.show_views = False
        self.dynamics_mode = int(dynamics_mode) % len(DYNAMICS_MODES)
        self.neural_view = neural_view if neural_view in NEURAL_VIEWS else NEURAL_VIEWS[0]
        self.paused = False
        # About 0.05 simulated seconds per frame, whatever the time step.
        self.steps_per_frame = max(1, round(0.05 / scenario.simulation.dt))
        self.dragging: tuple[str, int] | None = None
        self.selected = 0
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 17)
        self.tiny_font = pygame.font.Font(None, 15)
        self._equilibria_cache: dict[tuple, dict] = {}
        self._ring_patch_cache: dict[tuple, pygame.Surface] = {}
        self.reset()

    # --- state ---

    def reset(self) -> None:
        self.sim = Simulation(self.original)
        self.trails = [deque(maxlen=TRAIL_LENGTH) for _ in self.sim.vehicles]
        self.trail_kinematics = [deque(maxlen=TRAIL_LENGTH) for _ in self.sim.vehicles]
        self.phase_history = [deque(maxlen=PHASE_HISTORY_LEN) for _ in self.sim.vehicles]
        self.heatmap = None
        self.heatmap_key = None
        self.flow_cache_key = None
        self.flow_surface: pygame.Surface | None = None
        self.phase_bg_key = None
        self.phase_bg_surface: pygame.Surface | None = None
        self.phase_bg_meta: dict = {}
        self._prev_shannon_snap: dict | None = None
        self._shannon_snap: dict | None = None
        self._shannon_snap_key: tuple | None = None
        self._shannon_species_key: tuple | None = None
        self._shannon_snap_time: float = -1.0
        if self.selected >= len(self.sim.vehicles):
            self.selected = 0
        self.record_trails()

    def advance(self, steps: int) -> None:
        for step_i in range(steps):
            if self.dynamics_mode >= 2 and step_i == steps - 1 and steps >= 2:
                # Capture t - dt snapshot right before the final step so live Transfer Entropy
                # T_{S -> D} = I(D_t; S_{t-1} | D_{t-1}) always measures exact 1-step dynamics.
                self._get_live_shannon_snapshot()
            self.sim.step()
            self.record_trails()

    def record_trails(self) -> None:
        arena = self.sim.world.arena
        for idx, (trail, kin, hist, v) in enumerate(
            zip(self.trails, self.trail_kinematics, self.phase_history, self.sim.vehicles)
        ):
            trail.append((v.x, v.y))
            kin.append((v.speed, v.turn_rate))
            src_info = self._primary_source_for(v)
            if src_info is not None:
                stim_name, src_idx, src = src_info
                r_val, psi_val = _polar_coords(v.x, v.y, v.heading, src.position, arena)
                src_tag = (stim_name, src_idx)
            elif any(s.camera for s in v.definition.sensors) and len(self.sim.vehicles) > 1:
                tgt_info = self._primary_target_vehicle_for(idx, v)
                if tgt_info is not None:
                    _, tgt_idx, tgt_v = tgt_info
                    r_val, psi_val = _polar_coords(v.x, v.y, v.heading, (tgt_v.x, tgt_v.y), arena)
                    src_tag = ("target", tgt_idx)
                else:
                    r_val, psi_val, src_tag = 0.0, 0.0, None
            else:
                r_val, psi_val, src_tag = 0.0, 0.0, None
            e_frac = (v.energy / v.definition.energy.max
                      if v.energy is not None and v.definition.energy and v.definition.energy.max > 0
                      else 0.0)
            max_w = max(v.strengths.values(), default=0.0)
            z_re, z_im, _, active_sum, _ = _ring_order_parameter(v)
            hist.append((float(r_val), float(psi_val), src_tag, float(v.speed), float(v.turn_rate),
                         float(e_frac), float(max_w), float(z_re), float(z_im), float(active_sum)))

    @property
    def stimulus(self) -> str | None:
        return self.stimulus_names[self.stimulus_index] if self.stimulus_names else None

    # --- coordinates (world y points up, screen y points down) ---

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, self.size[1] - y * self.scale

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        return px / self.scale, (self.size[1] - py) / self.scale

    # --- input ---

    def handle(self, event) -> bool:
        """Process one event. Returns False when the window should close."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_RIGHT and self.paused:
                self.advance(1)
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_t:
                self.show_trail = not self.show_trail
            elif event.key == pygame.K_v:
                self.show_views = not self.show_views
            elif event.key == pygame.K_d:
                self.dynamics_mode = (self.dynamics_mode + 1) % len(DYNAMICS_MODES)
            elif event.key == pygame.K_n:
                nv_idx = NEURAL_VIEWS.index(self.neural_view)
                self.neural_view = NEURAL_VIEWS[(nv_idx + 1) % len(NEURAL_VIEWS)]
            elif event.key == pygame.K_TAB and self.stimulus_names:
                self.stimulus_index = (self.stimulus_index + 1) % len(self.stimulus_names)
                self.heatmap = None
                self.flow_cache_key = None
                self.phase_bg_key = None
                self._shannon_snap_key = None
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.steps_per_frame = min(self.steps_per_frame * 2, 256)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.steps_per_frame = max(self.steps_per_frame // 2, 1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = self.source_at(event.pos)
            if self.dragging is None:
                picked = self.vehicle_at(event.pos)
                if picked is not None and picked != self.selected:
                    self.selected = picked
                    self.flow_cache_key = None
                    self.phase_bg_key = None
                    self._shannon_snap_key = None
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = None
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            name, i = self.dragging
            arena = self.sim.world.arena
            x, y = self.to_world(*event.pos)
            x, y = min(max(x, 0), arena.width), min(max(y, 0), arena.height)
            old = self.sim.world.stimuli[name].sources[i]
            self.sim.world.stimuli[name].sources[i] = replace(old, position=(x, y))
            self.heatmap = None
            self.flow_cache_key = None
            self.phase_bg_key = None
            self._shannon_snap_key = None
        return True

    def source_at(self, pos) -> tuple[str, int] | None:
        for name, stimulus in self.sim.world.stimuli.items():
            for i, s in enumerate(stimulus.sources):
                sx, sy = self.to_screen(*s.position)
                if math.hypot(pos[0] - sx, pos[1] - sy) <= GRAB_RADIUS:
                    return name, i
        return None

    def vehicle_at(self, pos) -> int | None:
        if pos[0] >= self.size[0]:
            return None
        best, best_d = None, GRAB_RADIUS
        for i, v in enumerate(self.sim.vehicles):
            vx, vy = self.to_screen(v.x, v.y)
            d = math.hypot(pos[0] - vx, pos[1] - vy)
            if d <= best_d:
                best, best_d = i, d
        return best

    # --- Axiom 1 focal species filtering & live Shannon ensemble state ---

    def _focal_group(self) -> tuple[list[int], list, int]:
        """Return `(focal_sim_indices, focal_vehicles, hero_index)` strictly matching `hero`'s species."""
        if not self.sim.vehicles:
            return [], [], 0
        hero = self.sim.vehicles[self.selected]
        focal_indices = select_focal_indices(self.sim.vehicles, focal_definition=hero.definition)
        focal_vehicles = select_focal_vehicles(self.sim.vehicles, focal_definition=hero.definition)
        hero_index = focal_indices.index(self.selected) if self.selected in focal_indices else 0
        return focal_indices, focal_vehicles, hero_index

    def _primary_target_vehicle_for(self, sim_idx: int, v) -> tuple[str, int, object] | None:
        """Nearest moving target vehicle for camera-equipped vehicles (preferring environmental agents)."""
        if len(self.sim.vehicles) <= 1:
            return None
        arena = self.sim.world.arena
        vt, vn = v.definition.general.type, v.definition.general.name
        env_targets = [
            (i, w) for i, w in enumerate(self.sim.vehicles)
            if i != sim_idx and (w.definition.general.type != vt or w.definition.general.name != vn)
        ]
        pool = env_targets if env_targets else [
            (i, w) for i, w in enumerate(self.sim.vehicles) if i != sim_idx
        ]
        if not pool:
            return None
        best_i, best_w = min(
            pool,
            key=lambda item: math.hypot(*arena.delta(v.x - item[1].x, v.y - item[1].y)),
        )
        return ("target", best_i, best_w)

    def _focal_phase_coords(
        self, focal_indices: list[int], focal_vehicles: list, hero
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Compute live 2D phase coordinates `(px, py, kind)` across `focal_vehicles`."""
        n = len(focal_vehicles)
        arena = self.sim.world.arena
        if self._all_sensed_sources(hero, active_only=False):
            rx = np.zeros(n, dtype=float)
            py = np.zeros(n, dtype=float)
            for k, peer in enumerate(focal_vehicles):
                p_src = self._primary_source_for(peer)
                if p_src is not None:
                    rv, pv = _polar_coords(peer.x, peer.y, peer.heading, p_src[2].position, arena)
                    rx[k], py[k] = float(rv), float(pv)
            return rx, py, "polar_source"

        if any(s.camera for s in hero.definition.sensors) and len(self.sim.vehicles) > 1:
            rx = np.zeros(n, dtype=float)
            py = np.zeros(n, dtype=float)
            for k, (sim_i, peer) in enumerate(zip(focal_indices, focal_vehicles)):
                t_info = self._primary_target_vehicle_for(sim_i, peer)
                if t_info is not None:
                    rv, pv = _polar_coords(peer.x, peer.y, peer.heading, (t_info[2].x, t_info[2].y), arena)
                    rx[k], py[k] = float(rv), float(pv)
            return rx, py, "polar_target"

        zx = np.zeros(n, dtype=float)
        zy = np.zeros(n, dtype=float)
        for k, peer in enumerate(focal_vehicles):
            zr, zi, *_ = _ring_order_parameter(peer)
            zx[k], zy[k] = float(zr), float(zi)
        return zx, zy, "mandala"

    def _get_live_shannon_snapshot(self) -> dict | None:
        """Compute or retrieve the current step's Shannon ensemble snapshot for `focal_vehicles`."""
        focal_indices, focal_vehicles, hero_index = self._focal_group()
        if len(focal_vehicles) < 2 or compute_shannon_snapshot is None:
            return None
        hero = focal_vehicles[hero_index]
        species_key = (hero.definition.general.type, hero.definition.general.name)
        snap_key = (species_key, round(self.sim.time, 6), hero_index, len(focal_vehicles))
        if self._shannon_snap is not None and self._shannon_snap_key == snap_key:
            return self._shannon_snap

        if self._shannon_species_key != species_key:
            prev_snap = None
            self._prev_shannon_snap = None
        elif abs(self.sim.time - self._shannon_snap_time) > 1e-9:
            prev_snap = self._shannon_snap
            self._prev_shannon_snap = prev_snap
        else:
            prev_snap = self._prev_shannon_snap

        px_arr, py_arr, _ = self._focal_phase_coords(focal_indices, focal_vehicles, hero)
        snap = compute_shannon_snapshot(
            focal_vehicles,
            prev_snapshot=prev_snap,
            phase_coords=(px_arr, py_arr),
            hero_index=hero_index,
            max_codewords=len(CODEWORD_PALETTE),
            barcode_bins=20,
        )
        self._shannon_snap = snap
        self._shannon_snap_key = snap_key
        self._shannon_species_key = species_key
        self._shannon_snap_time = self.sim.time
        return snap

    # --- drawing ---

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(PANEL_BG)          # below a short arena, the window is empty
        self.draw_field(surface)
        if self.dynamics_mode in (1, 3):
            self.draw_arena_dynamics(surface)
        if self.show_trail:
            self.draw_trails(surface)
        self.draw_obstacles(surface)
        self.draw_sources(surface)
        self.draw_vehicles(surface)
        if self.show_views:
            self.draw_views(surface)
        if self.dynamics_mode in (2, 3):
            self.draw_phase_hud(surface)
        self.draw_hud(surface)
        self.draw_panel(surface)

    def draw_field(self, surface: pygame.Surface) -> None:
        if self.stimulus is None:
            surface.fill(BACKGROUND, (0, 0, *self.size))
            return
        # The field changes when sources are moved (heatmap reset) or switch on or off.
        key = (self.stimulus, self.sim.world.active_sources(self.sim.time))
        if self.heatmap is None or key != self.heatmap_key:
            self.heatmap_key = key
            w, h = self.size
            cols, rows = max(w // HEATMAP_CELL, 1), max(h // HEATMAP_CELL, 1)
            xs = (np.arange(cols) + 0.5) * w / cols / self.scale
            ys = self.size[1] / self.scale - (np.arange(rows) + 0.5) * h / rows / self.scale
            gx, gy = np.meshgrid(xs, ys, indexing="ij")  # surfarray is indexed [x, y]
            values = self.sim.world.stimulus_at(self.stimulus, gx, gy, self.sim.time)
            stimulus = self.sim.world.stimuli[self.stimulus]
            peak = max([s.intensity for s in stimulus.sources]
                       + ([stimulus.surfaces.intensity] if stimulus.surfaces else []), default=0)
            normalized = values / peak if peak > 0 else np.zeros_like(values)
            small = pygame.surfarray.make_surface(heat_colors(normalized))
            self.heatmap = pygame.transform.smoothscale(small, self.size)
        surface.blit(self.heatmap, (0, 0))

    def draw_trails(self, surface: pygame.Surface) -> None:
        arena = self.sim.world.arena
        if self.dynamics_mode != 3:
            for trail in self.trails:
                points = list(trail)
                n = len(points)
                for k in range(1, n):
                    (x0, y0), (x1, y1) = points[k - 1], points[k]
                    if abs(x1 - x0) > arena.width / 2 or abs(y1 - y0) > arena.height / 2:
                        continue  # wrapped around an edge
                    fade = k / n
                    color = tuple(int(b + (255 - b) * fade * 0.8) for b in BACKGROUND)
                    pygame.draw.line(surface, color, self.to_screen(x0, y0), self.to_screen(x1, y1), 2)
            return

        # Mode 3: luminous comet trails (hue-locked to Shannon codeword for focal ensembles).
        focal_indices, focal_vehicles, _ = self._focal_group()
        snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
        cw_by_sim: dict[int, tuple[int, int, int]] = {}
        if snap is not None and "vehicle_codeword_ids" in snap:
            cw_ids = snap["vehicle_codeword_ids"]
            for fi, sim_i in enumerate(focal_indices):
                cw_by_sim[sim_i] = codeword_color(int(cw_ids[fi]))
        focal_set = set(focal_indices)

        prev_clip = surface.get_clip()
        surface.set_clip((0, 0, *self.size))
        glow_layer = pygame.Surface(self.size, pygame.SRCALPHA)
        for idx, (trail, kin, v) in enumerate(zip(self.trails, self.trail_kinematics, self.sim.vehicles)):
            points = list(trail)
            kins = list(kin)
            n = len(points)
            if n < 2:
                continue
            v_ref = max((a.max_speed for a in v.definition.actuators), default=5.0)
            v_ref = max(v_ref, 1e-3)
            is_sel = idx == self.selected
            is_env = bool(focal_set) and (idx not in focal_set)
            cw_col = cw_by_sim.get(idx)
            for k in range(1, n):
                (x0, y0), (x1, y1) = points[k - 1], points[k]
                if abs(x1 - x0) > arena.width / 2 or abs(y1 - y0) > arena.height / 2:
                    continue
                if is_env:
                    seg_col = (135, 145, 168)
                elif cw_col is not None:
                    spd, _ = kins[k] if k < len(kins) else (v.speed, v.turn_rate)
                    s_norm = min(max(spd / v_ref, 0.0), 1.0)
                    seg_col = lerp_color(cw_col, (255, 255, 255), 0.18 * s_norm)
                else:
                    spd, omg = kins[k] if k < len(kins) else (v.speed, v.turn_rate)
                    s_norm = min(max(spd / v_ref, 0.0), 1.0)
                    c_norm = min(abs(omg) / 1.8, 1.0)
                    base_col = lerp_color(FLOW_ATTRACT, FLOW_ORBIT, s_norm)
                    seg_col = lerp_color(base_col, FLOW_REPEL, 0.72 * c_norm)
                fade = (k / n) ** 1.35
                p0, p1 = self.to_screen(x0, y0), self.to_screen(x1, y1)
                if is_sel and k > n - 90:
                    halo_a = int(95 * fade)
                    pygame.draw.line(glow_layer, (*SELECTED_RING, halo_a), p0, p1, 6)
                core_col = lerp_color(BACKGROUND, seg_col, (0.16 if is_env else 0.24) + 0.76 * fade)
                pygame.draw.line(surface, core_col, p0, p1, 3 if (is_sel and k > n - 35) else 2)
        surface.blit(glow_layer, (0, 0))
        surface.set_clip(prev_clip)

    def draw_obstacles(self, surface: pygame.Surface) -> None:
        """Obstacles, to scale: gray boxes seen from above."""
        for o in self.sim.world.obstacles:
            x, y, hw, hl, turn = o.box()
            c, s = math.cos(turn), math.sin(turn)
            corners = [self.to_screen(x + a * c + b * s, y + a * s - b * c)
                       for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            pygame.draw.polygon(surface, OBSTACLE_FILL, corners)
            pygame.draw.polygon(surface, VEHICLE_OUTLINE, corners, 1)

    def draw_sources(self, surface: pygame.Surface) -> None:
        # Sources of stimuli named after a color go last, so a red lamp that is also hot shows red.
        stimuli = sorted(self.sim.world.stimuli.items(), key=lambda item: item[0] in STIMULUS_COLORS)
        for name, stimulus in stimuli:
            color = STIMULUS_COLORS.get(name, SOURCE_FILL)
            shown = name == self.stimulus
            for s in stimulus.sources:
                center = self.to_screen(*s.position)
                # A thing cameras see is drawn to scale; a point source with a fixed size.
                radius = s.radius * self.scale if s.radius is not None else SOURCE_RADIUS
                if s.active(self.sim.time):     # on: filled; off: a hollow ring only
                    pygame.draw.circle(surface, color, center, radius)
                pygame.draw.circle(surface, color, center, radius + 3, 2 if shown else 1)

    def draw_vehicles(self, surface: pygame.Surface) -> None:
        cw_by_sim: dict[int, tuple[int, int, int]] = {}
        if self.dynamics_mode >= 2 and self.sim.vehicles:
            focal_indices, focal_vehicles, _ = self._focal_group()
            snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
            if snap is not None and "vehicle_codeword_ids" in snap:
                cw_ids = snap["vehicle_codeword_ids"]
                for fi, sim_i in enumerate(focal_indices):
                    cw_by_sim[sim_i] = codeword_color(int(cw_ids[fi]))

        for idx, v in enumerate(self.sim.vehicles):
            if idx in cw_by_sim:
                reach_p = max(math.hypot(v.half_width, v.half_length) * self.scale, MIN_VEHICLE)
                sx_v, sy_v = self.to_screen(v.x, v.y)
                pygame.draw.circle(
                    surface, cw_by_sim[idx], (round(sx_v), round(sy_v)), round(reach_p + 3), 2
                )
            # The body's rectangle to scale, enlarged so that a small vehicle is still visible.
            grow = max(1.0, MIN_VEHICLE / (max(v.half_width, v.half_length) * self.scale))
            corner = lambda bx, by: self.to_screen(*v.body_to_world(bx * grow, by * grow))
            points = [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)]
            # A vehicle that cameras see as a color, on some face, is drawn in that color.
            fill = next((STIMULUS_COLORS[name] for name, faces in v.definition.appearance.items()
                         if name in STIMULUS_COLORS and max(faces.values()) > 0), VEHICLE_FILL)
            pygame.draw.polygon(surface, fill, points)
            pygame.draw.polygon(surface, VEHICLE_OUTLINE, points, 1)
            # The front: a line from the center to the middle of the front side.
            pygame.draw.line(surface, VEHICLE_OUTLINE, corner(0, 0), corner(0, 1), 2)
        if self.sim.vehicles:
            v = self.sim.vehicles[self.selected]
            reach = max(math.hypot(v.half_width, v.half_length) * self.scale, MIN_VEHICLE)
            pygame.draw.circle(surface, SELECTED_RING, self.to_screen(v.x, v.y), reach + 5, 2)

    def draw_views(self, surface: pygame.Surface) -> None:
        """The edges of each camera's field of view, for the selected vehicle."""
        if not self.sim.vehicles:
            return
        v = self.sim.vehicles[self.selected]
        arena = self.sim.world.arena
        for sensor, x, y in v.cameras():
            reach = min(sensor.camera.range, max(arena.width, arena.height))
            for side in (-1, 1):
                a = v.heading + math.radians(sensor.camera.direction + side * sensor.field_of_view / 2)
                end = (x + reach * math.cos(a), y + reach * math.sin(a))
                pygame.draw.line(surface, CAMERA_VIEW, self.to_screen(x, y), self.to_screen(*end), 1)

    def draw_hud(self, surface: pygame.Surface) -> None:
        lines = [f"t = {self.sim.time:.1f}   x{self.steps_per_frame}"
                 + ("   PAUSED" if self.paused else ""),
                 f"stimulus: {self.stimulus or '-'}"]
        scheduled = [(name, any(s.active(self.sim.time) for s in st.sources))
                     for name, st in self.sim.world.stimuli.items()
                     if any(s.schedule for s in st.sources)]
        if scheduled:
            lines.append("   ".join(f"{name} {'on' if on else 'off'}" for name, on in scheduled))
        keys_line = (
            "space pause  Right step  R reset  T trail  V views  D dynamics  N neural  +/- speed"
            if self.dynamics_mode > 0
            else "space pause  Right step  R reset  T trail  V views  D dynamics  +/- speed"
        )
        lines += [keys_line, "Tab stimulus  drag source  click vehicle"]
        # Over a tall arena, in its top corner; below a short one, so it hides nothing.
        y = 6 if self.size[1] >= MIN_WINDOW_HEIGHT else self.size[1] + 10
        for line in lines:
            text = self.font.render(line, True, TEXT)
            surface.blit(text, (8, y))
            y += text.get_height() + 2

        if self.dynamics_mode > 0:
            _, focal_vehicles, _ = self._focal_group()
            label = f"Dynamics: {DYNAMICS_MODES[self.dynamics_mode]} [D]"
            if self.dynamics_mode >= 2:
                label += f"  ·  View: {self.neural_view} [N]"
            if focal_vehicles:
                label += f"  ·  N={len(focal_vehicles)}"
            txt = self.small_font.render(label, True, TEXT)
            bw, bh = txt.get_width() + 28, txt.get_height() + 10
            badge = pygame.Surface((bw + 4, bh + 4), pygame.SRCALPHA)
            pygame.draw.rect(badge, HUD_BORDER_GLOW, (0, 0, bw + 4, bh + 4), border_radius=12)
            pygame.draw.rect(badge, HUD_GLASS_BG, (2, 2, bw, bh), border_radius=10)
            pygame.draw.rect(badge, HUD_BORDER_CORE, (2, 2, bw, bh), 1, border_radius=10)
            dot_col = (FLOW_ATTRACT if self.dynamics_mode == 1
                       else PHASE_VIOLET if self.dynamics_mode == 2 else FLOW_ORBIT)
            pygame.draw.circle(badge, (*dot_col, 90), (13, 2 + bh // 2), 6)
            pygame.draw.circle(badge, dot_col, (13, 2 + bh // 2), 3)
            badge.blit(txt, (22, 2 + (bh - txt.get_height()) // 2))
            bx = max(8, self.size[0] - bw - 12)
            by = 8 if self.size[1] >= MIN_WINDOW_HEIGHT else self.size[1] + 8
            surface.blit(badge, (bx, by))

    # --- live Strogatz complex dynamics overlays (modes 1, 2, 3) ---

    def _sensed_stimuli(self, v) -> list[str]:
        names = []
        for s in v.definition.sensors:
            if not s.camera and s.type != ENERGY_SENSOR and s.type in self.sim.world.stimuli:
                if s.type not in names:
                    names.append(s.type)
        if v.definition.energy and v.definition.energy.charge_from in self.sim.world.stimuli:
            cf = v.definition.energy.charge_from
            if cf not in names:
                names.append(cf)
        return names

    def _all_sensed_sources(self, v, active_only: bool = False) -> list[tuple[str, int, object]]:
        out = []
        for stim_name in self._sensed_stimuli(v):
            stim = self.sim.world.stimuli[stim_name]
            for idx, src in enumerate(stim.sources):
                if not active_only or src.active(self.sim.time):
                    out.append((stim_name, idx, src))
        return out

    def _primary_source_for(self, v) -> tuple[str, int, object] | None:
        candidates = self._all_sensed_sources(v, active_only=True)
        if not candidates:
            candidates = self._all_sensed_sources(v, active_only=False)
        if not candidates:
            return None
        # Prefer active sources of the currently displayed stimulus when sensed by v.
        if self.stimulus:
            shown = [item for item in candidates if item[0] == self.stimulus]
            if shown:
                candidates = shown
        arena = self.sim.world.arena
        return min(
            candidates,
            key=lambda item: math.hypot(*arena.delta(v.x - item[2].position[0],
                                                     v.y - item[2].position[1])),
        )

    def _vehicle_state_key(self, v) -> tuple:
        slow_units = tuple(
            round(v.unit_outputs[u.name], 1)
            for u in v.definition.units
            if _is_slow_or_latched_unit(u)
        )
        learned_active = tuple(
            v.is_active(v.unit_names[k], j) for k, j in v.learning_inputs
        )
        energy_bin = round(v.energy / 10.0) if v.energy is not None else None
        src_info = self._primary_source_for(v)
        spin_sign = 1
        if src_info is not None:
            _, _, src = src_info
            _, psi_now = _polar_coords(v.x, v.y, v.heading, src.position, self.sim.world.arena)
            spin_sign = 1 if float(psi_now) >= 0.0 else -1
        return (slow_units, learned_active, energy_bin, spin_sign)

    def _sources_cache_key(self) -> tuple:
        return tuple(
            (name, tuple((round(s.position[0], 3), round(s.position[1], 3),
                          s.intensity, s.falloff, s.active(self.sim.time))
                         for s in st.sources),
             (st.surfaces.intensity, st.surfaces.falloff) if st.surfaces else None)
            for name, st in self.sim.world.stimuli.items()
        )

    def draw_arena_dynamics(self, surface: pygame.Surface) -> None:
        """Mode 1 & 3: Arena Flow Field, invariant limit-cycle/threshold circles & saddles."""
        if not self.sim.vehicles:
            return
        v = self.sim.vehicles[self.selected]
        sensed_sources = self._all_sensed_sources(v, active_only=True)
        has_surfaces = any(
            self.sim.world.stimuli[name].surfaces is not None for name in self._sensed_stimuli(v)
        )
        prev_clip = surface.get_clip()
        surface.set_clip((0, 0, *self.size))

        if not sensed_sources and not has_surfaces:
            # Sensorless or camera-target vehicle (e.g. Vehicle 12 Daydreamer or Vehicle 9 Shy):
            self._draw_autonomous_curvature_overlay(surface, v)
            surface.set_clip(prev_clip)
            return

        heading_q = round(v.heading / (math.pi / 18))
        key = (self.selected, self._sources_cache_key(), self._vehicle_state_key(v),
               heading_q if len(v.definition.actuators) < 2 else 0)
        if self.flow_surface is None or key != self.flow_cache_key:
            self.flow_cache_key = key
            self.flow_surface = self._build_flow_surface(v, sensed_sources)
        surface.blit(self.flow_surface, (0, 0))
        surface.set_clip(prev_clip)

    def _draw_autonomous_curvature_overlay(self, surface: pygame.Surface, v) -> None:
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        vx, vy = self.to_screen(v.x, v.y)
        cams = v.cameras()
        if cams:
            # Camera-equipped vehicle (e.g. Vehicle 9 Shy): draw live detection wedge & symmetry axis.
            arena = self.sim.world.arena
            for sensor, cx_w, cy_w in cams:
                reach_px = min(sensor.camera.range, max(arena.width, arena.height)) * self.scale
                csx, csy = self.to_screen(cx_w, cy_w)
                a_mid = v.heading + math.radians(sensor.camera.direction)
                half_fov = math.radians(sensor.field_of_view * 0.5)
                pts = [(csx, csy)]
                for k in range(19):
                    ak = a_mid - half_fov + (2.0 * half_fov) * (k / 18)
                    pts.append((csx + reach_px * math.cos(ak), csy - reach_px * math.sin(ak)))
                pygame.draw.polygon(overlay, (*FLOW_ATTRACT, 26), pts)
                pygame.draw.aalines(overlay, (*FLOW_ATTRACT, 130), True, pts)
                # Mirror-symmetry trigger axis (psi = 0)
                sym_end = (csx + reach_px * math.cos(a_mid), csy - reach_px * math.sin(a_mid))
                draw_dashed_line(overlay, (*FLOW_ORBIT, 175), (csx, csy), sym_end, 1, dash=5)
            tgt = self._primary_target_vehicle_for(self.selected, v)
            if tgt is not None:
                tsx, tsy = self.to_screen(tgt[2].x, tgt[2].y)
                _draw_glow_dashed_circle(overlay, (tsx, tsy), 18.0, FLOW_REPEL, dash_deg=15)
        else:
            # Subtle polar reticle centered in arena for autonomous ring vehicles.
            cx, cy = self.size[0] / 2, self.size[1] / 2
            for frac in (0.22, 0.42):
                r_px = min(self.size) * frac
                pygame.draw.circle(overlay, (56, 90, 150, 32), (round(cx), round(cy)), round(r_px), 1)

        # Live osculating circle of curvature R = v / |omega| when turning, or heading ray when straight.
        if v.speed > 0.05:
            if abs(v.turn_rate) > 0.04:
                r_curv = v.speed / abs(v.turn_rate)
                side = 1.0 if v.turn_rate > 0 else -1.0
                ox = v.x - side * r_curv * math.sin(v.heading)
                oy = v.y + side * r_curv * math.cos(v.heading)
                center_px = self.to_screen(ox, oy)
                rad_px = r_curv * self.scale
                if 6 <= rad_px <= max(self.size) * 1.5:
                    _draw_glow_dashed_circle(overlay, center_px, rad_px, PHASE_VIOLET,
                                             dash_deg=9, spin_dir=1 if side > 0 else -1)
                    pygame.draw.circle(overlay, (*PHASE_VIOLET, 160),
                                       (round(center_px[0]), round(center_px[1])), 3)
            ux, uy = math.cos(v.heading), -math.sin(v.heading)
            _draw_glow_chevron(overlay, vx + ux * 18, vy + uy * 18, ux, uy, 22, FLOW_ORBIT, 220)
        surface.blit(overlay, (0, 0))

    def _build_flow_surface(self, v, sensed_sources: list[tuple[str, int, object]]) -> pygame.Surface:
        aw, ah = self.size
        overlay = pygame.Surface((aw, ah), pygame.SRCALPHA)
        arena = self.sim.world.arena
        cols = max(6, min(17, round(aw / 42)))
        rows = max(3, min(17, round(ah / 42)))
        xs = (np.arange(cols) + 0.5) * arena.width / cols
        ys = (np.arange(rows) + 0.5) * arena.height / rows
        gx, gy = np.meshgrid(xs, ys)  # shape (rows, cols)

        # Aggregate active stimulus spatial gradient pointing toward active sensed sources.
        grad_x = np.zeros_like(gx)
        grad_y = np.zeros_like(gy)
        nearest_r = np.full_like(gx, max(arena.width, arena.height))
        for _, _, src in sensed_sources:
            dx, dy = arena.delta(gx - src.position[0], gy - src.position[1])
            r2 = dx * dx + dy * dy
            nearest_r = np.minimum(nearest_r, np.sqrt(r2))
            f2 = src.falloff * src.falloff
            denom = (1.0 + r2 / f2) ** 2
            weight = 2.0 * src.intensity / (f2 * np.maximum(denom, 1e-9))
            grad_x -= weight * dx
            grad_y -= weight * dy

        grad_mag = np.hypot(grad_x, grad_y)
        alpha = np.arctan2(grad_y, grad_x)  # uphill bearing toward local source blend

        # Probe K=24 half-step bearings psi_k in (-pi, pi) so 0 and +/-pi never hit grid ties.
        n_ang = 24
        psi_k = np.linspace(-math.pi + math.pi / n_ang, math.pi - math.pi / n_ang, n_ang)
        theta_grid = alpha[..., None] - psi_k[None, None, :]
        qx = np.broadcast_to(gx[..., None], theta_grid.shape).ravel()
        qy = np.broadcast_to(gy[..., None], theta_grid.shape).ravel()
        qt = theta_grid.ravel()
        v_flat, omg_flat = _eval_kinematics_batch(v, self.sim.world, self.sim.time, qx, qy, qt)
        v_k = v_flat.reshape(rows, cols, n_ang)
        omg_k = omg_flat.reshape(rows, cols, n_ang)

        r_eff = np.maximum(nearest_r[..., None], 1.5)
        dpsi_k = (v_k / r_eff) * np.sin(psi_k)[None, None, :] - omg_k
        steer_torque = (np.mean(omg_k[..., n_ang // 2:], axis=-1)
                        - np.mean(omg_k[..., :n_ang // 2], axis=-1))

        # Determine preferred spin sign from the selected vehicle's live bearing.
        src_info = self._primary_source_for(v)
        pref_sign = 1.0
        if src_info is not None:
            _, psi_live = _polar_coords(v.x, v.y, v.heading, src_info[2].position, arena)
            pref_sign = 1.0 if float(psi_live) >= 0.0 else -1.0

        steered = (np.ptp(omg_k, axis=-1) > 1e-4) & (grad_mag > 1e-6)
        psi_star, v_star = _select_stable_bearing(psi_k, dpsi_k, v_k, pref_sign, steer_torque)

        # Fallback where steering torque vanishes (e.g., Vehicle 1 or outside step-sensor range).
        v_cur, _ = _eval_kinematics_batch(
            v, self.sim.world, self.sim.time, gx.ravel(), gy.ravel(),
            np.full(gx.size, v.heading)
        )
        v_cur = v_cur.reshape(rows, cols)

        theta_flow = np.where(steered, alpha - psi_star, v.heading)
        speed_flow = np.where(steered, v_star, v_cur)
        taxis = np.where(steered, np.cos(psi_star), 0.0)

        v_max = max(float(speed_flow.max(initial=0.0)),
                    max((a.max_speed for a in v.definition.actuators), default=1.0), 1e-3)
        cell_px = min(aw / cols, ah / rows)

        for r_idx in range(rows):
            for c_idx in range(cols):
                spd = float(speed_flow[r_idx, c_idx])
                if spd <= 1e-3:
                    # Stagnation / rest zone: subtle equilibrium dot.
                    sx_p, sy_p = self.to_screen(float(gx[r_idx, c_idx]), float(gy[r_idx, c_idx]))
                    pygame.draw.circle(overlay, (*FLOW_ATTRACT, 95), (round(sx_p), round(sy_p)), 2)
                    continue
                s_frac = min(spd / v_max, 1.0)
                tx = float(taxis[r_idx, c_idx])
                if not steered[r_idx, c_idx]:
                    col = FLOW_CRUISE
                    alpha_a = int(55 + 75 * s_frac)
                else:
                    if tx >= 0.0:
                        col = lerp_color(FLOW_ORBIT, FLOW_ATTRACT, min(tx / 0.75, 1.0))
                    else:
                        col = lerp_color(FLOW_ORBIT, FLOW_REPEL, min(-tx / 0.75, 1.0))
                    alpha_a = int(85 + 145 * s_frac)
                th = float(theta_flow[r_idx, c_idx])
                ux_s, uy_s = math.cos(th), -math.sin(th)  # screen y points down
                cx_s, cy_s = self.to_screen(float(gx[r_idx, c_idx]), float(gy[r_idx, c_idx]))
                length = cell_px * (0.30 + 0.34 * s_frac)
                _draw_glow_chevron(overlay, cx_s, cy_s, ux_s, uy_s, length, col, alpha_a)

        # Draw glowing invariant circles (limit cycles, rest rings, threshold boundaries) around active sources.
        unique_positions: list[tuple[float, float]] = []
        for _, _, src in sensed_sources:
            if all(math.hypot(src.position[0] - px, src.position[1] - py) > 1.0
                   for px, py in unique_positions):
                unique_positions.append(src.position)

        for sx_w, sy_w in unique_positions:
            others = [(ox_w, oy_w) for ox_w, oy_w in unique_positions
                      if math.hypot(ox_w - sx_w, oy_w - sy_w) > 1.0]
            rings = self._detect_invariant_rings(v, (sx_w, sy_w), pref_sign, others)
            center_px = self.to_screen(sx_w, sy_w)
            for r_world, kind in rings:
                rad_px = r_world * self.scale
                if rad_px < 8 or rad_px > max(aw, ah) * 1.1:
                    continue
                ring_col = (FLOW_ORBIT if kind == "orbit"
                            else FLOW_ATTRACT if kind == "rest" else FLOW_REPEL)
                spin = (1 if pref_sign >= 0 else -1) if kind == "orbit" else 0
                _draw_glow_dashed_circle(overlay, center_px, rad_px, ring_col,
                                         dash_deg=8 if kind == "orbit" else 6, spin_dir=spin)
                tag_txt = (f"r*={r_world:.1f}" if kind in ("orbit", "rest")
                           else f"r={r_world:.1f}")
                lbl = self.tiny_font.render(tag_txt, True, ring_col)
                lx = round(center_px[0] - lbl.get_width() / 2)
                ly = round(center_px[1] - rad_px - lbl.get_height() - 3)
                if 4 <= lx <= aw - lbl.get_width() - 4 and 4 <= ly <= ah - lbl.get_height() - 4:
                    pill = pygame.Surface((lbl.get_width() + 8, lbl.get_height() + 4), pygame.SRCALPHA)
                    pygame.draw.rect(pill, (10, 14, 36, 195), pill.get_rect(), border_radius=5)
                    pill.blit(lbl, (4, 2))
                    overlay.blit(pill, (lx - 4, ly - 2))

        # Multi-source saddle markers: pairwise midpoints where stimulus gradient nearly vanishes.
        if len(unique_positions) >= 2:
            for i in range(len(unique_positions)):
                for j in range(i + 1, len(unique_positions)):
                    p1, p2 = unique_positions[i], unique_positions[j]
                    dx, dy = arena.delta(p2[0] - p1[0], p2[1] - p1[1])
                    mx, my = (p1[0] + 0.5 * dx) % arena.width, (p1[1] + 0.5 * dy) % arena.height
                    msx, msy = self.to_screen(mx, my)
                    _draw_saddle_marker(overlay, msx, msy)

        return overlay

    def _cached_equilibria(self, v, r_hi: float) -> dict:
        stim_info = self._primary_source_for(v)
        src_sig = (
            (stim_info[0], stim_info[2].intensity, stim_info[2].falloff)
            if stim_info is not None else None
        )
        eq_key = (id(v.definition), src_sig, round(r_hi, 1))
        if eq_key not in self._equilibria_cache:
            self._equilibria_cache[eq_key] = find_fixed_points_and_cycles(
                v.definition, self.sim.world, r_bounds=(1.8, r_hi), dt=self.sim.dt
            )
        return self._equilibria_cache[eq_key]

    def _detect_invariant_rings(
        self, v, source_pos: tuple[float, float], pref_sign: float,
        other_sources: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, str]]:
        """Find invariant orbit, rest, or bifurcation threshold radii around `source_pos`."""
        arena = self.sim.world.arena
        other_sources = other_sources or []
        r_hi = min(46.0, 0.46 * min(arena.width, arena.height))
        if r_hi <= 2.5:
            return []
        rings: list[tuple[float, str]] = []

        # Use core-math's find_fixed_points_and_cycles directly on single-source memoryless vehicles.
        if not other_sources and not v.definition.units:
            info = self._cached_equilibria(v, r_hi)
            lc = info.get("limit_cycle")
            if lc and lc.get("r_mean"):
                rings.append((float(lc["r_mean"]), "orbit"))
            for fp in info.get("fixed_points", ()):
                r_fp = float(fp.get("r", 0.0))
                psi_fp = abs(float(fp.get("psi", 0.0)))
                if fp.get("stability") == "stable" and 2.0 <= r_fp <= r_hi:
                    kind = "orbit" if 0.6 < psi_fp < 2.5 else "rest"
                    if all(abs(r_fp - existing_r) > 2.0 for existing_r, _ in rings):
                        rings.append((r_fp, kind))
            if rings:
                return rings[:2]

        # Choose a probe ray direction pointing away from neighbor sources so profiles don't collide.
        sx, sy = source_pos
        best_ang = math.pi / 2
        if other_sources:
            cand_angs = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
            best_clearance = -1.0
            for c_ang in cand_angs:
                tx = (sx + 22.0 * math.cos(c_ang)) % arena.width
                ty = (sy + 22.0 * math.sin(c_ang)) % arena.height
                clearance = min(
                    math.hypot(*arena.delta(tx - ox, ty - oy)) for ox, oy in other_sources
                )
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_ang = float(c_ang)
        else:
            best_ang = math.atan2(arena.height / 2 - sy, arena.width / 2 - sx)
            if abs(arena.height / 2 - sy) < 1e-3 and abs(arena.width / 2 - sx) < 1e-3:
                best_ang = 0.0

        nr = 84
        r_vals = np.linspace(2.0, r_hi, nr)
        rx = (sx + r_vals * math.cos(best_ang)) % arena.width
        ry = (sy + r_vals * math.sin(best_ang)) % arena.height
        alpha_ray = np.full(nr, float(_wrap_pi(best_ang + math.pi)))

        n_ang = 24
        psi_k = np.linspace(-math.pi + math.pi / n_ang, math.pi - math.pi / n_ang, n_ang)
        theta_g = alpha_ray[:, None] - psi_k[None, :]
        v_f, omg_f = _eval_kinematics_batch(
            v, self.sim.world, self.sim.time,
            np.broadcast_to(rx[:, None], theta_g.shape).ravel(),
            np.broadcast_to(ry[:, None], theta_g.shape).ravel(),
            theta_g.ravel(),
        )
        v_k = v_f.reshape(nr, n_ang)
        omg_k = omg_f.reshape(nr, n_ang)
        dpsi_k = (v_k / r_vals[:, None]) * np.sin(psi_k)[None, :] - omg_k
        steer_torque = (np.mean(omg_k[:, n_ang // 2:], axis=-1)
                        - np.mean(omg_k[:, :n_ang // 2], axis=-1))
        steered = np.ptp(omg_k, axis=-1) > 1e-4
        psi_star, v_star = _select_stable_bearing(psi_k, dpsi_k, v_k, pref_sign, steer_torque)
        radial_in = np.where(steered, np.cos(psi_star), 0.0)
        v_ref = max((a.max_speed for a in v.definition.actuators), default=1.0)

        # 1. Genuine stable limit cycle: active repulsion inside AND active attraction outside!
        has_inner_repel = float(np.min(steer_torque[: nr // 2])) < -0.02
        has_outer_attract = float(np.max(steer_torque)) > 0.02
        if has_inner_repel and has_outer_attract:
            for idx in range(nr - 1):
                c_in, c_out = float(radial_in[idx]), float(radial_in[idx + 1])
                if steered[idx] and steered[idx + 1] and c_in < 0.0 <= c_out:
                    if v_star[idx] > 0.08 * v_ref:
                        frac = -c_in / max(c_out - c_in, 1e-9)
                        r_star = float(r_vals[idx] + frac * (r_vals[idx + 1] - r_vals[idx]))
                        rings.append((r_star, "orbit"))
                        break

        # 2. Stable rest circle (e.g. Vehicle 3a Love): speed drops to ~0 with inward torque outside.
        for idx in range(nr - 1):
            if v_star[idx] <= 0.03 * v_ref < v_star[idx + 1] and steer_torque[idx + 1] > 0.01:
                r_star = float(0.5 * (r_vals[idx] + r_vals[idx + 1]))
                if all(abs(r_star - er) > 2.5 for er, _ in rings):
                    rings.append((r_star, "rest"))
                break

        # 3. Repulsive barrier / threshold shell (e.g. 4b Decider, burned Moth, learned Wary Moth).
        if not rings and float(np.min(steer_torque)) < -0.02:
            # Outermost radius where repulsive steering torque turns off (or switches to attraction).
            repel_idxs = np.flatnonzero(steer_torque < -0.02)
            last_i = int(repel_idxs[-1])
            if last_i < nr - 2 and steer_torque[last_i + 1] >= -0.01:
                r_star = float(0.5 * (r_vals[last_i] + r_vals[last_i + 1]))
                rings.append((r_star, "barrier"))

        return rings[:2]

    def draw_phase_hud(self, surface: pygame.Surface) -> None:
        """Mode 2 & 3: Translucent dark-glass live phase-space inset HUD."""
        aw, ah = self.size
        card_w = min(308, max(230, aw - 24))
        has_extra = False
        v = self.sim.vehicles[self.selected] if self.sim.vehicles else None
        if v is not None:
            has_extra = (v.energy is not None) or bool(v.learning_inputs)
        card_h = 236 if has_extra else 208
        if ah >= 340:
            card_h = min(card_h, ah - 102)
            cx0, cy0 = 12, ah - card_h - 12
        else:
            # Short arena (e.g., narrow road): sit cleanly below the HUD text lines without overlap.
            cy0 = ah + 108
            card_h = min(card_h, max(160, self.window_size[1] - cy0 - 12))
            cx0 = 12

        card = pygame.Surface((card_w + 6, card_h + 6), pygame.SRCALPHA)
        pygame.draw.rect(card, HUD_BORDER_GLOW, (0, 0, card_w + 6, card_h + 6), border_radius=12)
        pygame.draw.rect(card, HUD_GLASS_BG, (3, 3, card_w, card_h), border_radius=10)
        pygame.draw.rect(card, HUD_HEADER_BG, (3, 3, card_w, 24),
                         border_top_left_radius=10, border_top_right_radius=10)
        pygame.draw.line(card, (70, 120, 190, 120), (4, 27), (card_w + 2, 27), 1)

        if v is None:
            card.blit(self.small_font.render("PHASE PORTRAIT", True, FLOW_ATTRACT), (12, 8))
            card.blit(self.small_font.render("No vehicle in scenario", True, PANEL_DIM), (12, 42))
            pygame.draw.rect(card, HUD_BORDER_CORE, (3, 3, card_w, card_h), 1, border_radius=10)
            surface.blit(card, (cx0 - 3, cy0 - 3))
            return

        all_sources = self._all_sensed_sources(v, active_only=False)
        if all_sources:
            self._draw_polar_phase_card(card, 3, 3, card_w, card_h, v)
        elif any(s.camera for s in v.definition.sensors) and len(self.sim.vehicles) > 1:
            self._draw_target_polar_phase_card(card, 3, 3, card_w, card_h, v)
        else:
            self._draw_neural_mandala_card(card, 3, 3, card_w, card_h, v)

        pygame.draw.rect(card, HUD_BORDER_CORE, (3, 3, card_w, card_h), 1, border_radius=10)
        surface.blit(card, (cx0 - 3, cy0 - 3))

    def _draw_polar_phase_card(
        self, card: pygame.Surface, ox: int, oy: int, cw: int, ch: int, v
    ) -> None:
        src_info = self._primary_source_for(v)
        assert src_info is not None
        stim_name, src_idx, src = src_info
        arena = self.sim.world.arena
        r_now, psi_now = _polar_coords(v.x, v.y, v.heading, src.position, arena)
        r_now, psi_now = float(r_now), float(psi_now)

        # Header
        title = self.small_font.render("POLAR PHASE PLANE  (r, ψ)", True, FLOW_ATTRACT)
        readout = self.tiny_font.render(
            f"r={r_now:.1f}  ψ={math.degrees(psi_now):+.0f}°", True, TEXT
        )
        card.blit(title, (ox + 9, oy + 6))
        card.blit(readout, (ox + cw - readout.get_width() - 9, oy + 7))

        has_gauge = (v.energy is not None) or bool(v.learning_inputs)
        gauge_h = 32 if has_gauge else 0
        px0, py0 = ox + 30, oy + 31
        pw = cw - 40
        ph = ch - 46 - gauge_h
        r_max = round(0.55 * math.hypot(arena.width, arena.height), 1)
        r_max = max(r_max, 25.0)

        # Cached phase-plane background (vector ticks + live dr=0 & dpsi=0 nullclines + equilibria).
        bg_key = (
            self.selected, pw, ph, stim_name,
            round(src.position[0], 2), round(src.position[1], 2),
            src.intensity, src.falloff, src.active(self.sim.time),
            self._vehicle_state_key(v)[:3],
        )
        if self.phase_bg_surface is None or bg_key != self.phase_bg_key:
            self.phase_bg_key = bg_key
            self.phase_bg_surface, self.phase_bg_meta = self._build_polar_phase_bg(
                v, stim_name, src, pw, ph, r_max
            )
        card.blit(self.phase_bg_surface, (px0, py0))

        def to_plot(rv: float, psiv: float) -> tuple[float, float]:
            u = min(max(rv / r_max, 0.0), 1.0)
            w = (psiv + math.pi) / (2.0 * math.pi)
            return px0 + u * pw, py0 + (1.0 - w) * ph

        # Axis ticks & labels
        for val, lbl_str in ((math.pi, "+π"), (0.0, "0"), (-math.pi, "-π")):
            _, ty = to_plot(0.0, val)
            lbl = self.tiny_font.render(lbl_str, True, PANEL_DIM)
            card.blit(lbl, (px0 - lbl.get_width() - 4, round(ty - lbl.get_height() / 2)))
        r0_lbl = self.tiny_font.render("0", True, PANEL_DIM)
        rm_lbl = self.tiny_font.render(f"r={r_max:.0f}", True, PANEL_DIM)
        card.blit(r0_lbl, (px0, py0 + ph + 2))
        card.blit(rm_lbl, (px0 + pw - rm_lbl.get_width(), py0 + ph + 2))
        leg_psi = self.tiny_font.render("ψ'=0", True, FLOW_ATTRACT)
        leg_r = self.tiny_font.render("r'=0", True, FLOW_REPEL)
        mid_x = px0 + pw // 2 - (leg_psi.get_width() + leg_r.get_width() + 10) // 2
        card.blit(leg_psi, (mid_x, py0 + ph + 2))
        card.blit(leg_r, (mid_x + leg_psi.get_width() + 10, py0 + ph + 2))

        _, focal_vehicles, hero_idx = self._focal_group()
        snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
        cw_ids = snap.get("vehicle_codeword_ids") if snap is not None else None
        hero_col = (
            codeword_color(int(cw_ids[hero_idx]))
            if cw_ids is not None and hero_idx < len(cw_ids)
            else FLOW_ORBIT
        )

        # Live velocity/state-colored comet trail on (r, psi), skipping cylindrical +/-pi wraps.
        hist = list(self.phase_history[self.selected])
        v_ref = max((a.max_speed for a in v.definition.actuators), default=5.0)
        v_ref = max(v_ref, 1e-3)
        prev_clip = card.get_clip()
        card.set_clip((px0, py0, pw, ph))
        n = len(hist)
        for k in range(1, n):
            r0, p0, tag0, spd0, _, ef0, mw0, *_ = hist[k - 1]
            r1, p1, tag1, spd1, _, ef1, mw1, *_ = hist[k]
            if tag0 != tag1 or abs(p1 - p0) > 1.65 or abs(r1 - r0) > 0.25 * r_max:
                continue
            fade = (k / n) ** 1.5
            if cw_ids is not None:
                col = lerp_color(FLOW_ATTRACT, hero_col, 0.35 + 0.65 * fade)
            elif v.learning_inputs:
                col = lerp_color(FLOW_ATTRACT, PHASE_EMERALD if mw1 > ACTIVE_STRENGTH else FLOW_ORBIT,
                                 min(mw1, 1.0))
            elif v.energy is not None:
                col = lerp_color(FLOW_REPEL, PHASE_EMERALD, min(max(ef1, 0.0), 1.0))
            else:
                col = lerp_color(FLOW_ATTRACT, FLOW_ORBIT, min(spd1 / v_ref, 1.0))
            pt0, pt1 = to_plot(r0, p0), to_plot(r1, p1)
            if k > n - 55:
                pygame.draw.line(card, (*col, int(70 * fade)), pt0, pt1, 5)
            pygame.draw.line(card, (*col, int(55 + 195 * fade)), pt0, pt1, 2)

        # Plot homogeneous focal peer ensemble hue-locked to live Shannon Codeword Hue!
        if len(focal_vehicles) >= 2:
            for fi, peer in enumerate(focal_vehicles):
                if fi == hero_idx:
                    continue
                p_src = self._primary_source_for(peer)
                if p_src is None:
                    continue
                rp, pp = _polar_coords(peer.x, peer.y, peer.heading, p_src[2].position, arena)
                px_i, py_i = to_plot(float(rp), float(pp))
                p_col = codeword_color(int(cw_ids[fi])) if cw_ids is not None else FLOW_ATTRACT
                pygame.draw.circle(card, (*p_col, 80), (round(px_i), round(py_i)), 6)
                pygame.draw.circle(card, p_col, (round(px_i), round(py_i)), 3)

        # Hero comet head + signature cyan selection target ring!
        hx, hy = to_plot(r_now, psi_now)
        pygame.draw.circle(card, (*hero_col, 95), (round(hx), round(hy)), 9)
        pygame.draw.circle(card, hero_col, (round(hx), round(hy)), 4)
        pygame.draw.circle(card, (255, 255, 255), (round(hx), round(hy)), 2)
        if len(focal_vehicles) >= 2:
            pygame.draw.circle(card, SELECTED_RING, (round(hx), round(hy)), 8, 2)
        card.set_clip(prev_clip)

        if has_gauge:
            self._draw_secondary_gauge(card, ox + 10, py0 + ph + 15, cw - 20, gauge_h - 4, v, hist)

    def _draw_target_polar_phase_card(
        self, card: pygame.Surface, ox: int, oy: int, cw: int, ch: int, v
    ) -> None:
        """Target-relative symmetry space `(r_target, psi_bearing)` for camera vehicles (Vehicle 9 Shy)."""
        arena = self.sim.world.arena
        tgt_info = self._primary_target_vehicle_for(self.selected, v)
        if tgt_info is not None:
            r_now, psi_now = _polar_coords(v.x, v.y, v.heading, (tgt_info[2].x, tgt_info[2].y), arena)
            r_now, psi_now = float(r_now), float(psi_now)
        else:
            r_now, psi_now = 0.0, 0.0

        title = self.small_font.render("TARGET PHASE  (r, ψ)", True, FLOW_ATTRACT)
        readout = self.tiny_font.render(
            f"r={r_now:.1f}  ψ={math.degrees(psi_now):+.0f}°", True, TEXT
        )
        card.blit(title, (ox + 9, oy + 6))
        card.blit(readout, (ox + cw - readout.get_width() - 9, oy + 7))

        px0, py0 = ox + 30, oy + 31
        pw, ph = cw - 40, ch - 46
        r_max = max(45.0, round(0.55 * math.hypot(arena.width, arena.height), 1))

        pygame.draw.rect(card, HUD_PLOT_BG, (px0, py0, pw, ph), border_radius=5)

        def to_plot(rv: float, psiv: float) -> tuple[float, float]:
            u = min(max(rv / r_max, 0.0), 1.0)
            w = (psiv + math.pi) / (2.0 * math.pi)
            return px0 + u * pw, py0 + (1.0 - w) * ph

        for frac in (0.25, 0.5, 0.75):
            gx_p = px0 + round(frac * pw)
            pygame.draw.line(card, HUD_GRID_LINE, (gx_p, py0), (gx_p, py0 + ph), 1)

        # Camera FOV cone & range boundaries (e.g. range=30, FOV=+-30 deg on Vehicle 9 Shy).
        cam_sensor = next((s for s in v.definition.sensors if s.camera), None)
        if cam_sensor is not None:
            r_cam = float(cam_sensor.camera.range)
            half_fov = math.radians(cam_sensor.field_of_view * 0.5)
            cx_r, _ = to_plot(r_cam, 0.0)
            _, y_top = to_plot(0.0, half_fov)
            _, y_bot = to_plot(0.0, -half_fov)
            fov_rect = pygame.Rect(
                px0, round(min(y_top, y_bot)),
                max(2, round(cx_r - px0)), max(2, round(abs(y_bot - y_top)))
            )
            pygame.draw.rect(card, (*FLOW_ATTRACT, 24), fov_rect)
            if r_cam < r_max:
                draw_dashed_line(card, (*FLOW_ORBIT, 170), (cx_r, py0), (cx_r, py0 + ph), 1, dash=4)
            draw_dashed_line(card, (*FLOW_ATTRACT, 135), (px0, y_top), (cx_r, y_top), 1, dash=4)
            draw_dashed_line(card, (*FLOW_ATTRACT, 135), (px0, y_bot), (cx_r, y_bot), 1, dash=4)

        # Head-on symmetry trigger axis psi = 0
        _, y_zero = to_plot(0.0, 0.0)
        draw_dashed_line(card, (*FLOW_REPEL, 185), (px0, y_zero), (px0 + pw, y_zero), 1, dash=4)
        pygame.draw.rect(card, (75, 110, 170, 130), (px0, py0, pw, ph), 1, border_radius=5)

        for val, lbl_str in ((math.pi, "+π"), (0.0, "0"), (-math.pi, "-π")):
            _, ty = to_plot(0.0, val)
            lbl = self.tiny_font.render(lbl_str, True, PANEL_DIM)
            card.blit(lbl, (px0 - lbl.get_width() - 4, round(ty - lbl.get_height() / 2)))
        card.blit(self.tiny_font.render("0", True, PANEL_DIM), (px0, py0 + ph + 2))
        rm_lbl = self.tiny_font.render(f"r={r_max:.0f}", True, PANEL_DIM)
        card.blit(rm_lbl, (px0 + pw - rm_lbl.get_width(), py0 + ph + 2))
        sym_lbl = self.tiny_font.render("ψ=0 sym", True, FLOW_REPEL)
        fov_lbl = self.tiny_font.render("FOV", True, FLOW_ATTRACT)
        mid_x = px0 + pw // 2 - (sym_lbl.get_width() + fov_lbl.get_width() + 10) // 2
        card.blit(sym_lbl, (mid_x, py0 + ph + 2))
        card.blit(fov_lbl, (mid_x + sym_lbl.get_width() + 10, py0 + ph + 2))

        focal_indices, focal_vehicles, hero_idx = self._focal_group()
        snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
        cw_ids = snap.get("vehicle_codeword_ids") if snap is not None else None
        hero_col = (
            codeword_color(int(cw_ids[hero_idx]))
            if cw_ids is not None and hero_idx < len(cw_ids)
            else FLOW_ORBIT
        )

        prev_clip = card.get_clip()
        card.set_clip((px0, py0, pw, ph))
        hist = list(self.phase_history[self.selected])
        n = len(hist)
        for k in range(1, n):
            r0, p0, tag0, *_ = hist[k - 1]
            r1, p1, tag1, *_ = hist[k]
            if tag0 is None or tag0 != tag1 or abs(p1 - p0) > 1.65 or abs(r1 - r0) > 0.25 * r_max:
                continue
            fade = (k / n) ** 1.5
            col = lerp_color(FLOW_ATTRACT, hero_col, 0.30 + 0.70 * fade)
            pt0, pt1 = to_plot(r0, p0), to_plot(r1, p1)
            if k > n - 55:
                pygame.draw.line(card, (*col, int(65 * fade)), pt0, pt1, 5)
            pygame.draw.line(card, (*col, int(55 + 190 * fade)), pt0, pt1, 2)

        # Plot ONLY focal species peers (environmental Chasers/Walkers strictly excluded!).
        if len(focal_vehicles) >= 2:
            for fi, (sim_i, peer) in enumerate(zip(focal_indices, focal_vehicles)):
                if fi == hero_idx:
                    continue
                tp = self._primary_target_vehicle_for(sim_i, peer)
                if tp is None:
                    continue
                rp, pp = _polar_coords(peer.x, peer.y, peer.heading, (tp[2].x, tp[2].y), arena)
                px_i, py_i = to_plot(float(rp), float(pp))
                p_col = codeword_color(int(cw_ids[fi])) if cw_ids is not None else FLOW_ATTRACT
                pygame.draw.circle(card, (*p_col, 85), (round(px_i), round(py_i)), 6)
                pygame.draw.circle(card, p_col, (round(px_i), round(py_i)), 3)

        hx, hy = to_plot(r_now, psi_now)
        pygame.draw.circle(card, (*hero_col, 95), (round(hx), round(hy)), 9)
        pygame.draw.circle(card, hero_col, (round(hx), round(hy)), 4)
        pygame.draw.circle(card, (255, 255, 255), (round(hx), round(hy)), 2)
        pygame.draw.circle(card, SELECTED_RING, (round(hx), round(hy)), 8, 2)
        card.set_clip(prev_clip)

    def _build_polar_phase_bg(
        self, v, stim_name: str, src, pw: int, ph: int, r_max: float
    ) -> tuple[pygame.Surface, dict]:
        bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(bg, HUD_PLOT_BG, (0, 0, pw, ph), border_radius=5)

        # Grid hairlines: psi = -pi/2, 0, +pi/2 and r quartiles.
        for frac in (0.25, 0.5, 0.75):
            gx_p = round(frac * pw)
            pygame.draw.line(bg, HUD_GRID_LINE, (gx_p, 0), (gx_p, ph), 1)
        for psi_mark, is_zero in ((-math.pi / 2, False), (0.0, True), (math.pi / 2, False)):
            gy_p = round((1.0 - (psi_mark + math.pi) / (2.0 * math.pi)) * ph)
            if is_zero:
                draw_dashed_line(bg, HUD_AXIS_LINE, (0, gy_p), (pw, gy_p), 1, dash=4)
            else:
                pygame.draw.line(bg, HUD_GRID_LINE, (0, gy_p), (pw, gy_p), 1)

        nr, npsi = 28, 26  # even npsi straddles psi=0 cleanly without zero-degeneracy
        r_grid = np.linspace(0.8, r_max, nr)
        psi_grid = np.linspace(-math.pi + 0.05, math.pi - 0.05, npsi)

        used_core = False
        dr = dpsi = None
        if not v.definition.units and src.active(self.sim.time):
            pf = polar_field(
                v.definition, self.sim.world, r_grid=r_grid, psi_grid=psi_grid,
                source_pos=src.position, stimulus_name=stim_name, dt=self.sim.dt
            )
            dr = np.asarray(pf["dr"], dtype=float)
            dpsi = np.asarray(pf["dpsi"], dtype=float)
            used_core = True

        if not used_core:
            arena = self.sim.world.arena
            sx, sy = src.position
            ray_ang = math.atan2(arena.height / 2 - sy, arena.width / 2 - sx)
            rr, pp = np.meshgrid(r_grid, psi_grid)  # shape (npsi, nr)
            qx = (sx + rr * np.cos(ray_ang)) % arena.width
            qy = (sy + rr * np.sin(ray_ang)) % arena.height
            alpha_ray = _wrap_pi(ray_ang + math.pi)
            qt = alpha_ray - pp
            v_f, omg_f = _eval_kinematics_batch(
                v, self.sim.world, self.sim.time, qx.ravel(), qy.ravel(), qt.ravel()
            )
            vv = v_f.reshape(npsi, nr)
            oo = omg_f.reshape(npsi, nr)
            dr = -vv * np.cos(pp)
            dpsi = (vv / np.maximum(rr, 0.8)) * np.sin(pp) - oo

        def to_px(rv: float, psiv: float) -> tuple[float, float]:
            return (rv / r_max) * pw, (1.0 - (psiv + math.pi) / (2.0 * math.pi)) * ph

        # Coarse background phase-flow direction ticks.
        for j in range(1, npsi, 4):
            for i in range(1, nr, 4):
                u_r = float(dr[j, i]) / max(r_max, 1.0)
                u_p = float(dpsi[j, i]) / (2.0 * math.pi)
                mag = math.hypot(u_r * pw, u_p * ph)
                if mag < 1e-6:
                    continue
                dx_p = (u_r * pw / mag) * 4.5
                dy_p = -(u_p * ph / mag) * 4.5
                cx_p, cy_p = to_px(float(r_grid[i]), float(psi_grid[j]))
                pygame.draw.aaline(bg, (85, 118, 175, 95),
                                   (cx_p - dx_p, cy_p - dy_p), (cx_p + dx_p, cy_p + dy_p))
                pygame.draw.circle(bg, (115, 155, 215, 115),
                                   (round(cx_p + dx_p), round(cy_p + dy_p)), 1)

        # Live nullcline contours: dr = 0 (coral) and dpsi = 0 (cyan).
        for seg_a, seg_b in _zero_contours(dr, r_grid, psi_grid):
            p0, p1 = to_px(*seg_a), to_px(*seg_b)
            pygame.draw.line(bg, (*FLOW_REPEL, 48), p0, p1, 3)
            pygame.draw.aaline(bg, (*FLOW_REPEL, 175), p0, p1)
        for seg_a, seg_b in _zero_contours(dpsi, r_grid, psi_grid):
            p0, p1 = to_px(*seg_a), to_px(*seg_b)
            pygame.draw.line(bg, (*FLOW_ATTRACT, 55), p0, p1, 3)
            pygame.draw.aaline(bg, (*FLOW_ATTRACT, 205), p0, p1)

        # Strogatz equilibrium markers on (r, psi): solid = stable, open ring = unstable/saddle.
        others = [s2.position for _, _, s2 in self._all_sensed_sources(v, active_only=True)
                  if math.hypot(s2.position[0] - src.position[0], s2.position[1] - src.position[1]) > 1.0]
        if used_core and not others:
            eq_info = self._cached_equilibria(v, min(46.0, r_max))
            for fp in eq_info.get("fixed_points", ()):
                r_fp, psi_fp = float(fp["r"]), float(fp["psi"])
                if not (0.0 <= r_fp <= r_max):
                    continue
                ex, ey = to_px(r_fp, psi_fp)
                stab = fp.get("stability", "")
                if stab == "stable":
                    col = FLOW_ORBIT if abs(psi_fp) > 0.6 else FLOW_ATTRACT
                    pygame.draw.circle(bg, (*col, 90), (round(ex), round(ey)), 6)
                    pygame.draw.circle(bg, col, (round(ex), round(ey)), 3)
                else:
                    pygame.draw.circle(bg, HUD_PLOT_BG, (round(ex), round(ey)), 4)
                    pygame.draw.circle(bg, (*FLOW_REPEL, 215), (round(ex), round(ey)), 4, 1)
        else:
            rings = self._detect_invariant_rings(v, src.position, 1.0, others)
            for r_eq, kind in rings:
                if kind == "orbit":
                    for sgn in (-1.0, 1.0):
                        ex, ey = to_px(r_eq, sgn * math.pi / 2)
                        pygame.draw.circle(bg, (*FLOW_ORBIT, 90), (round(ex), round(ey)), 6)
                        pygame.draw.circle(bg, FLOW_ORBIT, (round(ex), round(ey)), 3)
                elif kind == "rest":
                    ex, ey = to_px(r_eq, 0.0)
                    pygame.draw.circle(bg, (*FLOW_ATTRACT, 95), (round(ex), round(ey)), 5)
                    pygame.draw.circle(bg, FLOW_ATTRACT, (round(ex), round(ey)), 3)

        pygame.draw.rect(bg, (75, 110, 170, 130), (0, 0, pw, ph), 1, border_radius=5)
        return bg, {"used_core": used_core, "r_max": r_max}

    def _draw_secondary_gauge(
        self, card: pygame.Surface, gx: int, gy: int, gw: int, gh: int, v, hist: list
    ) -> None:
        has_e = v.energy is not None and v.definition.energy is not None
        has_w = bool(v.learning_inputs)

        if has_e and has_w:
            half = (gw - 10) // 2
            self._draw_energy_bar(card, gx, gy, half, gh, v, hist)
            self._draw_learning_bar(card, gx + half + 10, gy, half, gh, v, hist)
        elif has_w:
            self._draw_learning_bar(card, gx, gy, gw, gh, v, hist)
        elif has_e:
            self._draw_energy_bar(card, gx, gy, gw, gh, v, hist)

    def _draw_energy_bar(
        self, card: pygame.Surface, gx: int, gy: int, gw: int, gh: int, v, hist: list
    ) -> None:
        emax = max(v.definition.energy.max, 1e-6)
        efrac = min(max(v.energy / emax, 0.0), 1.0)
        lbl = self.tiny_font.render(f"E {v.energy:.0f}/{emax:g}", True, PHASE_EMERALD)
        card.blit(lbl, (gx, gy + 2))
        bx = gx + lbl.get_width() + 6
        bw = max(30, gw - lbl.get_width() - 6)
        by = gy + 4
        bh = 10
        pygame.draw.rect(card, HUD_PLOT_BG, (bx, by, bw, bh), border_radius=4)
        _, focal_vehicles, hero_idx = self._focal_group()
        snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
        cw_ids = snap.get("vehicle_codeword_ids") if snap is not None else None
        if len(focal_vehicles) >= 2:
            for fi, peer in enumerate(focal_vehicles):
                if peer.energy is None:
                    continue
                pf = min(max(peer.energy / emax, 0.0), 1.0)
                px_e = bx + 1 + round((bw - 3) * pf)
                pcol = codeword_color(int(cw_ids[fi])) if cw_ids is not None else PHASE_EMERALD
                pygame.draw.line(card, (*pcol, 175), (px_e, by + 1), (px_e, by + bh - 2), 2)
            hx_e = bx + 1 + round((bw - 3) * efrac)
            pygame.draw.line(card, SELECTED_RING, (hx_e, by - 2), (hx_e, by + bh + 2), 2)
        else:
            fill_w = round(bw * efrac)
            if fill_w > 0:
                col = lerp_color(FLOW_REPEL, PHASE_EMERALD, efrac)
                pygame.draw.rect(card, (*col, 220), (bx, by, fill_w, bh), border_radius=4)
        # Mark energy latch thresholds (e.g., 30 hungry / 90 charged on Forager).
        for u in v.definition.units:
            for c in u.inputs:
                if c.source == ENERGY_SENSOR or any(
                    s.name == c.source and s.type == ENERGY_SENSOR for s in v.definition.sensors
                ):
                    thr = abs(u.threshold)
                    if 0 < thr < emax:
                        tx = bx + round(bw * (thr / emax))
                        pygame.draw.line(card, FLOW_ORBIT, (tx, by - 2), (tx, by + bh + 2), 1)
        pygame.draw.rect(card, (80, 120, 175, 140), (bx, by, bw, bh), 1, border_radius=4)

    def _draw_learning_bar(
        self, card: pygame.Surface, gx: int, gy: int, gw: int, gh: int, v, hist: list
    ) -> None:
        max_w = max(v.strengths.values(), default=0.0)
        active = max_w > ACTIVE_STRENGTH
        col = PHASE_EMERALD if active else FLOW_ORBIT
        lbl = self.tiny_font.render(f"max w={max_w:.2f}", True, col)
        card.blit(lbl, (gx, gy + 2))
        bx = gx + lbl.get_width() + 6
        bw = max(36, gw - lbl.get_width() - 6)
        by = gy + 4
        bh = 10
        pygame.draw.rect(card, HUD_PLOT_BG, (bx, by, bw, bh), border_radius=4)
        _, focal_vehicles, _ = self._focal_group()
        snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
        cw_ids = snap.get("vehicle_codeword_ids") if snap is not None else None
        if len(focal_vehicles) >= 2:
            for fi, peer in enumerate(focal_vehicles):
                pw_max = min(max(max(peer.strengths.values(), default=0.0), 0.0), 1.0)
                px_w = bx + 1 + round((bw - 3) * pw_max)
                pcol = codeword_color(int(cw_ids[fi])) if cw_ids is not None else col
                pygame.draw.line(card, (*pcol, 185), (px_w, by + 1), (px_w, by + bh - 2), 2)
            hx_w = bx + 1 + round((bw - 3) * min(max(max_w, 0.0), 1.0))
            pygame.draw.line(card, SELECTED_RING, (hx_w, by - 2), (hx_w, by + bh + 2), 2)
        else:
            fill_w = round(bw * min(max(max_w, 0.0), 1.0))
            if fill_w > 0:
                pygame.draw.rect(card, (*(PHASE_EMERALD if active else (55, 145, 130)), 225),
                                 (bx, by, fill_w, bh), border_radius=4)
        # Critical synaptic bifurcation threshold wc = 0.7!
        tx = bx + round(bw * STRENGTH_TICK)
        pygame.draw.line(card, FLOW_ORBIT, (tx, by - 3), (tx, by + bh + 3), 2)
        wc_lbl = self.tiny_font.render("0.7", True, FLOW_ORBIT)
        card.blit(wc_lbl, (max(bx, min(bx + bw - wc_lbl.get_width(), tx - wc_lbl.get_width() // 2)),
                           by + bh + 2))
        pygame.draw.rect(card, (80, 120, 175, 140), (bx, by, bw, bh), 1, border_radius=4)

    def _draw_neural_mandala_card(
        self, card: pygame.Surface, ox: int, oy: int, cw: int, ch: int, v
    ) -> None:
        """Complex Neural Ring Mandala Z(t) + (v, omega) trace for autonomous/sensorless vehicles."""
        z_re, z_im, z_mag, active_sum, ring_units = _ring_order_parameter(v)
        z_ang = math.degrees(math.atan2(z_im, z_re)) if z_mag > 1e-6 else 0.0

        title = self.small_font.render("NEURAL RING  Z(t)", True, PHASE_VIOLET)
        readout = self.tiny_font.render(f"|Z|={z_mag:.2f}  Φ={z_ang:+.0f}°", True, TEXT)
        card.blit(title, (ox + 9, oy + 6))
        card.blit(readout, (ox + cw - readout.get_width() - 9, oy + 7))

        _, focal_vehicles, hero_idx = self._focal_group()
        snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
        cw_ids = snap.get("vehicle_codeword_ids") if snap is not None else None
        hero_col = (
            codeword_color(int(cw_ids[hero_idx]))
            if cw_ids is not None and hero_idx < len(cw_ids)
            else PHASE_VIOLET
        )

        # Left subpanel: unit-disc neural mandala & complex order parameter Z(t).
        disc_w = round(cw * 0.50)
        mcx = ox + 8 + disc_w // 2
        mcy = oy + 28 + (ch - 38) // 2
        rad = min(disc_w // 2 - 14, (ch - 54) // 2)
        rad = max(rad, 32)

        pygame.draw.circle(card, HUD_PLOT_BG, (mcx, mcy), rad + 5)
        pygame.draw.circle(card, HUD_GRID_LINE, (mcx, mcy), round(rad * 0.5), 1)
        pygame.draw.line(card, HUD_GRID_LINE, (mcx - rad, mcy), (mcx + rad, mcy), 1)
        pygame.draw.line(card, HUD_GRID_LINE, (mcx, mcy - rad), (mcx, mcy + rad), 1)
        pygame.draw.circle(card, HUD_AXIS_LINE, (mcx, mcy), rad, 1)

        # Radial display magnification sqrt(|Z|) so sparse 1-of-N waves sweep a clear rose curve.
        def z_to_px(zr: float, zi: float) -> tuple[float, float]:
            m = math.hypot(zr, zi)
            if m < 1e-9:
                return float(mcx), float(mcy)
            scale = (min(m, 1.0) ** 0.55) * (rad - 6) / m
            return mcx + zr * scale, mcy - zi * scale

        hist = list(self.phase_history[self.selected])
        nh = len(hist)
        for k in range(1, nh):
            zr0, zi0 = hist[k - 1][7], hist[k - 1][8]
            zr1, zi1 = hist[k][7], hist[k][8]
            fade = (k / nh) ** 1.6
            p0, p1 = z_to_px(zr0, zi0), z_to_px(zr1, zi1)
            col = lerp_color(FLOW_ATTRACT, hero_col, fade)
            pygame.draw.line(card, (*col, int(45 + 185 * fade)), p0, p1, 2)

        # Draw ring neurons around outer rim.
        n_ring = len(ring_units)
        if n_ring > 0:
            node_pts = []
            for k, u in enumerate(ring_units):
                ang = 2.0 * math.pi * k / n_ring
                nx = mcx + rad * math.cos(ang)
                ny = mcy - rad * math.sin(ang)
                node_pts.append((nx, ny, v.unit_outputs.get(u.name, 0.0)))
            for k, (nx, ny, act) in enumerate(node_pts):
                if act > 0.5:
                    pygame.draw.circle(card, (*FLOW_ORBIT, 95), (round(nx), round(ny)), 7)
                    pygame.draw.circle(card, FLOW_ORBIT, (round(nx), round(ny)), 4)
                else:
                    pygame.draw.circle(card, (65, 80, 125, 190), (round(nx), round(ny)), 3)
                    pygame.draw.circle(card, (125, 145, 195, 180), (round(nx), round(ny)), 3, 1)

        # Plot focal peer Z_i(t) order parameters hue-locked to Shannon Codeword Hue!
        if len(focal_vehicles) >= 2:
            for fi, peer in enumerate(focal_vehicles):
                if fi == hero_idx:
                    continue
                p_zr, p_zi, *_ = _ring_order_parameter(peer)
                px_z, py_z = z_to_px(p_zr, p_zi)
                p_col = codeword_color(int(cw_ids[fi])) if cw_ids is not None else FLOW_ATTRACT
                pygame.draw.circle(card, (*p_col, 80), (round(px_z), round(py_z)), 5)
                pygame.draw.circle(card, p_col, (round(px_z), round(py_z)), 3)

        zx, zy = z_to_px(z_re, z_im)
        pygame.draw.line(card, (*hero_col, 190), (mcx, mcy), (zx, zy), 2)
        pygame.draw.circle(card, (*hero_col, 95), (round(zx), round(zy)), 8)
        pygame.draw.circle(card, hero_col, (round(zx), round(zy)), 4)
        pygame.draw.circle(card, (255, 255, 255), (round(zx), round(zy)), 2)
        if len(focal_vehicles) >= 2:
            pygame.draw.circle(card, SELECTED_RING, (round(zx), round(zy)), 7, 2)

        cap = self.tiny_font.render(f"ring active: {active_sum:g}/{max(n_ring, 1)}", True, PANEL_DIM)
        card.blit(cap, (mcx - cap.get_width() // 2, mcy + rad + 5))

        # Right subpanel: (v, omega) kinematic phase portrait + caudate control gauge.
        rx0 = ox + disc_w + 14
        rw = cw - disc_w - 22
        controls = [u for u in v.definition.units if u.is_control]
        ctrl_h = 28 if controls else 0
        ry0 = oy + 32
        rh = ch - 48 - ctrl_h

        pygame.draw.rect(card, HUD_PLOT_BG, (rx0, ry0, rw, rh), border_radius=5)
        mid_y = ry0 + rh // 2
        draw_dashed_line(card, HUD_AXIS_LINE, (rx0, mid_y), (rx0 + rw, mid_y), 1, dash=4)
        pygame.draw.rect(card, (75, 110, 170, 130), (rx0, ry0, rw, rh), 1, border_radius=5)

        v_max = max((a.max_speed for a in v.definition.actuators), default=4.0)
        v_max = max(v_max, 1.0)
        omg_max = max(
            max((abs(h[4]) for h in hist), default=0.8),
            max((abs(peer.turn_rate) for peer in focal_vehicles), default=0.8),
            0.8,
        )

        def vw_to_px(sv: float, ov: float) -> tuple[float, float]:
            u = min(max(sv / v_max, 0.0), 1.0)
            w = min(max(ov / (1.15 * omg_max), -1.0), 1.0)
            return rx0 + 4 + u * (rw - 8), mid_y - w * (rh * 0.44)

        for k in range(1, nh):
            s0, o0 = hist[k - 1][3], hist[k - 1][4]
            s1, o1 = hist[k][3], hist[k][4]
            fade = (k / nh) ** 1.5
            col = lerp_color(FLOW_ATTRACT, hero_col if cw_ids is not None else FLOW_ORBIT,
                             min(abs(o1) / omg_max, 1.0))
            pygame.draw.line(card, (*col, int(50 + 185 * fade)),
                             vw_to_px(s0, o0), vw_to_px(s1, o1), 2)

        if len(focal_vehicles) >= 2:
            for fi, peer in enumerate(focal_vehicles):
                if fi == hero_idx:
                    continue
                px_k, py_k = vw_to_px(peer.speed, peer.turn_rate)
                p_col = codeword_color(int(cw_ids[fi])) if cw_ids is not None else FLOW_ATTRACT
                pygame.draw.circle(card, (*p_col, 80), (round(px_k), round(py_k)), 5)
                pygame.draw.circle(card, p_col, (round(px_k), round(py_k)), 3)

        kx, ky = vw_to_px(v.speed, v.turn_rate)
        k_col = hero_col if cw_ids is not None else FLOW_ORBIT
        pygame.draw.circle(card, (*k_col, 90), (round(kx), round(ky)), 6)
        pygame.draw.circle(card, k_col, (round(kx), round(ky)), 3)
        if len(focal_vehicles) >= 2:
            pygame.draw.circle(card, SELECTED_RING, (round(kx), round(ky)), 7, 2)

        vw_lbl = self.tiny_font.render("(v, ω) trace", True, FLOW_ATTRACT)
        card.blit(vw_lbl, (rx0 + (rw - vw_lbl.get_width()) // 2, ry0 + rh + 2))

        if controls:
            ctrl = controls[0]
            offset_val = v.unit_outputs.get(ctrl.name, 0.0)
            cy_g = ry0 + rh + 15
            clbl = self.tiny_font.render(f"{ctrl.name} Δθ={offset_val:+.2f}", True, PHASE_VIOLET)
            card.blit(clbl, (rx0, cy_g))

    # --- diagram panel ---

    def draw_panel(self, surface: pygame.Surface) -> None:
        left, height = self.size[0], self.window_size[1]
        surface.fill(PANEL_BG, (left, 0, PANEL_WIDTH, height))
        pygame.draw.line(surface, PANEL_BORDER, (left, 0), (left, height), 2)
        x0 = left + 16
        if not self.sim.vehicles:
            surface.blit(self.font.render("No vehicles in this scenario", True, PANEL_DIM),
                         (x0, 16))
            return
        diagram = self.sim.vehicles[self.selected].diagram()
        if self.dynamics_mode >= 2:
            _, focal_vehicles, hero_idx = self._focal_group()
            snap = self._get_live_shannon_snapshot() if len(focal_vehicles) >= 2 else None
            if len(focal_vehicles) >= 2 and snap is not None:
                self._draw_shannon_panel(surface, diagram, focal_vehicles, hero_idx, snap)
                return

        surface.blit(self.title_font.render(diagram.title, True, TEXT), (x0, 14))
        surface.blit(self.font.render(diagram.subtitle, True, PANEL_DIM), (x0, 38))
        # Readouts and vehicle values fill the bottom; the diagram gets the space above them.
        learning = [w for w in diagram.wires if w.learning]
        readouts_top = height - 10 - ROW * (len(diagram.parts) + len(learning) + 2) - 8
        top = 62
        radius = min(MAX_DIAGRAM_RADIUS, (PANEL_WIDTH - 60) / 2, (readouts_top - top - 60) / 2)
        center = (left + PANEL_WIDTH // 2, top + 30 + radius)
        self.draw_diagram(surface, diagram, center, radius)
        y = self.draw_readouts(surface, diagram, x0, min(readouts_top, center[1] + radius + 24))
        self.draw_params(surface, diagram, x0, y + 8)

    def _draw_shannon_panel(
        self,
        surface: pygame.Surface,
        diagram: Diagram,
        focal_vehicles: list,
        hero_idx: int,
        snap: dict,
    ) -> None:
        """Live Hero + Shannon Ensemble Connectome Panel (`PANEL_WIDTH = 420`)."""
        left, height = self.size[0], self.window_size[1]
        x0 = left + 16
        inner_w = PANEL_WIDTH - 32

        cw_ids = snap.get("vehicle_codeword_ids")
        hero_col = (
            codeword_color(int(cw_ids[hero_idx]))
            if cw_ids is not None and hero_idx < len(cw_ids)
            else FLOW_ORBIT
        )

        # Header: Hero title with signature cyan selection ring + Homogeneous Ensemble badge.
        pygame.draw.circle(surface, hero_col, (x0 + 7, 22), 4)
        pygame.draw.circle(surface, SELECTED_RING, (x0 + 7, 22), 7, 2)
        title_surf = self.title_font.render(diagram.title, True, TEXT)
        surface.blit(title_surf, (x0 + 19, 12))
        sub_surf = self.small_font.render(diagram.subtitle, True, PANEL_DIM)
        surface.blit(sub_surf, (x0, 35))

        b_line1 = self.tiny_font.render(f"Focal N={len(focal_vehicles)}", True, FLOW_ATTRACT)
        b_line2 = self.tiny_font.render(f"[N] {self.neural_view}", True, PHASE_EMERALD)
        pill_w = max(b_line1.get_width(), b_line2.get_width()) + 16
        pill_h = 34
        pill_x = left + PANEL_WIDTH - 14 - pill_w
        pill = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
        pygame.draw.rect(pill, (14, 19, 38, 235), (0, 0, pill_w, pill_h), border_radius=7)
        pygame.draw.rect(pill, (68, 96, 150, 185), (0, 0, pill_w, pill_h), 1, border_radius=7)
        pill.blit(b_line1, ((pill_w - b_line1.get_width()) // 2, 4))
        pill.blit(b_line2, ((pill_w - b_line2.get_width()) // 2, 18))
        surface.blit(pill, (pill_x, 12))

        learning = [w for w in diagram.wires if w.learning]
        n_rows = len(diagram.parts) + len(learning)
        params_h = 34
        row_h = (
            18 if n_rows <= 12
            else 16 if n_rows <= 18
            else 15 if n_rows <= 24
            else 14
        )
        strip_h = 98 if self.neural_view == "raster" else 72
        diag_top = 54
        min_diag_room = 206 if self.neural_view == "raster" else 242
        diag_room = height - diag_top - params_h - n_rows * row_h - strip_h - 14
        if diag_room < min_diag_room and row_h > 14:
            row_h = max(14, row_h - 1)
            diag_room = height - diag_top - params_h - n_rows * row_h - strip_h - 14
        if diag_room < min_diag_room and strip_h > 58:
            strip_h = max(58, strip_h - (min_diag_room - diag_room))
            diag_room = height - diag_top - params_h - n_rows * row_h - strip_h - 14

        radius = min(MAX_DIAGRAM_RADIUS, (PANEL_WIDTH - 64) / 2, max(80.0, (diag_room - 44) / 2))
        center = (left + PANEL_WIDTH // 2, round(diag_top + 20 + radius))
        self._draw_shannon_diagram(surface, diagram, center, radius, snap)

        max_strip_y = height - params_h - n_rows * row_h - strip_h - 6
        strip_y = min(round(center[1] + radius + 24), max_strip_y)
        self._draw_shannon_strip(surface, x0, strip_y, inner_w, strip_h, focal_vehicles, hero_idx, snap)

        readouts_y = strip_y + strip_h + 6
        end_y = self._draw_shannon_readouts(surface, diagram, x0, readouts_y, inner_w, row_h, snap)
        self._draw_shannon_params(surface, diagram, x0, end_y + 3)

    def _get_shannon_ring_patch(
        self,
        arcs: list[tuple[int, float]],
        entropy_bits: float,
        r_in: float = 11.8,
        r_out: float = 15.4,
    ) -> pygame.Surface:
        """Build (and cache) a 2x-supersampled antialiased Codebook Arc-Ring + Entropy Halo."""
        q_arcs = tuple((int(c), round(float(p), 3)) for c, p in arcs if float(p) > 0.004)
        q_ent = round(min(max(float(entropy_bits), 0.0), 1.0), 2)
        key = (q_arcs, q_ent, self.neural_view, round(r_in, 1), round(r_out, 1))
        cached = self._ring_patch_cache.get(key)
        if cached is not None:
            return cached

        half = 22
        ss = 2
        dim_px = 2 * half * ss
        cxy = half * ss
        patch_hi = pygame.Surface((dim_px, dim_px), pygame.SRCALPHA)

        # 1. Outer Marginal Entropy H(A_k) glow (0 bits -> zero halo!).
        if self.neural_view in ("composite", "entropy") and q_ent > 0.02:
            ent_col = lerp_color(PHASE_VIOLET, FLOW_ORBIT, q_ent)
            base_a = 120 if self.neural_view == "entropy" else 68
            pygame.draw.circle(
                patch_hi, (*ent_col, int(base_a * 0.45 * q_ent)), (cxy, cxy), round((r_out + 3.4) * ss), 6
            )
            pygame.draw.circle(
                patch_hi, (*ent_col, int(base_a * q_ent)), (cxy, cxy), round((r_out + 1.6) * ss), 3
            )
            if self.neural_view == "entropy":
                _fill_annular_sector(
                    patch_hi, cxy, cxy, (r_out + 0.8) * ss, (r_out + 2.4) * ss,
                    math.pi / 2, math.pi / 2 - 2.0 * math.pi * q_ent, (*FLOW_ORBIT, 225)
                )

        # 2. Quiescent ring track (1 - P(A_k = 1)).
        _fill_annular_sector(
            patch_hi, cxy, cxy, r_in * ss, r_out * ss,
            math.pi / 2, math.pi / 2 - 2.0 * math.pi, (38, 44, 68, 220)
        )

        # 3. Active Joint Codebook Arc Sectors (sum of spans = 2 * pi * P(A_k = 1)).
        ang_cur = math.pi / 2
        multi = len(q_arcs) >= 2
        for cw_idx, prob in q_arcs:
            span = 2.0 * math.pi * min(max(prob, 0.0), 1.0)
            gap = 0.04 if (multi and span > 0.14) else 0.0
            a0 = ang_cur - 0.5 * gap
            a1 = ang_cur - span + 0.5 * gap
            if a0 - a1 > 0.01:
                col = codeword_color(cw_idx)
                alpha_arc = 250 if self.neural_view != "entropy" else 180
                _fill_annular_sector(patch_hi, cxy, cxy, r_in * ss, r_out * ss, a0, a1, (*col, alpha_arc))
            ang_cur -= span

        patch = pygame.transform.smoothscale(patch_hi, (2 * half, 2 * half))
        if len(self._ring_patch_cache) > 256:
            self._ring_patch_cache.clear()
        self._ring_patch_cache[key] = patch
        return patch

    def _draw_shannon_diagram(
        self,
        surface: pygame.Surface,
        diagram: Diagram,
        center: tuple[int, int],
        radius: float,
        snap: dict,
    ) -> None:
        def at(position):
            return (center[0] + position[0] * radius * diagram.body[0],
                    center[1] - position[1] * radius * diagram.body[1])

        bw, bl = diagram.body
        body = pygame.Rect(0, 0, 2 * radius * bw, 2 * radius * bl)
        body.center = center
        pygame.draw.rect(surface, (28, 32, 48), body, border_radius=int(radius * min(bw, bl) * 0.3))
        pygame.draw.rect(surface, BODY_OUTLINE, body, 2, border_radius=int(radius * min(bw, bl) * 0.3))

        parts = {p.name: p for p in diagram.parts}
        arcs_map: dict = snap.get("part_codeword_arcs", {}) or {}
        ent_map: dict = snap.get("part_entropy", {}) or {}

        for p in diagram.parts:
            if p.kind == "module":
                self.draw_module(surface, p, at(p.position))

        # Separate parallel / antiparallel wires between the same pair so MI/TE halos stay crisp.
        bundled = bundles(diagram.wires)
        pair_counts: dict[frozenset, int] = {}
        pair_dirs: dict[frozenset, set] = {}
        for w in bundled:
            if w.source != w.target:
                pk = frozenset((w.source, w.target))
                pair_counts[pk] = pair_counts.get(pk, 0) + 1
                pair_dirs.setdefault(pk, set()).add((w.source, w.target))
        pair_seen: dict[frozenset, int] = {}

        glow_rect = body.inflate(64, 64)
        wire_glow = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        gx0, gy0 = glow_rect.topleft

        wire_geom = []
        for w in bundled:
            a, b = wire_ends(w, parts, at, PANEL_PORT_GAP, PANEL_BOX_HEIGHT)
            if w.source != w.target:
                pk = frozenset((w.source, w.target))
                idx_p = pair_seen[pk] = pair_seen.get(pk, -1) + 1
                cnt_p = pair_counts[pk]
                if cnt_p > 1:
                    dx_w, dy_w = b[0] - a[0], b[1] - a[1]
                    len_w = math.hypot(dx_w, dy_w) or 1.0
                    ux_w, uy_w = dx_w / len_w, dy_w / len_w
                    shift = (idx_p - (cnt_p - 1) / 2.0) * 5.0
                    if len(pair_dirs[pk]) > 1 and w.source > w.target:
                        shift = -shift
                    a = (a[0] - uy_w * shift, a[1] + ux_w * shift)
                    b = (b[0] - uy_w * shift, b[1] + ux_w * shift)
            info_bits = _lookup_wire_bits(snap, w)
            wire_geom.append((w, a, b, info_bits))

            # Draw live Channel MI / Transfer Entropy halo on wire_glow layer.
            if not w.control and self.neural_view in ("composite", "entropy") and info_bits > 0.015:
                info_norm = min(info_bits / 0.75, 1.0)
                halo_col = (
                    PHASE_EMERALD if w.learning
                    else FLOW_ATTRACT if w.sign > 0 else FLOW_REPEL
                )
                halo_w = round(3 + (6 if self.neural_view == "entropy" else 4.5) * info_norm)
                halo_a = int((55 if self.neural_view == "entropy" else 38) + 125 * info_norm)
                if w.source == w.target:
                    o = -1 if parts[w.target].position[0] < 0 else 1
                    pygame.draw.circle(
                        wire_glow, (*halo_col, halo_a),
                        (round(a[0] - gx0 + o * 12), round(a[1] - gy0 - 12)), 12, min(halo_w, 6)
                    )
                else:
                    pygame.draw.line(
                        wire_glow, (*halo_col, halo_a),
                        (a[0] - gx0, a[1] - gy0), (b[0] - gx0, b[1] - gy0), halo_w
                    )

        surface.blit(wire_glow, (gx0, gy0))

        # Draw Hero wire cores + compact sign badges.
        for w, a, b, info_bits in wire_geom:
            if w.control:
                draw_dashed_line(surface, CONTROL_COLOR, a, b, 1, dash=4)
                continue
            if w.learning:
                dim, on = learning_colors(w)
                color = lerp_color(dim, on, w.level) if w.active else dim
                width = 1 + round(3 * w.strength)
            else:
                if not w.active:
                    color, width = WIRE_INACTIVE, 1
                elif w.level > 0.05:
                    color = lerp_color(WIRE_OFF, WIRE_ON, w.level)
                    width = 3
                elif self.neural_view == "entropy" and info_bits > 0.03:
                    t_i = min(info_bits / 0.75, 1.0)
                    color = lerp_color(WIRE_OFF, FLOW_ATTRACT if w.sign > 0 else FLOW_REPEL, 0.75 * t_i)
                    width = 2
                else:
                    color, width = WIRE_OFF, 1
            if w.source == w.target:
                o = -1 if parts[w.target].position[0] < 0 else 1
                pygame.draw.circle(surface, color, (round(a[0] + o * 12), round(a[1] - 12)), 11, min(width, 3))
                label_at = (a[0] + o * 23 - (8 if o < 0 else 0), a[1] - 24)
            else:
                if w.learning and not w.active:
                    draw_dashed_line(surface, color, a, b, width, dash=5)
                else:
                    pygame.draw.line(surface, color, a, b, width)
                dx_w, dy_w = b[0] - a[0], b[1] - a[1]
                len_w = math.hypot(dx_w, dy_w) or 1.0
                ux_w, uy_w = dx_w / len_w, dy_w / len_w
                label_at = (a[0] + 0.76 * dx_w - uy_w * 7 - 4, a[1] + 0.76 * dy_w + ux_w * 7 - 7)
            sign = ("+" if w.sign > 0 else "-") + LEARNING_LABELS.get(w.learning, "")
            if not w.label:
                s_txt = self.small_font.render(sign, True, color)
                surface.blit(s_txt, (round(label_at[0]), round(label_at[1])))

        # Draw Parts (Hero inner fill + Shannon Joint Codebook Arc-Rings + Entropy halos).
        r_core = 11
        r_ring_in = 12.0
        r_ring_out = 16.0 if self.neural_view == "codebook" else 15.2
        part_labels_to_draw: list[tuple[float, float, str]] = []

        for p in diagram.parts:
            x, y = at(p.position)
            p_arcs = arcs_map.get(p.name, [])
            p_ent = float(ent_map.get(p.name, 0.0))
            fill = (tuple(int(c) for c in heat_colors(np.array(p.level)))
                    if p.shape is None and p.kind != "module" else None)

            if p.kind == "module":
                name = self.small_font.render(p.name, True, TEXT, PANEL_BG)
                surface.blit(name, (x - name.get_width() / 2, y - name.get_height() / 2))
                continue
            elif p.kind == "sensor":
                for dx in (-8, 8):
                    pygame.draw.line(surface, BODY_OUTLINE, (x, y), (x + dx, y - 19), 2)
                if p_arcs or p_ent > 0.02:
                    patch = self._get_shannon_ring_patch(p_arcs, p_ent, r_in=11.5, r_out=14.6)
                    surface.blit(patch, (round(x - patch.get_width() / 2), round(y - patch.get_height() / 2)))
                pygame.draw.circle(surface, fill, (round(x), round(y)), 11)
                pygame.draw.circle(surface, BODY_OUTLINE, (round(x), round(y)), 11, 1)
                below = 15.0
            elif p.shape is not None:
                self._draw_shannon_cells(surface, p, (x, y), GRID_WIDTH * radius, p_arcs, p_ent, snap)
                below = self.cells_height(p.level, GRID_WIDTH * radius) / 2 + 4.0
            elif p.kind == "control":
                patch = self._get_shannon_ring_patch(p_arcs, p_ent, r_in=r_ring_in + 1, r_out=r_ring_out + 1)
                surface.blit(patch, (round(x - patch.get_width() / 2), round(y - patch.get_height() / 2)))
                rc = r_core + 1
                diamond = [(x, y - rc), (x + rc, y), (x, y + rc), (x - rc, y)]
                pygame.draw.polygon(surface, fill, diamond)
                pygame.draw.polygon(surface, CONTROL_COLOR, diamond, 2)
                below = r_ring_out + 3.0
            elif p.kind == "motor":
                rect = pygame.Rect(0, 0, 32, 20)
                rect.center = (round(x), round(y))
                if p_ent > 0.02 and self.neural_view in ("composite", "entropy"):
                    ecol = lerp_color(PHASE_VIOLET, FLOW_ORBIT, min(p_ent, 1.0))
                    glow_m = pygame.Surface((42, 30), pygame.SRCALPHA)
                    pygame.draw.rect(glow_m, (*ecol, int(90 * min(p_ent, 1.0))), (0, 0, 42, 30), 2, border_radius=6)
                    surface.blit(glow_m, (rect.x - 5, rect.y - 5))
                pygame.draw.rect(surface, fill, rect, border_radius=4)
                if p_arcs:
                    _draw_segmented_bar(surface, rect.x + 2, rect.bottom - 4, rect.width - 4, 3, p_arcs)
                pygame.draw.rect(surface, BODY_OUTLINE, rect, 2, border_radius=4)
                below = 13.0
            else:
                patch = self._get_shannon_ring_patch(p_arcs, p_ent, r_in=r_ring_in, r_out=r_ring_out)
                surface.blit(patch, (round(x - patch.get_width() / 2), round(y - patch.get_height() / 2)))
                pygame.draw.circle(surface, fill, (round(x), round(y)), r_core)
                pygame.draw.circle(surface, BODY_OUTLINE, (round(x), round(y)), r_core, 1)
                if p.threshold is not None:
                    t_col = VEHICLE_OUTLINE if p.level > 0.5 else TEXT
                    text = f"{p.effective:.1f}" if p.effective is not None else f"{p.threshold:g}"
                    font_use = self.tiny_font if len(text) >= 3 else self.small_font
                    value = font_use.render(text, True, t_col)
                    surface.blit(value, (round(x - value.get_width() / 2), round(y - value.get_height() / 2)))
                below = r_ring_out + 2.0

            part_labels_to_draw.append((x, y + below, p.name))

        # Render all part labels on top with pairwise de-collision & dark halo pills!
        part_centers = [at(p.position) for p in diagram.parts if p.kind != "module"]
        lbl_items = []
        for idx_l, (lx, ly, name_str) in enumerate(part_labels_to_draw):
            lbl = self.tiny_font.render(name_str, True, PANEL_DIM)
            pw_l, ph_l = lbl.get_width() + 4, lbl.get_height()
            # If another part circle sits right below this label (e.g. symmetric -> fleeing),
            # shift label horizontally beside its own unit so it never covers the lower unit's threshold!
            ox_self, oy_self = part_centers[idx_l]
            for j_c, (ox_o, oy_o) in enumerate(part_centers):
                if j_c != idx_l and abs(ox_o - lx) < 13 and 16 <= (oy_o - oy_self) <= 32:
                    lx = ox_self - (pw_l / 2 + 16)
                    ly = oy_self - ph_l / 2
                    break
            lbl_items.append([lx, ly, pw_l, ph_l, lbl])

        # Iterative horizontal nudge for same-row neighbors (e.g. white_eye & red_eye on Wary Moth).
        for _ in range(3):
            for i in range(len(lbl_items)):
                for j in range(i + 1, len(lbl_items)):
                    xi, yi, wi, hi, _ = lbl_items[i]
                    xj, yj, wj, hj, _ = lbl_items[j]
                    if abs(yi - yj) < max(hi, hj) - 1:
                        min_sep = 0.5 * (wi + wj) + 3.0
                        dx_ij = xj - xi
                        if abs(dx_ij) < min_sep:
                            push = 0.5 * (min_sep - abs(dx_ij))
                            sgn = 1.0 if dx_ij >= 0 else -1.0
                            lbl_items[i][0] -= sgn * push
                            lbl_items[j][0] += sgn * push

        for lx, ly, pw_l, ph_l, lbl in lbl_items:
            px_l = round(lx - pw_l / 2)
            py_l = round(ly)
            bg_pill = pygame.Surface((pw_l, ph_l), pygame.SRCALPHA)
            pygame.draw.rect(bg_pill, (22, 24, 34, 215), (0, 0, pw_l, ph_l), border_radius=3)
            bg_pill.blit(lbl, (2, 0))
            surface.blit(bg_pill, (px_l, py_l))

    def _draw_shannon_cells(
        self,
        surface: pygame.Surface,
        p,
        at_pos: tuple[float, float],
        width: float,
        p_arcs: list[tuple[int, float]],
        p_ent: float,
        snap: dict,
    ) -> None:
        """Draw camera/array cells blending Hero state with swarm firing probability / cell entropy."""
        rows = self.rows_of(p.level)
        n_r, n_c = len(rows), len(rows[0])
        cell = min(width / n_c, 8.0)
        gw_px, gh_px = cell * n_c, cell * n_r
        x0 = at_pos[0] - gw_px / 2
        y0 = at_pos[1] - gh_px / 2

        grid_prob = _lookup_part_grid(snap, "grid_prob", p.name, (n_r, n_c))
        grid_ent = _lookup_part_grid(snap, "grid_entropy", p.name, (n_r, n_c))

        for r, row in enumerate(rows):
            for c, hero_lvl in enumerate(row):
                if self.neural_view == "entropy" and grid_ent is not None:
                    e_cell = min(max(float(grid_ent[r, c]), 0.0), 1.0)
                    h_col = tuple(int(v) for v in heat_colors(np.array(hero_lvl)))
                    e_col = lerp_color((14, 18, 42), PHASE_VIOLET if e_cell < 0.65 else FLOW_ORBIT, e_cell)
                    color = lerp_color(e_col, h_col, 0.45 if hero_lvl > 0 else 0.0)
                else:
                    eff = float(hero_lvl)
                    if grid_prob is not None and eff < float(grid_prob[r, c]):
                        eff = max(eff, 0.42 * float(grid_prob[r, c]))
                    color = tuple(int(v) for v in heat_colors(np.array(min(max(eff, 0.0), 1.0))))
                pygame.draw.rect(
                    surface, color,
                    (x0 + c * cell, y0 + r * cell, math.ceil(cell), math.ceil(cell))
                )
        border_col = (
            lerp_color(BODY_OUTLINE, FLOW_ORBIT, min(p_ent, 1.0))
            if self.neural_view in ("composite", "entropy") and p_ent > 0.05
            else BODY_OUTLINE
        )
        pygame.draw.rect(surface, border_col, (x0, y0, gw_px, gh_px), 1)
        if p_arcs:
            _draw_segmented_bar(surface, round(x0), round(y0 - 4), max(8, round(gw_px)), 3, p_arcs)

    def _draw_shannon_strip(
        self,
        surface: pygame.Surface,
        x0: int,
        y0: int,
        width: int,
        height: int,
        focal_vehicles: list,
        hero_idx: int,
        snap: dict,
    ) -> None:
        """Compact Shannon Telemetry Header + Lossless Sorted N x M Swarm Microstate Bit-Raster."""
        card = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(card, (13, 17, 32, 245), (0, 0, width, height), border_radius=6)
        pygame.draw.rect(card, (62, 84, 134, 190), (0, 0, width, height), 1, border_radius=6)
        pygame.draw.line(card, (48, 66, 110, 150), (4, 18), (width - 4, 18), 1)

        h_w = float(snap.get("joint_entropy_bits", 0.0))
        h_sum = float(snap.get("marginal_entropy_sum_bits", 0.0))
        tc = float(snap.get("total_correlation_bits", 0.0))
        mi_xw = float(snap.get("spatial_mi_bits", 0.0))

        # Top 18px telemetry strip
        tx = 7
        t1 = self.tiny_font.render(f"H(W):{h_w:.2f}b", True, FLOW_ATTRACT)
        card.blit(t1, (tx, 4))
        tx += t1.get_width() + 9
        if self.neural_view == "entropy":
            ts = self.tiny_font.render(f"ΣH:{h_sum:.2f}b", True, TEXT)
            card.blit(ts, (tx, 4))
            tx += ts.get_width() + 9
        t2 = self.tiny_font.render(f"TC:{tc:.2f}b", True, PHASE_VIOLET)
        card.blit(t2, (tx, 4))
        tx += t2.get_width() + 9
        t3 = self.tiny_font.render(f"I(X;W):{mi_xw:.2f}b", True, FLOW_ORBIT)
        card.blit(t3, (tx, 4))

        mode_tag = self.tiny_font.render(f"[{self.neural_view}]", True, PHASE_EMERALD)
        if tx + t3.get_width() + 6 < width - mode_tag.get_width() - 6:
            card.blit(mode_tag, (width - mode_tag.get_width() - 7, 4))

        ry0 = 21
        raster = np.asarray(snap.get("raster_matrix", np.zeros((len(focal_vehicles), 1))), dtype=float)
        if raster.ndim == 1:
            raster = raster[:, None]
        n_veh, n_units = raster.shape
        r_cw = np.asarray(snap.get("raster_codeword_ids", np.zeros(n_veh, dtype=int)), dtype=int)
        hero_row = int(snap.get("hero_sorted_row", 0))
        unit_names = list(snap.get("raster_unit_names", []))

        # In "codebook" view, show a horizontal Joint Codebook Empirical Mass Bar above the raster!
        if self.neural_view == "codebook" and n_veh > 0:
            uniq_c, counts_c = np.unique(r_cw, return_counts=True)
            order_c = np.argsort(-counts_c)
            cw_arcs = [(int(uniq_c[i]), float(counts_c[i]) / n_veh) for i in order_c]
            _draw_segmented_bar(card, 44, ry0, width - 52, 5, cw_arcs)
            ry0 += 8

        bottom_pad = 13 if (self.neural_view == "raster" and unit_names) else 4
        rx0 = 44
        rw = max(40, width - rx0 - 7)
        rh = max(12, height - ry0 - bottom_pad)

        # Left margin metadata + Hero row pointer
        dim_lbl = self.tiny_font.render(f"{n_veh}×{n_units}", True, PANEL_DIM)
        card.blit(dim_lbl, (5, ry0))

        if n_veh > 0 and n_units > 0:
            for r in range(n_veh):
                yr0 = ry0 + round(r * rh / n_veh)
                yr1 = max(yr0 + 1, ry0 + round((r + 1) * rh / n_veh))
                cw_col = codeword_color(int(r_cw[r])) if r < len(r_cw) else CODEWORD_RESIDUAL
                # Codeword cluster ribbon at rx0 - 7
                pygame.draw.rect(card, cw_col, (rx0 - 7, yr0, 5, yr1 - yr0))
                for c in range(n_units):
                    xc0 = rx0 + round(c * rw / n_units)
                    xc1 = max(xc0 + 1, rx0 + round((c + 1) * rw / n_units))
                    cw_cell = xc1 - xc0
                    gap_x = 1 if cw_cell >= 5 else 0
                    bit_on = raster[r, c] > 0.5
                    cell_col = cw_col if bit_on else (22, 27, 46)
                    pygame.draw.rect(card, cell_col, (xc0, yr0, max(1, cw_cell - gap_x), yr1 - yr0))

            # Highlight Hero's sorted row cleanly with signature cyan ring & vector pointer!
            if 0 <= hero_row < n_veh:
                yh0 = ry0 + round(hero_row * rh / n_veh)
                yh1 = max(yh0 + 2, ry0 + round((hero_row + 1) * rh / n_veh))
                pygame.draw.rect(card, SELECTED_RING, (rx0 - 8, yh0 - 1, rw + 9, yh1 - yh0 + 2), 1)
                h_tag = self.tiny_font.render(f"#{self.selected}", True, SELECTED_RING)
                hy_txt = max(ry0 + 10, min(ry0 + rh - h_tag.get_height(), (yh0 + yh1 - h_tag.get_height()) // 2))
                ym_tri = hy_txt + h_tag.get_height() // 2
                pygame.draw.polygon(
                    card, SELECTED_RING, [(4, ym_tri - 3), (9, ym_tri), (4, ym_tri + 3)]
                )
                card.blit(h_tag, (11, hy_txt))

        if self.neural_view == "raster" and unit_names and n_units > 0:
            step_c = max(1, math.ceil(n_units / 9))
            for c in range(0, n_units, step_c):
                xc_mid = rx0 + round((c + 0.5) * rw / n_units)
                short = unit_names[c][:3]
                ut = self.tiny_font.render(short, True, PANEL_DIM)
                card.blit(ut, (max(rx0, min(rx0 + rw - ut.get_width(), xc_mid - ut.get_width() // 2)),
                               ry0 + rh + 1))

        surface.blit(card, (x0, y0))

    def _draw_shannon_readouts(
        self,
        surface: pygame.Surface,
        diagram: Diagram,
        x0: int,
        y: int,
        width: int,
        row_h: int,
        snap: dict,
    ) -> int:
        """1D Empirical Density Barcode strips over [0, 1] + Hero vertical cursor tick."""
        bar_x = x0 + 196
        bar_width = width - 196 - 50
        bar_h = max(6, min(8, row_h - 7))
        barcodes: dict = snap.get("barcodes", {}) or {}
        max_text_w = bar_x - x0 - 6

        for p in diagram.parts:
            name_surf = self.small_font.render(p.name, True, TEXT)
            surface.blit(name_surf, (x0, y + (row_h - name_surf.get_height()) // 2))
            rem_w = max_text_w - name_surf.get_width() - 5
            if rem_w >= 24:
                label = p.label if p.effective is None else f"thr {p.effective:.1f}"
                lbl_surf = self.tiny_font.render(label, True, PANEL_DIM)
                if lbl_surf.get_width() > rem_w:
                    lbl_surf = lbl_surf.subsurface((0, 0, rem_w, lbl_surf.get_height()))
                surface.blit(lbl_surf, (x0 + name_surf.get_width() + 5,
                                        y + (row_h - lbl_surf.get_height()) // 2 + 1))

            bc = barcodes.get(p.name)
            by = y + (row_h - bar_h) // 2
            if p.kind == "module":
                lit = sum(1 for lv in p.level if lv > 0)
                hero_frac = lit / max(len(p.level), 1)
                val_str = f"{lit}/{len(p.level)}"
            elif p.shape is not None:
                rows = self.rows_of(p.level)
                total_cells = max(len(rows) * len(rows[0]), 1)
                lit = sum(1 for row in rows for lv in row if lv > 0)
                hero_frac = lit / total_cells
                val_str = f"{lit}/{total_cells}"
            else:
                hero_frac = float(p.level)
                val_str = f"{p.value:+.2f}" if p.kind == "control" else f"{p.value:.2f}"

            self._draw_barcode_strip(
                surface, bar_x, by, bar_width, bar_h, bc, hero_frac, is_learning=False
            )
            v_surf = self.small_font.render(val_str, True, TEXT)
            surface.blit(v_surf, (x0 + width - v_surf.get_width(),
                                  y + (row_h - v_surf.get_height()) // 2))
            y += row_h

        for w in diagram.wires:
            if not w.learning:
                continue
            _, on = learning_colors(w)
            w_key = f"{w.source}->{w.target}"
            name_surf = self.small_font.render(w_key, True, TEXT)
            if name_surf.get_width() > max_text_w - 18:
                name_surf = name_surf.subsurface((0, 0, max_text_w - 18, name_surf.get_height()))
            surface.blit(name_surf, (x0, y + (row_h - name_surf.get_height()) // 2))
            letter = LEARNING_LABELS[w.learning] + ("•" if w.waiting else "")
            l_surf = self.tiny_font.render(letter, True, on)
            surface.blit(l_surf, (x0 + name_surf.get_width() + 4,
                                  y + (row_h - l_surf.get_height()) // 2 + 1))

            bc = barcodes.get(w_key) or barcodes.get((w.source, w.target))
            by = y + (row_h - bar_h) // 2
            self._draw_barcode_strip(
                surface, bar_x, by, bar_width, bar_h, bc, float(w.strength), is_learning=True
            )
            v_surf = self.small_font.render(f"{w.strength:.2f}", True, TEXT)
            surface.blit(v_surf, (x0 + width - v_surf.get_width(),
                                  y + (row_h - v_surf.get_height()) // 2))
            y += row_h

        return y

    def _draw_barcode_strip(
        self,
        surface: pygame.Surface,
        bx: int,
        by: int,
        bw: int,
        bh: int,
        bc: dict | None,
        fallback_hero_norm: float,
        *,
        is_learning: bool = False,
    ) -> None:
        """Render a 1D Empirical Density Barcode over [0, 1] + Hero vertical cursor tick."""
        pygame.draw.rect(surface, (11, 14, 28), (bx, by, bw, bh), border_radius=3)
        hist = None
        hero_norm = fallback_hero_norm
        if bc is not None and "hist" in bc:
            hist = np.asarray(bc["hist"], dtype=float).ravel()
            if "hero_val" in bc:
                hero_norm = float(bc["hero_val"])

        if hist is not None and hist.size > 0:
            n_bins = hist.size
            h_max = max(float(hist.max()), 1e-9)
            inner_w = bw - 2
            for b_i in range(n_bins):
                mass = float(hist[b_i])
                if mass <= 1e-6:
                    continue
                x_b0 = bx + 1 + round(b_i * inner_w / n_bins)
                x_b1 = max(x_b0 + 1, bx + 1 + round((b_i + 1) * inner_w / n_bins))
                inten = (mass / h_max) ** 0.52
                bin_center = (b_i + 0.5) / n_bins
                if is_learning:
                    base_c = PHASE_EMERALD if bin_center > STRENGTH_TICK else FLOW_ATTRACT
                    col = lerp_color((18, 36, 48), base_c, 0.25 + 0.75 * inten)
                elif self.neural_view == "entropy":
                    col = lerp_color((24, 22, 54), PHASE_VIOLET if inten < 0.7 else FLOW_ORBIT, inten)
                else:
                    mid_c = lerp_color((24, 78, 152), FLOW_ATTRACT, min(inten * 1.35, 1.0))
                    col = lerp_color(mid_c, FLOW_ORBIT, max(0.0, (inten - 0.55) / 0.45))
                pygame.draw.rect(surface, col, (x_b0, by + 1, x_b1 - x_b0, max(1, bh - 2)))
        else:
            filled = round((bw - 2) * min(max(fallback_hero_norm, 0.0), 1.0))
            if filled > 0:
                col = tuple(int(c) for c in heat_colors(np.array(min(max(fallback_hero_norm, 0.0), 1.0))))
                pygame.draw.rect(surface, col, (bx + 1, by + 1, filled, max(1, bh - 2)))

        pygame.draw.rect(surface, (62, 76, 114), (bx, by, bw, bh), 1, border_radius=3)

        if is_learning:
            tick_x = bx + 1 + round((bw - 3) * STRENGTH_TICK)
            pygame.draw.line(surface, FLOW_ORBIT, (tick_x, by - 2), (tick_x, by + bh + 1), 2)

        # Crisp bright Hero cursor tick!
        hx_tick = bx + 1 + round((bw - 3) * min(max(hero_norm, 0.0), 1.0))
        pygame.draw.line(surface, (8, 10, 20), (hx_tick, by - 2), (hx_tick, by + bh + 1), 4)
        pygame.draw.line(surface, SELECTED_RING, (hx_tick, by - 2), (hx_tick, by + bh + 1), 2)

    def _draw_shannon_params(self, surface: pygame.Surface, diagram: Diagram, x0: int, y: int) -> None:
        items = [f"{name} {value}" for name, value in diagram.params.items()]
        for line in ("   ".join(items[:2]), "   ".join(items[2:])):
            surface.blit(self.small_font.render(line, True, PANEL_DIM), (x0, y + 1))
            y += 15

    def draw_diagram(self, surface: pygame.Surface, diagram: Diagram,
                     center: tuple[int, int], radius: float) -> None:
        def at(position):  # body coordinates (+y forward) to screen
            return (center[0] + position[0] * radius * diagram.body[0],
                    center[1] - position[1] * radius * diagram.body[1])

        bw, bl = diagram.body   # the body spans [-1, 1] on both axes, drawn with its proportions
        body = pygame.Rect(0, 0, 2 * radius * bw, 2 * radius * bl)
        body.center = center
        pygame.draw.rect(surface, BODY_OUTLINE, body, 2, border_radius=int(radius * min(bw, bl) * 0.3))

        parts = {p.name: p for p in diagram.parts}
        # Modules' boxes go under the wires, so that a wire is seen leaving each of its ports.
        for p in diagram.parts:
            if p.kind == "module":
                self.draw_module(surface, p, at(p.position))
        for w in bundles(diagram.wires):
            a, b = wire_ends(w, parts, at, PANEL_PORT_GAP, PANEL_BOX_HEIGHT)
            if w.control:   # a threshold control's arrow: dashed, it carries no signal
                draw_dashed_line(surface, CONTROL_COLOR, a, b, 2)
                continue
            if w.learning:  # wider as it grows stronger, dashed until it is active
                dim, on = learning_colors(w)
                color = lerp_color(dim, on, w.level) if w.active else dim
                width = 1 + round(4 * w.strength)
            else:
                color = lerp_color(WIRE_OFF, WIRE_ON, w.level) if w.active else WIRE_INACTIVE
                width = 4 if w.active else 1
            if w.source == w.target:  # a unit feeding back into itself, toward the outside
                o = -1 if parts[w.target].position[0] < 0 else 1
                pygame.draw.circle(surface, color, (a[0] + o * 12, a[1] - 12), 12, min(width, 3))
                label_at = (a[0] + o * 26 - (12 if o < 0 else 0), a[1] - 30)
            else:
                if w.learning and not w.active:
                    draw_dashed_line(surface, color, a, b, width)
                else:
                    pygame.draw.line(surface, color, a, b, width)
                # Near the target, so crossed wires do not label the same spot.
                label_at = (a[0] + 0.85 * (b[0] - a[0]) + 8, a[1] + 0.85 * (b[1] - a[1]) - 22)
            sign = ("+" if w.sign > 0 else "-") + LEARNING_LABELS.get(w.learning, "")
            if not w.label:   # bundles of an array's wires would crowd the diagram with signs
                surface.blit(self.title_font.render(sign, True, color), label_at)

        for p in diagram.parts:
            x, y = at(p.position)
            fill = (tuple(int(c) for c in heat_colors(np.array(p.level)))
                    if p.shape is None and p.kind != "module" else None)
            if p.kind == "module":
                # Its box is drawn under the wires; its name goes over them, in the middle. Its
                # ports are too small here to be named.
                name = self.small_font.render(p.name, True, TEXT, PANEL_BG)
                surface.blit(name, (x - name.get_width() / 2, y - name.get_height() / 2))
            elif p.kind == "sensor":
                # A round sensor with two short antennae pointing forward.
                for dx in (-9, 9):
                    pygame.draw.line(surface, BODY_OUTLINE, (x, y), (x + dx, y - 22), 2)
                pygame.draw.circle(surface, fill, (x, y), 13)
                pygame.draw.circle(surface, BODY_OUTLINE, (x, y), 13, 2)
            elif p.shape is not None:
                self.draw_cells(surface, p.level, (x, y), GRID_WIDTH * radius, centered=True)
            elif p.kind == "control":
                # A diamond, lit by the activity it sees (half lit at its target).
                r = UNIT_RADIUS + 2
                diamond = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
                pygame.draw.polygon(surface, fill, diamond)
                pygame.draw.polygon(surface, CONTROL_COLOR, diamond, 2)
            elif p.kind == "motor":
                rect = pygame.Rect(0, 0, 34, 22)
                rect.center = (x, y)
                pygame.draw.rect(surface, fill, rect, border_radius=4)
                pygame.draw.rect(surface, BODY_OUTLINE, rect, 2, border_radius=4)
            else:
                pygame.draw.circle(surface, fill, (x, y), UNIT_RADIUS)
                pygame.draw.circle(surface, BODY_OUTLINE, (x, y), UNIT_RADIUS, 2)
                if p.threshold is not None:
                    # The threshold inside the circle, dark on a lit unit and light on a dark one;
                    # a controlled unit's threshold now, with one decimal.
                    color = VEHICLE_OUTLINE if p.level > 0.5 else TEXT
                    text = f"{p.effective:.1f}" if p.effective is not None else f"{p.threshold:g}"
                    value = self.small_font.render(text, True, color)
                    surface.blit(value, (x - value.get_width() / 2, y - value.get_height() / 2))
            if p.kind == "module":
                continue   # its name is inside its box
            label = self.small_font.render(p.name, True, PANEL_DIM)
            below = (UNIT_RADIUS + 4 if p.kind in ("unit", "control")
                     else self.cells_height(p.level, GRID_WIDTH * radius) / 2 + 3
                     if p.shape is not None else 15)
            surface.blit(label, (x - label.get_width() / 2, y + below))

    def draw_module(self, surface: pygame.Surface, p, at) -> None:
        """A box with its inputs on top and its outputs below; each output lit by its level."""
        w, h = module_box(p, PANEL_PORT_GAP, PANEL_BOX_HEIGHT)
        box = pygame.Rect(0, 0, w, h)
        box.center = at
        pygame.draw.rect(surface, PANEL_BG, box, border_radius=5)
        pygame.draw.rect(surface, BODY_OUTLINE, box, 2, border_radius=5)
        levels = dict(zip(p.outputs, p.level))
        for port in p.inputs + p.outputs:
            dx, dy = port_offset(p, port, PANEL_PORT_GAP, PANEL_BOX_HEIGHT)
            x, y = at[0] + dx, at[1] + dy
            fill = tuple(int(c) for c in heat_colors(np.array(levels.get(port, 0.0))))
            pygame.draw.circle(surface, fill, (x, y), 3)
            pygame.draw.circle(surface, BODY_OUTLINE, (x, y), 3, 1)

    @staticmethod
    def rows_of(levels) -> list[list[float]]:
        return levels if levels and isinstance(levels[0], list) else [levels]

    def cells_height(self, levels, width: float) -> float:
        rows = self.rows_of(levels)
        return len(rows) * min(width / len(rows[0]), 8)

    def draw_cells(self, surface: pygame.Surface, levels, at, width: float,
                   centered: bool = False) -> None:
        """A camera's pixels or an array's elements, as a grid of cells lit by their levels."""
        rows = self.rows_of(levels)
        cell = min(width / len(rows[0]), 8)
        x0, y0 = at
        if centered:
            x0, y0 = x0 - cell * len(rows[0]) / 2, y0 - cell * len(rows) / 2
        for r, row in enumerate(rows):
            for c, level in enumerate(row):
                color = tuple(int(v) for v in heat_colors(np.array(level)))
                pygame.draw.rect(surface, color, (x0 + c * cell, y0 + r * cell,
                                                  math.ceil(cell), math.ceil(cell)))
        pygame.draw.rect(surface, BODY_OUTLINE, (x0, y0, cell * len(rows[0]), cell * len(rows)), 1)

    def draw_readouts(self, surface: pygame.Surface, diagram: Diagram, x0: int, y: int) -> int:
        """One line per part: name and label, a level bar, and the value."""
        width = PANEL_WIDTH - 32
        bar_x, bar_width = x0 + 200, width - 200 - 52
        for p in diagram.parts:
            name = self.font.render(p.name, True, TEXT)
            surface.blit(name, (x0, y + 2))
            label = p.label if p.effective is None else f"threshold {p.effective:.1f}"
            surface.blit(self.small_font.render(label, True, PANEL_DIM),
                         (x0 + name.get_width() + 6, y + 4))
            if p.kind == "module":
                # Its outputs, as cells, and how many are on.
                self.draw_cells(surface, p.level, (bar_x, y + 4), min(bar_width, 12 * len(p.level)))
                lit = sum(1 for level in p.level if level > 0)
                value = self.font.render(f"{lit}/{len(p.level)}", True, TEXT)
                surface.blit(value, (x0 + width - value.get_width(), y + 2))
                y += ROW
                continue
            if p.shape is not None:
                # The image, or the array's elements, instead of a bar; and how many are on.
                rows = self.rows_of(p.level)
                cells_width = min(bar_width, (ROW - 4) / len(rows) * len(rows[0]))
                top = y + (ROW - self.cells_height(p.level, cells_width)) / 2
                self.draw_cells(surface, p.level, (bar_x, top), cells_width)
                lit = sum(1 for row in rows for level in row if level > 0)
                value = self.font.render(f"{lit}/{len(rows) * len(rows[0])}", True, TEXT)
                surface.blit(value, (x0 + width - value.get_width(), y + 2))
                y += ROW
                continue
            pygame.draw.rect(surface, PANEL_BORDER, (bar_x, y + 7, bar_width, 6), border_radius=3)
            filled = round(bar_width * p.level)
            if filled:
                color = tuple(int(c) for c in heat_colors(np.array(p.level)))
                pygame.draw.rect(surface, color, (bar_x, y + 7, filled, 6), border_radius=3)
            value = self.font.render(f"{p.value:+.2f}" if p.kind == "control" else f"{p.value:.2f}",
                                     True, TEXT)
            surface.blit(value, (x0 + width - value.get_width(), y + 2))
            y += ROW
        # Then each learning connection: its strength, with a tick where it becomes active.
        for w in diagram.wires:
            if not w.learning:
                continue
            dim, on = learning_colors(w)
            name = self.font.render(f"{w.source}->{w.target}", True, TEXT)
            surface.blit(name, (x0, y + 2))
            # An Ergotrix wire waiting for its target shows a dot after its letter.
            letter = LEARNING_LABELS[w.learning] + (" •" if w.waiting else "")
            surface.blit(self.small_font.render(letter, True, on), (x0 + name.get_width() + 6, y + 4))
            pygame.draw.rect(surface, PANEL_BORDER, (bar_x, y + 7, bar_width, 6), border_radius=3)
            filled = round(bar_width * w.strength)
            if filled:
                pygame.draw.rect(surface, on if w.active else dim,
                                 (bar_x, y + 7, filled, 6), border_radius=3)
            tick = bar_x + round(bar_width * STRENGTH_TICK)
            pygame.draw.line(surface, TEXT, (tick, y + 4), (tick, y + 15), 1)
            value = self.font.render(f"{w.strength:.2f}", True, TEXT)
            surface.blit(value, (x0 + width - value.get_width(), y + 2))
            y += ROW
        return y

    def draw_params(self, surface: pygame.Surface, diagram: Diagram, x0: int, y: int) -> None:
        items = [f"{name} {value}" for name, value in diagram.params.items()]
        for line in ("   ".join(items[:2]), "   ".join(items[2:])):
            surface.blit(self.font.render(line, True, PANEL_DIM), (x0, y + 2))
            y += ROW


# --- mathematical & rendering helpers for the live dynamics overlay ---


def _wrap_pi(angle):
    return wrap_pi(angle)


def _polar_coords(x, y, heading, source_pos, arena=None):
    return polar_coords(x, y, heading, source_pos, arena=arena)


def _is_slow_or_latched_unit(u) -> bool:
    return (
        u.is_control
        or u.delay > 0.15
        or any(c.source == u.name for c in u.inputs)
    )


def _eval_kinematics_batch(v, world, sim_time: float, qx, qy, qtheta) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized deterministic (speed, turn_rate) for `v` across poses `(qx, qy, qtheta)`."""
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qtheta = np.asarray(qtheta, dtype=float)
    m = qx.size
    sig: dict[str, np.ndarray] = {}

    c_th, s_th = np.cos(qtheta), np.sin(qtheta)
    for sensor in v.definition.sensors:
        if sensor.camera:
            for px_name in v.pixel_names[sensor.name]:
                sig[px_name] = np.full(m, v.sensor_values.get(px_name, 0.0))
            continue
        if sensor.type == ENERGY_SENSOR:
            raw = np.full(m, v.energy if v.energy is not None else 0.0)
        elif sensor.type in world.stimuli:
            bx = sensor.position[0] * v.half_width
            by = sensor.position[1] * v.half_length
            sx = qx + by * c_th + bx * s_th
            sy = qy + by * s_th - bx * c_th
            if sensor.field_of_view >= 360.0:
                raw = np.asarray(world.stimulus_at(sensor.type, sx, sy, sim_time), dtype=float)
            else:
                stim = world.stimuli[sensor.type]
                raw = np.zeros(m)
                look = qtheta + math.radians(sensor.direction)
                half_fov = sensor.field_of_view * 0.5
                for src in stim.sources:
                    if not src.active(sim_time):
                        continue
                    dx, dy = world.arena.delta(sx - src.position[0], sy - src.position[1])
                    off = _wrap_pi(np.arctan2(-dy, -dx) - look)
                    in_view = (np.degrees(np.abs(off)) <= half_fov) | ((dx == 0) & (dy == 0))
                    val = src.intensity / (1.0 + (dx * dx + dy * dy) / (src.falloff * src.falloff))
                    raw = raw + np.where(in_view, val, 0.0)
            if sensor.habituation:
                ref = v.habituation_refs.get(sensor.name)
                raw = np.zeros(m) if ref is None else (raw - ref)
        else:
            raw = np.zeros(m)
        sig[sensor.name] = sensor.response.outputs(raw)

    units = v.definition.units
    if units:
        comb_indices = []
        for k, u in enumerate(units):
            sig[u.name] = np.full(m, v.unit_outputs.get(u.name, 0.0))
            if not _is_slow_or_latched_unit(u):
                comb_indices.append((k, u))
        if comb_indices and len(comb_indices) <= 40:
            for _ in range(4):
                for k, u in comb_indices:
                    total = np.zeros(m)
                    for j, conn in enumerate(u.inputs):
                        if v.connection_strength(u.name, j) > ACTIVE_STRENGTH:
                            total = total + conn.sign * sig[conn.source]
                    sig[u.name] = (total >= float(v.effective_thresholds[k])).astype(float)

    actuators = v.definition.actuators
    motor_speeds = []
    for a in actuators:
        total = np.zeros(m)
        for j, conn in enumerate(a.inputs):
            if v.connection_strength(a.name, j) > ACTIVE_STRENGTH:
                total = total + conn.sign * sig[conn.source]
        motor_speeds.append(np.clip(a.base_speed + a.gain * total, 0.0, a.max_speed))

    speeds_arr = np.stack(motor_speeds, axis=0)  # (n_act, m)
    v_out = speeds_arr.mean(axis=0)
    xs = np.array([a.position[0] * v.half_width for a in actuators], dtype=float)
    sxx = float(np.sum(xs * xs))
    omega_out = (xs[:, None] * speeds_arr).sum(axis=0) / sxx if sxx > 0 else np.zeros(m)
    return v_out, omega_out


def _select_stable_bearing(
    psi_k: np.ndarray, dpsi_k: np.ndarray, v_k: np.ndarray, pref_sign: float,
    steer_torque: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the stable slow-manifold bearing `psi*` (`dpsi = 0`, `d(dpsi)/dpsi < 0`)."""
    lead_shape = dpsi_k.shape[:-1]
    n_ang = psi_k.size
    dpsi_flat = dpsi_k.reshape(-1, n_ang)
    v_flat = v_k.reshape(-1, n_ang)
    st_flat = steer_torque.reshape(-1) if steer_torque is not None else np.zeros(dpsi_flat.shape[0])
    dstep = float(psi_k[1] - psi_k[0])

    psi_out = np.zeros(dpsi_flat.shape[0])
    v_out = np.zeros(dpsi_flat.shape[0])
    nxt = (np.arange(n_ang) + 1) % n_ang

    for row_i in range(dpsi_flat.shape[0]):
        dp = dpsi_flat[row_i]
        vv = v_flat[row_i]
        st = float(st_flat[row_i])
        roots = []
        for m in range(n_ang):
            a, b = float(dp[m]), float(dp[nxt[m]])
            if a >= 0.0 and b <= 0.0 and (a > 0.0 or b < 0.0):
                frac = a / (a - b) if (a - b) != 0.0 else 0.0
                # Handle wrap across +pi -> -pi at m == n_ang - 1 cleanly:
                p_base = psi_k[m] + frac * dstep
                p_root = float(_wrap_pi(p_base))
                v_root = float(vv[m] + frac * (vv[nxt[m]] - vv[m]))
                slope = b - a  # more negative = stronger angular restoring torque
                roots.append((p_root, v_root, slope))
        # When motor steering torque is attractive (st > 0) and only the kinematic flyby root
        # near +/-pi survived because v/r > omega_max close to the lamp, keep head-on attraction (psi*=0).
        if st > 1e-4 and roots and all(abs(r_item[0]) > 2.2 for r_item in roots):
            mid = n_ang // 2
            psi_out[row_i] = 0.0
            v_out[row_i] = 0.5 * float(vv[mid - 1] + vv[mid])
        elif not roots:
            idx_min = int(np.argmin(np.abs(dp)))
            psi_out[row_i] = float(psi_k[idx_min])
            v_out[row_i] = float(vv[idx_min])
        elif len(roots) == 1:
            psi_out[row_i], v_out[row_i], _ = roots[0]
        else:
            # Prefer the branch matching current orbit spin handedness.
            roots.sort(key=lambda item: (0 if item[0] * pref_sign >= -0.05 else 1, item[2]))
            psi_out[row_i], v_out[row_i], _ = roots[0]

    return psi_out.reshape(lead_shape), v_out.reshape(lead_shape)


def _zero_contours(
    field_2d: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract linear-interpolated zero-crossing segments of `field_2d[j, i]` via Marching Squares."""
    ny, nx = field_2d.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for j in range(ny - 1):
        y0, y1 = float(ys[j]), float(ys[j + 1])
        for i in range(nx - 1):
            x0, x1 = float(xs[i]), float(xs[i + 1])
            f00 = float(field_2d[j, i])
            f10 = float(field_2d[j, i + 1])
            f11 = float(field_2d[j + 1, i + 1])
            f01 = float(field_2d[j + 1, i])
            pts = []
            if (f00 >= 0.0) != (f10 >= 0.0):
                t = f00 / (f00 - f10)
                pts.append((x0 + t * (x1 - x0), y0))
            if (f10 >= 0.0) != (f11 >= 0.0):
                t = f10 / (f10 - f11)
                pts.append((x1, y0 + t * (y1 - y0)))
            if (f11 >= 0.0) != (f01 >= 0.0):
                t = f01 / (f01 - f11)
                pts.append((x0 + t * (x1 - x0), y1))
            if (f01 >= 0.0) != (f00 >= 0.0):
                t = f00 / (f00 - f01)
                pts.append((x0, y0 + t * (y1 - y0)))
            if len(pts) == 2:
                segments.append((pts[0], pts[1]))
            elif len(pts) == 4:
                segments.append((pts[0], pts[1]))
                segments.append((pts[2], pts[3]))
    return segments


def _ring_order_parameter(v) -> tuple[float, float, float, float, list]:
    """Kuramoto/Strogatz circular ring order parameter Z(t) = (1/N) sum a_k exp(i 2 pi k / N)."""
    ideas = [u for u in v.definition.units if u.name.startswith("idea_")]
    if ideas:
        ideas.sort(key=lambda u: int(u.name.split("_")[1]) if u.name.split("_")[1].isdigit() else 0)
        ring = ideas
    else:
        ring = [u for u in v.definition.units
                if not u.is_control and u.name not in ("wake", "awake")]
    if not ring:
        return 0.0, 0.0, 0.0, 0.0, []
    acts = np.array([float(v.unit_outputs.get(u.name, 0.0)) for u in ring])
    info = compute_ring_order_parameter(acts)
    z = info["Z"]
    return float(z.real), float(z.imag), float(info["R"]), float(info["total_activity"]), ring


def _draw_glow_chevron(
    surface: pygame.Surface, cx: float, cy: float, ux: float, uy: float,
    length: float, color: tuple[int, int, int], alpha: int
) -> None:
    half = 0.5 * length
    tail = (cx - ux * half, cy - uy * half)
    head = (cx + ux * half, cy + uy * half)
    wing_len = max(3.5, 0.38 * length)
    px, py = -uy, ux
    w_left = (head[0] - ux * wing_len + px * wing_len * 0.58,
              head[1] - uy * wing_len + py * wing_len * 0.58)
    w_right = (head[0] - ux * wing_len - px * wing_len * 0.58,
               head[1] - uy * wing_len - py * wing_len * 0.58)
    halo_a = max(18, alpha // 4)
    pygame.draw.line(surface, (*color, halo_a), tail, head, 4)
    pygame.draw.aaline(surface, (*color, alpha), tail, head)
    pygame.draw.aaline(surface, (*color, alpha), w_left, head)
    pygame.draw.aaline(surface, (*color, alpha), w_right, head)


def _draw_glow_dashed_circle(
    surface: pygame.Surface, center: tuple[float, float], radius_px: float,
    color: tuple[int, int, int], dash_deg: int = 8, spin_dir: int = 0
) -> None:
    cx, cy = center
    pygame.draw.circle(surface, (*color, 36), (round(cx), round(cy)), round(radius_px), 5)
    step = max(4, dash_deg)
    for deg in range(0, 360, 2 * step):
        a0 = math.radians(deg)
        a1 = math.radians(deg + step)
        p0 = (cx + radius_px * math.cos(a0), cy - radius_px * math.sin(a0))
        p1 = (cx + radius_px * math.cos(a1), cy - radius_px * math.sin(a1))
        pygame.draw.aaline(surface, (*color, 215), p0, p1)
    if spin_dir != 0:
        for deg in (45, 135, 225, 315):
            ang = math.radians(deg)
            px = cx + radius_px * math.cos(ang)
            py = cy - radius_px * math.sin(ang)
            tx = -spin_dir * math.sin(ang)
            ty = -spin_dir * math.cos(ang)
            _draw_glow_chevron(surface, px, py, tx, ty, 11.0, color, 235)


def _draw_saddle_marker(surface: pygame.Surface, sx: float, sy: float) -> None:
    r = 5
    pygame.draw.circle(surface, (*FLOW_REPEL, 55), (round(sx), round(sy)), r + 3)
    pygame.draw.circle(surface, (10, 14, 36, 220), (round(sx), round(sy)), r)
    pygame.draw.circle(surface, (*FLOW_REPEL, 215), (round(sx), round(sy)), r, 1)
    pygame.draw.aaline(surface, (*FLOW_REPEL, 220), (sx - 3, sy - 3), (sx + 3, sy + 3))
    pygame.draw.aaline(surface, (*FLOW_REPEL, 220), (sx - 3, sy + 3), (sx + 3, sy - 3))


def draw_dashed_line(surface, color, a, b, width: int, dash: float = 6) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        return
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    d = 0.0
    while d < length:
        e = min(d + dash, length)
        pygame.draw.line(surface, color, (a[0] + ux * d, a[1] + uy * d),
                         (a[0] + ux * e, a[1] + uy * e), width)
        d += 2 * dash


def lerp_color(a, b, t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _fill_annular_sector(
    surface: pygame.Surface,
    cx: float,
    cy: float,
    r_in: float,
    r_out: float,
    a_start: float,
    a_end: float,
    color: tuple[int, ...],
) -> None:
    """Draw a smooth filled annular polygon sector between `a_start` and `a_end` (radians)."""
    span = abs(a_end - a_start)
    if span < 1e-4 or r_out <= r_in:
        return
    n_steps = max(6, int(math.ceil(span / 0.12)))
    angles = [a_start + (a_end - a_start) * (k / n_steps) for k in range(n_steps + 1)]
    outer = [(cx + r_out * math.cos(a), cy - r_out * math.sin(a)) for a in angles]
    inner = [(cx + r_in * math.cos(a), cy - r_in * math.sin(a)) for a in reversed(angles)]
    pygame.draw.polygon(surface, color, outer + inner)


def _draw_segmented_bar(
    surface: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    arcs: list[tuple[int, float]],
) -> None:
    """Draw a horizontal empirical codeword mass bar across `[x, x + w]`."""
    pygame.draw.rect(surface, (30, 36, 56), (x, y, w, h), border_radius=1)
    cur_f = 0.0
    for cw_idx, prob in arcs:
        p = min(max(float(prob), 0.0), 1.0 - cur_f)
        if p <= 0.004:
            continue
        x0 = x + round(cur_f * w)
        x1 = max(x0 + 1, x + round((cur_f + p) * w))
        pygame.draw.rect(surface, codeword_color(int(cw_idx)), (x0, y, x1 - x0, h))
        cur_f += p


def _lookup_wire_bits(snap: dict, w) -> float:
    """Look up live Channel MI / Transfer Entropy bits for bundled wire `w`."""
    candidates = [
        snap.get("wire_info_bits"),
        snap.get("wire_te_bits"),
        snap.get("wire_mi_bits"),
        snap.get("wire_info_by_pair"),
    ]
    keys = (
        (w.source, w.target, w.sign),
        (w.source, w.target, w.source_port, w.target_port, w.sign),
        (w.source, w.target),
        f"{w.source}->{w.target}",
        f"{w.source}->{w.target}:{'+' if w.sign > 0 else '-'}",
    )
    best = 0.0
    for table in candidates:
        if not isinstance(table, dict):
            continue
        for k in keys:
            if k in table:
                val = table[k]
                if isinstance(val, dict):
                    val = val.get("info_bits", val.get("bits", 0.0))
                best = max(best, float(val))
                break
    return max(0.0, best)


def _lookup_part_grid(
    snap: dict, field_name: str, part_name: str, shape: tuple[int, int]
) -> np.ndarray | None:
    """Extract a 2D `(rows, cols)` cell probability or cell entropy array for camera/array parts."""
    table = snap.get(field_name) or snap.get(f"part_{field_name}")
    arr = None
    if isinstance(table, dict) and part_name in table:
        arr = table[part_name]
    else:
        grids = snap.get("part_grids")
        if isinstance(grids, dict) and part_name in grids:
            sub = grids[part_name]
            if isinstance(sub, dict):
                arr = sub.get(field_name) or sub.get(field_name.replace("grid_", ""))
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float)
    if out.size == shape[0] * shape[1]:
        return out.reshape(shape)
    return None


def run_window(scenario: Scenario, dynamics_mode: int = 0) -> None:
    pygame.init()
    try:
        renderer = Renderer(scenario, dynamics_mode=dynamics_mode)
        screen = pygame.display.set_mode(renderer.window_size)
        pygame.display.set_caption(scenario.name)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                running = renderer.handle(event) and running
            if not renderer.paused:
                renderer.advance(renderer.steps_per_frame)
            renderer.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()
