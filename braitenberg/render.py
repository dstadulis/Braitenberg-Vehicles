"""Real-time pygame renderer. The only module that imports pygame.

It reads the simulation state and draws it; user interaction (dragging sources,
pausing, resetting) changes only the running simulation, never the files.

Keys: space pause/resume, right arrow step (when paused), R reset, T trail,
V camera views, D dynamics overlay, Tab next stimulus, +/- simulation speed, Esc quit.
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

# Live Strogatz complex-dynamics overlay (cycled with the D key).
DYNAMICS_MODES = ("Off", "Flow Field", "Phase Portrait", "Flow + Phase")
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
    def __init__(self, scenario: Scenario, dynamics_mode: int = 0):
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
        if self.selected >= len(self.sim.vehicles):
            self.selected = 0
        self.record_trails()

    def advance(self, steps: int) -> None:
        for _ in range(steps):
            self.sim.step()
            self.record_trails()

    def record_trails(self) -> None:
        for idx, (trail, kin, hist, v) in enumerate(
            zip(self.trails, self.trail_kinematics, self.phase_history, self.sim.vehicles)
        ):
            trail.append((v.x, v.y))
            kin.append((v.speed, v.turn_rate))
            src_info = self._primary_source_for(v)
            if src_info is not None:
                stim_name, src_idx, src = src_info
                r_val, psi_val = _polar_coords(v.x, v.y, v.heading, src.position, self.sim.world.arena)
                src_tag = (stim_name, src_idx)
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
            elif event.key == pygame.K_TAB and self.stimulus_names:
                self.stimulus_index = (self.stimulus_index + 1) % len(self.stimulus_names)
                self.heatmap = None
                self.flow_cache_key = None
                self.phase_bg_key = None
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

        # Mode 3: luminous velocity + turn-curvature colored comet trails.
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
            for k in range(1, n):
                (x0, y0), (x1, y1) = points[k - 1], points[k]
                if abs(x1 - x0) > arena.width / 2 or abs(y1 - y0) > arena.height / 2:
                    continue
                spd, omg = kins[k] if k < len(kins) else (v.speed, v.turn_rate)
                s_norm = min(max(spd / v_ref, 0.0), 1.0)
                c_norm = min(abs(omg) / 1.8, 1.0)
                base_col = lerp_color(FLOW_ATTRACT, FLOW_ORBIT, s_norm)
                seg_col = lerp_color(base_col, FLOW_REPEL, 0.72 * c_norm)
                fade = (k / n) ** 1.35
                p0, p1 = self.to_screen(x0, y0), self.to_screen(x1, y1)
                if is_sel and k > n - 90:
                    halo_a = int(85 * fade)
                    pygame.draw.line(glow_layer, (*seg_col, halo_a), p0, p1, 6)
                core_col = lerp_color(BACKGROUND, seg_col, 0.22 + 0.78 * fade)
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
        for v in self.sim.vehicles:
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
        lines += ["space pause  Right step  R reset  T trail  V views  D dynamics  +/- speed",
                  "Tab stimulus  drag source  click vehicle"]
        # Over a tall arena, in its top corner; below a short one, so it hides nothing.
        y = 6 if self.size[1] >= MIN_WINDOW_HEIGHT else self.size[1] + 10
        for line in lines:
            text = self.font.render(line, True, TEXT)
            surface.blit(text, (8, y))
            y += text.get_height() + 2

        if self.dynamics_mode > 0:
            label = f"Dynamics: {DYNAMICS_MODES[self.dynamics_mode]} [D]"
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
            # Sensorless or dark-field vehicle (e.g. Vehicle 12 Daydreamer):
            # draw subtle geometric reticle + live instantaneous osculating curvature circle!
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
        # Subtle polar reticle centered in arena.
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
            if v.learning_inputs:
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

        # Live glowing comet head at current (r_now, psi_now).
        hx, hy = to_plot(r_now, psi_now)
        pygame.draw.circle(card, (*FLOW_ORBIT, 85), (round(hx), round(hy)), 8)
        pygame.draw.circle(card, FLOW_ORBIT, (round(hx), round(hy)), 4)
        pygame.draw.circle(card, (255, 255, 255), (round(hx), round(hy)), 2)
        card.set_clip(prev_clip)

        if has_gauge:
            self._draw_secondary_gauge(card, ox + 10, py0 + ph + 15, cw - 20, gauge_h - 4, v, hist)

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
            col = lerp_color(FLOW_ATTRACT, PHASE_VIOLET, fade)
            pygame.draw.line(card, (*col, int(45 + 185 * fade)), p0, p1, 2)

        # Draw ring neurons around outer rim + excitatory chord arcs between active ideas.
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

        zx, zy = z_to_px(z_re, z_im)
        pygame.draw.line(card, (*PHASE_VIOLET, 190), (mcx, mcy), (zx, zy), 2)
        pygame.draw.circle(card, (*PHASE_VIOLET, 95), (round(zx), round(zy)), 7)
        pygame.draw.circle(card, (255, 255, 255), (round(zx), round(zy)), 3)

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
        omg_max = max(max((abs(h[4]) for h in hist), default=0.8), 0.8)

        def vw_to_px(sv: float, ov: float) -> tuple[float, float]:
            u = min(max(sv / v_max, 0.0), 1.0)
            w = min(max(ov / (1.15 * omg_max), -1.0), 1.0)
            return rx0 + 4 + u * (rw - 8), mid_y - w * (rh * 0.44)

        for k in range(1, nh):
            s0, o0 = hist[k - 1][3], hist[k - 1][4]
            s1, o1 = hist[k][3], hist[k][4]
            fade = (k / nh) ** 1.5
            col = lerp_color(FLOW_ATTRACT, FLOW_ORBIT, min(abs(o1) / omg_max, 1.0))
            pygame.draw.line(card, (*col, int(50 + 185 * fade)),
                             vw_to_px(s0, o0), vw_to_px(s1, o1), 2)

        kx, ky = vw_to_px(v.speed, v.turn_rate)
        pygame.draw.circle(card, (*FLOW_ORBIT, 90), (round(kx), round(ky)), 6)
        pygame.draw.circle(card, FLOW_ORBIT, (round(kx), round(ky)), 3)

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
