"""Tri-Panel Strogatz × Shannon scientific visualization studio for Braitenberg Vehicles.

Every figure and animation produced by this module strictly enforces **Axiom 1**
(:func:`braitenberg.dynamics.shannon.validate_single_experiment_axiom`):
one visualization = strictly one isolated experiment and one homogeneous focal vehicle
species, uniting three synchronized panels:

1. **Panel A — Physical Arena**: Stimulus fields / forward FTLE ridges, swarm comet
   trajectories colored by live **Shannon Codeword Hue ``c``**, and **Cyan-Ringed Hero #0**.
2. **Panel B — Ensemble Phase Portrait**: Geometric Strogatz phase space (nullclines,
   limit cycles, hysteresis folds, ``w_c = 0.7`` learning gates, target symmetry cones,
   or Kuramoto order-parameter unit disc) with codeword-colored ensemble comets and
   **Cyan-Ringed Hero #0**.
3. **Panel C — 3-Tier Hero #0 + Shannon Swarm Nervous System** (:mod:`diagram_mpl`):
   Top-down Braitenberg schematic with Joint Codebook Arc-Rings, marginal entropy halos,
   and MI/TE wire glow + 1D Empirical Density Barcodes + Lossless Sorted N×M Bit-Raster.

Architecture rule (AGENTS.md): Never imports ``pygame`` and never opens GUI windows.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Sequence

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from braitenberg.definition import ACTIVE_STRENGTH, VehicleDefinition, load_definition
from braitenberg.dynamics.diagram_mpl import NEURAL_VIEWS, ShannonNeuralPanelArtist, render_shannon_neural_panel
from braitenberg.dynamics.themes import (
    Theme,
    add_poster_header,
    aspect_figsize,
    badge_bbox,
    codeword_color,
    field_cmap,
    ftle_cmap,
    get_theme,
    hero_ring_color,
    plot_comet_trails,
    plot_glow_line,
    plot_hero_beacon,
    plot_lamp_beacon,
    plot_strogatz_equilibrium,
    style_axes,
    style_legend,
    text_halo,
    theme_rc_params,
)
from braitenberg.scenario import Scenario, load_scenario
from braitenberg.world import Arena, Source, Stimulus, World

REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_EXPERIMENTS: tuple[str, ...] = (
    "v4a_orbiter",
    "v5_moth",
    "v5_forager",
    "v7_wary_moth",
    "v9_shy",
    "v12_daydreamer",
)

EXPERIMENT_STEPS: dict[str, tuple[int, int]] = {
    "v4a_orbiter": (480, 3),
    "v5_moth": (520, 3),
    "v5_forager": (640, 4),
    "v7_wary_moth": (640, 4),
    "v9_shy": (680, 4),
    "v12_daydreamer": (520, 3),
}

_ROLLOUT_CACHE: dict[tuple[str, int, int, int], dict[str, Any]] = {}
_FTLE_BG_CACHE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Figure, Grid & FFmpeg Helpers
# ---------------------------------------------------------------------------


def _ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    hb = Path("/opt/homebrew/bin/ffmpeg")
    return str(hb) if hb.exists() else None


def _new_figure(
    aspect: str, target: str, theme: Theme, dpi: int | None = None
) -> tuple[Figure, FigureCanvasAgg, int]:
    w_in, h_in, default_dpi = aspect_figsize(aspect, target)
    actual_dpi = dpi if dpi is not None else default_dpi
    # Even pixel dimensions required by H.264 yuv420p
    px_w = max(2, int(round(w_in * actual_dpi / 2.0)) * 2)
    px_h = max(2, int(round(h_in * actual_dpi / 2.0)) * 2)
    fig = Figure(figsize=(px_w / actual_dpi, px_h / actual_dpi), dpi=actual_dpi, facecolor=theme.fig_bg)
    canvas = FigureCanvasAgg(fig)
    return fig, canvas, actual_dpi


def _save_figure(fig: Figure, out_path: Path, theme: Theme, dpi: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(out_path, dpi=dpi, facecolor=theme.fig_bg, edgecolor="none")
    return out_path


def _make_tripanel_slots(
    fig: Figure, aspect: str
) -> tuple[mpl.axes.Axes, mpl.axes.Axes, SubplotSpec]:
    """Create ``(ax_arena, ax_phase, neural_slot)`` tailored to ``16:9``, ``1:1``, or ``9:16``."""
    if aspect == "9:16":
        gs = fig.add_gridspec(
            3,
            1,
            left=0.105,
            right=0.955,
            bottom=0.048,
            top=0.908,
            height_ratios=[0.265, 0.265, 0.470],
            hspace=0.22,
        )
        return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), gs[2, 0]

    if aspect == "1:1":
        gs = fig.add_gridspec(
            2,
            2,
            left=0.068,
            right=0.974,
            bottom=0.068,
            top=0.876,
            width_ratios=[0.95, 1.07],
            height_ratios=[1.0, 1.0],
            wspace=0.22,
            hspace=0.27,
        )
        return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), gs[:, 1]

    gs = fig.add_gridspec(
        1,
        3,
        left=0.046,
        right=0.984,
        bottom=0.092,
        top=0.848,
        width_ratios=[0.98, 1.02, 1.16],
        wspace=0.20,
    )
    return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), gs[0, 2]


def _add_footer(fig: Figure, theme: Theme, exp_id: str, n_focal: int, aspect: str) -> None:
    y = 0.011 if aspect == "9:16" else 0.017
    x = 0.105 if aspect == "9:16" else (0.068 if aspect == "1:1" else 0.046)
    fs = 6.5 if aspect == "9:16" else (7.0 if aspect == "1:1" else 7.5)
    full_msg = (
        f"Axiom 1 Pure Experiment [{exp_id}] · N = {n_focal} identical vehicles · "
        "Cyan Ring = Hero #0 · Palette Hues = Lossless Joint Shannon Codewords W · Wire Glow = Channel MI / Transfer Entropy"
    )
    short_msg = (
        f"Axiom 1 [{exp_id}] · N={n_focal} · Cyan Ring = Hero #0 · Hues = Joint Shannon Codewords"
    )
    fig.text(x, y, short_msg if aspect != "16:9" else full_msg, color=theme.fg_dim, fontsize=fs, va="bottom", ha="left")


def _format_psi_axis(ax: mpl.axes.Axes) -> None:
    ax.set_ylim(-math.pi, math.pi)
    ax.set_yticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])


def _orient_grid_2d(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.shape == (ny, nx):
        return a
    if a.shape == (nx, ny):
        return a.T
    return a


# ---------------------------------------------------------------------------
# Zero-Allocation Per-Frame Artists (CometArtist + HeroMarkerArtist)
# ---------------------------------------------------------------------------


class CometArtist:
    """Zero-allocation per-frame updater for glowing comet-tailed particle swarms."""

    def __init__(
        self,
        ax: mpl.axes.Axes,
        theme: Theme,
        *,
        max_lw: float = 2.0,
        alpha: float = 0.92,
        head_size: float = 24.0,
        arena_size: tuple[float, float] | None = None,
        psi_wrap: bool = False,
        zorder: float = 6.0,
    ) -> None:
        self.ax = ax
        self.theme = theme
        self.max_lw = max_lw
        self.alpha = alpha
        self.head_size = head_size
        self.arena_size = arena_size
        self.psi_wrap = psi_wrap

        self.halo_lc: LineCollection | None = None
        if theme.glow_enabled:
            self.halo_lc = LineCollection([], capstyle="round", zorder=zorder - 0.2)
            ax.add_collection(self.halo_lc)

        self.core_lc = LineCollection([], capstyle="round", zorder=zorder)
        ax.add_collection(self.core_lc)

        self.halo_heads = None
        if theme.glow_enabled:
            self.halo_heads = ax.scatter([], [], s=head_size * 3.6, edgecolors="none", zorder=zorder + 0.1)

        self.core_heads = ax.scatter(
            [],
            [],
            s=head_size,
            edgecolors="#ffffff" if theme.is_dark else theme.fg,
            linewidths=0.5,
            zorder=zorder + 0.2,
        )

    def update(self, xs: np.ndarray, ys: np.ndarray, base_rgba: np.ndarray) -> None:
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        if xs_arr.ndim == 1:
            xs_arr = xs_arr[:, None]
            ys_arr = ys_arr[:, None]
        t_len, n_p = xs_arr.shape
        if t_len >= 2 and n_p > 0:
            p0 = np.stack([xs_arr[:-1], ys_arr[:-1]], axis=-1)
            p1 = np.stack([xs_arr[1:], ys_arr[1:]], axis=-1)
            segs = np.stack([p0, p1], axis=2)

            valid = np.isfinite(segs).all(axis=(2, 3))
            if self.arena_size is not None:
                w, h = self.arena_size
                valid &= (np.abs(xs_arr[1:] - xs_arr[:-1]) < 0.45 * w) & (
                    np.abs(ys_arr[1:] - ys_arr[:-1]) < 0.45 * h
                )
            if self.psi_wrap:
                valid &= np.abs(ys_arr[1:] - ys_arr[:-1]) < 1.65

            ramp = np.linspace(0.08, 1.0, t_len - 1)[:, None]
            rgba = np.broadcast_to(base_rgba[None, :, :], (t_len - 1, n_p, 4)).copy()
            rgba[:, :, 3] = self.alpha * (ramp**1.55)
            lws = np.broadcast_to(0.45 + (self.max_lw - 0.45) * ramp, (t_len - 1, n_p))

            flat_segs = segs[valid]
            flat_rgba = rgba[valid]
            flat_lws = lws[valid]

            self.core_lc.set_segments(flat_segs)
            self.core_lc.set_color(flat_rgba)
            self.core_lc.set_linewidths(flat_lws)
            if self.halo_lc is not None:
                halo_rgba = flat_rgba.copy()
                halo_rgba[:, 3] *= 0.25
                self.halo_lc.set_segments(flat_segs)
                self.halo_lc.set_color(halo_rgba)
                self.halo_lc.set_linewidths(flat_lws * 2.8)
        else:
            self.core_lc.set_segments([])
            if self.halo_lc is not None:
                self.halo_lc.set_segments([])

        if t_len >= 1 and n_p > 0:
            hx, hy = xs_arr[-1], ys_arr[-1]
            mask = np.isfinite(hx) & np.isfinite(hy)
            pts = np.column_stack([hx[mask], hy[mask]]) if np.any(mask) else np.empty((0, 2))
            hc = base_rgba[mask].copy()
            if len(hc) > 0:
                hc[:, 3] = min(1.0, self.alpha * 1.04)
            self.core_heads.set_offsets(pts)
            if len(hc) > 0:
                self.core_heads.set_facecolor(hc)
            if self.halo_heads is not None:
                self.halo_heads.set_offsets(pts)
                if len(hc) > 0:
                    hh = hc.copy()
                    hh[:, 3] = 0.25 * self.alpha
                    self.halo_heads.set_facecolor(hh)


class HeroMarkerArtist:
    """Signature Cyan Target Ring + history ribbon tracking ``Hero #0`` in Arena & Phase Space."""

    def __init__(
        self,
        ax: mpl.axes.Axes,
        theme: Theme,
        *,
        size: float = 95.0,
        arena_size: tuple[float, float] | None = None,
        psi_wrap: bool = False,
        fov_cone: tuple[float, float] | None = None,
        zorder: float = 9.5,
    ) -> None:
        self.ax = ax
        self.theme = theme
        self.ring_col = hero_ring_color(theme)
        self.arena_size = arena_size
        self.psi_wrap = psi_wrap
        self.fov_cone = fov_cone  # (fov_deg, range_units) for camera vehicles like v9_shy

        self.trace_lc = LineCollection([], colors=self.ring_col, linewidths=1.55, alpha=0.68, zorder=zorder - 1.2)
        ax.add_collection(self.trace_lc)

        self.fov_line: Line2D | None = None
        if fov_cone is not None:
            self.fov_line = ax.plot(
                [], [], color=self.ring_col, lw=0.95, ls="--", alpha=0.52, zorder=zorder - 1.0
            )[0]

        self.glow_sc = None
        if theme.glow_enabled:
            self.glow_sc = ax.scatter(
                [], [], s=size * 4.2, c=[self.ring_col], alpha=0.24, edgecolors="none", zorder=zorder - 0.3
            )
        self.outer_ring = ax.scatter(
            [], [], s=size * 2.15, facecolors="none", edgecolors=self.ring_col, linewidths=1.85, zorder=zorder
        )
        self.inner_dot = ax.scatter(
            [],
            [],
            s=size * 0.68,
            facecolors=self.ring_col,
            edgecolors="#ffffff" if theme.is_dark else theme.fg,
            linewidths=1.1,
            zorder=zorder + 0.2,
        )

    def update(
        self,
        hx_hist: np.ndarray,
        hy_hist: np.ndarray,
        fill_rgba: tuple[float, float, float, float] | str,
        heading_rad: float | None = None,
    ) -> None:
        x_arr = np.asarray(hx_hist, dtype=float)
        y_arr = np.asarray(hy_hist, dtype=float)
        if len(x_arr) >= 2:
            p0 = np.column_stack([x_arr[:-1], y_arr[:-1]])
            p1 = np.column_stack([x_arr[1:], y_arr[1:]])
            segs = np.stack([p0, p1], axis=1)
            valid = np.isfinite(segs).all(axis=(1, 2))
            if self.arena_size is not None:
                w, h = self.arena_size
                valid &= (np.abs(x_arr[1:] - x_arr[:-1]) < 0.45 * w) & (
                    np.abs(y_arr[1:] - y_arr[:-1]) < 0.45 * h
                )
            if self.psi_wrap:
                valid &= np.abs(y_arr[1:] - y_arr[:-1]) < 1.65
            self.trace_lc.set_segments(segs[valid])
        else:
            self.trace_lc.set_segments([])

        if len(x_arr) >= 1 and np.isfinite(x_arr[-1]) and np.isfinite(y_arr[-1]):
            pt = np.array([[x_arr[-1], y_arr[-1]]])
            self.outer_ring.set_offsets(pt)
            self.inner_dot.set_offsets(pt)
            self.inner_dot.set_facecolor([fill_rgba])
            if self.glow_sc is not None:
                self.glow_sc.set_offsets(pt)
            if self.fov_line is not None and heading_rad is not None and self.fov_cone is not None:
                fov_deg, rng = self.fov_cone
                half = math.radians(0.5 * fov_deg)
                x0, y0 = float(x_arr[-1]), float(y_arr[-1])
                angs = np.linspace(heading_rad - half, heading_rad + half, 18)
                arc_x = np.concatenate([[x0], x0 + rng * np.cos(angs), [x0]])
                arc_y = np.concatenate([[y0], y0 + rng * np.sin(angs), [y0]])
                self.fov_line.set_data(arc_x, arc_y)


# ---------------------------------------------------------------------------
# Precomputed Experiment Rollout Cache & Normalization
# ---------------------------------------------------------------------------


def _resolve_experiment_list(experiment: str) -> list[str]:
    key = experiment.strip().lower()
    if key == "all":
        return list(CANONICAL_EXPERIMENTS)
    if key not in CANONICAL_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Choose from: {', '.join(CANONICAL_EXPERIMENTS)}, all"
        )
    return [key]


def get_experiment_rollout(
    experiment_id: str,
    *,
    steps: int | None = None,
    stride: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Load, validate (Axiom 1), simulate, and normalize an isolated experiment rollout."""
    from braitenberg.dynamics.shannon import (
        EXPERIMENTS,
        simulate_experiment_rollout,
        validate_single_experiment_axiom,
    )

    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment_id}'.")

    spec = validate_single_experiment_axiom(EXPERIMENTS[experiment_id])
    def_steps, def_stride = EXPERIMENT_STEPS.get(experiment_id, (480, 3))
    s_steps = steps if steps is not None else def_steps
    s_stride = stride if stride is not None else def_stride

    cache_key = (experiment_id, s_steps, s_stride, seed)
    if cache_key in _ROLLOUT_CACHE:
        return _ROLLOUT_CACHE[cache_key]

    raw = simulate_experiment_rollout(experiment_id, steps=s_steps, stride=s_stride, seed=seed)
    scenario: Scenario = raw.get("scenario") or load_scenario(REPO_ROOT / spec.scenario_path)
    focal_def: VehicleDefinition = raw.get("focal_definition") or load_definition(
        REPO_ROOT / spec.focal_vehicle_path
    )
    validate_single_experiment_axiom(spec, scenario=scenario)

    snapshots: list[dict[str, Any]] = list(raw.get("shannon_snapshots") or raw.get("snapshots") or [])
    n_frames = len(snapshots)
    if n_frames == 0:
        raise RuntimeError(f"simulate_experiment_rollout('{experiment_id}') returned 0 snapshots")

    def _align_2d(arr_like: Any, default_shape: tuple[int, int], fill: float = 0.0) -> np.ndarray:
        if arr_like is None:
            return np.full(default_shape, fill, dtype=float)
        arr = np.asarray(arr_like, dtype=float)
        if arr.shape[0] == n_frames:
            return arr
        idx = raw.get("frame_indices") or raw.get("snapshot_steps")
        if idx is not None and len(idx) == n_frames:
            return arr[np.asarray(idx, dtype=int)]
        lin = np.linspace(0, arr.shape[0] - 1, n_frames).round().astype(int)
        return arr[lin]

    def _align_1d(arr_like: Any) -> np.ndarray:
        if arr_like is None:
            return np.arange(n_frames, dtype=float) * float(scenario.simulation.dt * s_stride)
        arr = np.asarray(arr_like, dtype=float).ravel()
        if len(arr) == n_frames:
            return arr
        idx = raw.get("frame_indices") or raw.get("snapshot_steps")
        if idx is not None and len(idx) == n_frames:
            return arr[np.asarray(idx, dtype=int)]
        lin = np.linspace(0, len(arr) - 1, n_frames).round().astype(int)
        return arr[lin]

    x = np.asarray(raw["x"], dtype=float)
    n_focal = x.shape[1] if x.ndim == 2 else len(snapshots[0].get("vehicle_codeword_ids", [1]))
    x = _align_2d(raw.get("x"), (n_frames, n_focal))
    y = _align_2d(raw.get("y"), (n_frames, n_focal))
    ux = _align_2d(raw.get("ux"), (n_frames, n_focal))
    uy = _align_2d(raw.get("uy"), (n_frames, n_focal))
    heading = _align_2d(raw.get("heading"), (n_frames, n_focal))
    speed = _align_2d(raw.get("speed"), (n_frames, n_focal))
    r_arr = _align_2d(raw.get("r"), (n_frames, n_focal))
    psi_arr = _align_2d(raw.get("psi"), (n_frames, n_focal))
    dr_arr = _align_2d(raw.get("dr"), (n_frames, n_focal))
    energy_arr = _align_2d(raw.get("energy"), (n_frames, n_focal), fill=np.nan)
    t_arr = _align_1d(raw.get("t"))

    # Extract optional phase-space coordinates (px, py) if pre-packed by shannon.py,
    # plus learning weight w_red (for v7_wary_moth) and Z(t) (for v12_daydreamer).
    w_red = raw.get("w_red")
    if w_red is None and "strengths" in raw and isinstance(raw["strengths"], dict):
        for k_s, v_s in raw["strengths"].items():
            if "red" in k_s:
                w_red = v_s
                break
    w_red_arr = _align_2d(w_red, (n_frames, n_focal)) if w_red is not None else None

    z_re = raw.get("z_re")
    z_im = raw.get("z_im")
    if z_re is None and "Z" in raw:
        z_c = np.asarray(raw["Z"], dtype=complex)
        z_re, z_im = np.real(z_c), np.imag(z_c)
    z_re_arr = _align_2d(z_re, (n_frames, n_focal)) if z_re is not None else None
    z_im_arr = _align_2d(z_im, (n_frames, n_focal)) if z_im is not None else None

    phase_x_raw = raw.get("phase_x")
    phase_y_raw = raw.get("phase_y")
    phase_x = _align_2d(phase_x_raw, (n_frames, n_focal)) if phase_x_raw is not None else None
    phase_y = _align_2d(phase_y_raw, (n_frames, n_focal)) if phase_y_raw is not None else None

    # Environmental non-focal targets (e.g. Chaser / Walker in v9_shy)
    env_x, env_y, env_h = None, None, None
    for prefix in ("env_", "stimulus_vehicles_", "non_focal_", "target_"):
        if f"{prefix}x" in raw and raw[f"{prefix}x"] is not None:
            ex = np.asarray(raw[f"{prefix}x"], dtype=float)
            if ex.ndim == 2 and ex.shape[1] > 0:
                env_x = _align_2d(ex, (n_frames, ex.shape[1]))
                env_y = _align_2d(raw.get(f"{prefix}y"), (n_frames, ex.shape[1]))
                env_h = _align_2d(raw.get(f"{prefix}heading"), (n_frames, ex.shape[1]))
                break
    if env_x is None and isinstance(raw.get("env_vehicles"), dict):
        ev = raw["env_vehicles"]
        if "x" in ev and np.asarray(ev["x"]).size > 0:
            ex = np.asarray(ev["x"], dtype=float)
            env_x = _align_2d(ex, (n_frames, ex.shape[1]))
            env_y = _align_2d(ev.get("y"), (n_frames, ex.shape[1]))
            env_h = _align_2d(ev.get("heading"), (n_frames, ex.shape[1]))

    # Choose the peak-synergy showcase frame in [42%..94%] of rollout for static 4K posters
    i_start = max(0, int(0.42 * n_frames))
    i_end = max(i_start + 1, int(0.95 * n_frames))
    best_frame = n_frames - 1
    best_score = -1.0
    for idx in range(i_start, min(n_frames, i_end)):
        sn = snapshots[idx]
        hw = float(sn.get("joint_entropy_bits", 0.0))
        tc = float(sn.get("total_correlation_bits", 0.0))
        smi = float(sn.get("spatial_mi_bits", 0.0))
        hero_diag = sn["hero_diagram"]
        hero_act = sum(
            float(p.level) for p in hero_diag.parts if not isinstance(p.level, list) and p.kind == "unit"
        )
        sc_val = hw + 1.15 * tc + 0.65 * smi + 0.08 * min(hero_act, 4.0)
        if sc_val > best_score:
            best_score = sc_val
            best_frame = idx

    norm = {
        "spec": spec,
        "experiment_id": experiment_id,
        "scenario": scenario,
        "world": scenario.world,
        "focal_def": focal_def,
        "n_focal": n_focal,
        "n_frames": n_frames,
        "t": t_arr,
        "x": x,
        "y": y,
        "ux": ux,
        "uy": uy,
        "heading": heading,
        "speed": speed,
        "r": r_arr,
        "psi": psi_arr,
        "dr": dr_arr,
        "energy": energy_arr,
        "w_red": w_red_arr,
        "z_re": z_re_arr,
        "z_im": z_im_arr,
        "phase_x": phase_x,
        "phase_y": phase_y,
        "env_x": env_x,
        "env_y": env_y,
        "env_heading": env_h,
        "snapshots": snapshots,
        "poster_frame": best_frame,
        "raw": raw,
    }
    _ROLLOUT_CACHE[cache_key] = norm
    return norm


# ---------------------------------------------------------------------------
# Panel A (Arena) & Panel B (Phase Portrait) Setup Per Experiment
# ---------------------------------------------------------------------------


def _get_orbiter_ftle_bg(vdef: VehicleDefinition, world: World) -> dict[str, Any]:
    key = f"{world.name}_{len(world.stimuli.get('light', Stimulus('light')).sources)}"
    if key not in _FTLE_BG_CACHE:
        from braitenberg.dynamics.lagrangian import compute_ftle_and_basins
        from braitenberg.dynamics.phase_plane import find_fixed_points_and_cycles, polar_field

        lcs = compute_ftle_and_basins(vdef, world, nx=96, ny=96, horizon=10.5, dt=0.05)
        s0 = world.stimuli["light"].sources[0]
        ref_world = World(
            name="one_ref",
            arena=Arena(width=120.0, height=120.0, edges="wrap"),
            stimuli={"light": Stimulus(name="light", sources=[Source((60.0, 60.0), s0.intensity, s0.falloff)])},
        )
        r_grid = np.linspace(1.8, 46.0, 75)
        psi_grid = np.linspace(-math.pi, math.pi, 75)
        pf = polar_field(vdef, ref_world, r_grid=r_grid, psi_grid=psi_grid)
        fpc = find_fixed_points_and_cycles(vdef, ref_world, r_bounds=(2.0, 44.0))
        _FTLE_BG_CACHE[key] = {"lcs": lcs, "pf": pf, "fpc": fpc}
    return _FTLE_BG_CACHE[key]


def _setup_arena_and_phase(
    ax_arena: mpl.axes.Axes,
    ax_phase: mpl.axes.Axes,
    rollout: dict[str, Any],
    theme: Theme,
    aspect: str,
) -> tuple[Callable[[int], None], None]:
    """Configure static backgrounds for Panel A & Panel B and return ``update_ab(frame_idx)``."""
    exp_id: str = rollout["experiment_id"]
    world: World = rollout["world"]
    vdef: VehicleDefinition = rollout["focal_def"]
    arena = world.arena
    n_focal = rollout["n_focal"]
    t_arr = rollout["t"]
    xs, ys = rollout["x"], rollout["y"]
    uxs, uys = rollout["ux"], rollout["uy"]
    hdgs = rollout["heading"]
    snapshots = rollout["snapshots"]

    # --- PANEL A: Physical Arena Background ---
    is_unwrapped_dark = exp_id == "v12_daydreamer"
    style_axes(
        ax_arena,
        theme,
        tag="A",
        title=(
            f"Physical Dark Arena · {n_focal} Daydreamers"
            if is_unwrapped_dark
            else f"Physical Arena · {n_focal} {vdef.general.name}s + Hero #0"
        ),
        xlabel=r"Unwrapped $x$" if is_unwrapped_dark else r"Arena $x$",
        ylabel=r"Unwrapped $y$" if is_unwrapped_dark else r"Arena $y$",
        grid=is_unwrapped_dark,
    )
    ax_arena.set_aspect("equal", adjustable="box")

    if exp_id == "v4a_orbiter" and "light" in world.stimuli and world.stimuli["light"].sources:
        bg = _get_orbiter_ftle_bg(vdef, world)
        lcs = bg["lcs"]
        xg, yg = np.asarray(lcs["x"], dtype=float), np.asarray(lcs["y"], dtype=float)
        x1d = xg[0, :] if xg.ndim == 2 else xg
        y1d = yg[:, 0] if yg.ndim == 2 else yg
        ftle = _orient_grid_2d(lcs["ftle"], len(x1d), len(y1d))
        vmin, vmax = float(np.nanpercentile(ftle, 5.0)), float(np.nanpercentile(ftle, 99.2))
        ax_arena.imshow(
            ftle,
            origin="lower",
            extent=[float(x1d[0]), float(x1d[-1]), float(y1d[0]), float(y1d[-1])],
            cmap=ftle_cmap(theme),
            vmin=vmin,
            vmax=max(vmax, vmin + 1e-4),
            interpolation="bilinear",
            aspect="equal",
            zorder=1,
        )
        lc = bg["fpc"].get("limit_cycle")
        r_star = float(lc.get("r_mean", 19.9)) if lc else 19.9
        th_ring = np.linspace(0, 2 * math.pi, 160)
        for src in world.stimuli["light"].sources:
            sx, sy = src.position
            ax_arena.plot(
                sx + r_star * np.cos(th_ring),
                sy + r_star * np.sin(th_ring),
                color=theme.gold if theme.is_dark else theme.amber,
                lw=0.95,
                ls=":",
                alpha=0.58,
                zorder=2.5,
            )
            plot_lamp_beacon(ax_arena, sx, sy, theme, size=78, zorder=8)
        ax_arena.set_xlim(0.0, arena.width)
        ax_arena.set_ylim(0.0, arena.height)

    elif not is_unwrapped_dark and world.stimuli:
        stim_key = "light" if "light" in world.stimuli else next(iter(world.stimuli))
        gx, gy = np.meshgrid(np.linspace(0, arena.width, 110), np.linspace(0, arena.height, 110))
        field = np.asarray(world.stimulus_at(stim_key, gx, gy, 0.0), dtype=float)
        peak = max(float(np.nanmax(field)), 1.0)
        ax_arena.imshow(
            field / peak,
            origin="lower",
            extent=[0.0, arena.width, 0.0, arena.height],
            cmap=field_cmap(theme),
            interpolation="bilinear",
            alpha=0.46 if theme.is_dark else 0.32,
            aspect="equal",
            zorder=1,
        )
        th_ring = np.linspace(0, 2 * math.pi, 160)
        # Draw all stimulus sources (color-coded for red/white lamps in v7_wary_moth)
        drawn_pos: list[tuple[float, float]] = []
        for s_name, stim in sorted(world.stimuli.items(), key=lambda kv: kv[0] == "red"):
            col_lamp = (
                theme.coral
                if s_name == "red"
                else ("#e2e8f0" if (s_name == "white" and theme.is_dark) else None)
            )
            for src in stim.sources:
                sx, sy = src.position
                if any(math.hypot(sx - px, sy - py) < 0.8 for px, py in drawn_pos) and s_name != "red":
                    continue
                drawn_pos.append((sx, sy))
                plot_lamp_beacon(ax_arena, sx, sy, theme, color=col_lamp, size=80, zorder=8)
                if exp_id == "v5_moth" and s_name == "light" and src.intensity > 9.0:
                    r_pain = src.falloff * math.sqrt(max(2.0 * src.intensity / 18.0 - 1.0, 0.0))
                    if r_pain > 1.5:
                        ax_arena.plot(
                            sx + r_pain * np.cos(th_ring),
                            sy + r_pain * np.sin(th_ring),
                            color=theme.coral,
                            lw=1.25,
                            ls="--",
                            alpha=0.80,
                            zorder=3,
                        )
                elif exp_id == "v7_wary_moth" and s_name == "red":
                    r_red_vis = src.falloff * math.sqrt(max(src.intensity / 2.5 - 1.0, 0.0))
                    ax_arena.plot(
                        sx + r_red_vis * np.cos(th_ring),
                        sy + r_red_vis * np.sin(th_ring),
                        color=theme.coral,
                        lw=1.2,
                        ls="--",
                        alpha=0.75,
                        zorder=3,
                    )
                elif exp_id == "v5_forager" and s_name == "light":
                    r_sees = src.falloff * math.sqrt(max(2.0 * src.intensity / 3.0 - 1.0, 0.0))
                    ax_arena.plot(
                        sx + r_sees * np.cos(th_ring),
                        sy + r_sees * np.sin(th_ring),
                        color=theme.emerald,
                        lw=1.05,
                        ls=":",
                        alpha=0.65,
                        zorder=2.8,
                    )
        ax_arena.set_xlim(0.0, arena.width)
        ax_arena.set_ylim(0.0, arena.height)

    elif exp_id == "v9_shy":
        ax_arena.set_xlim(0.0, arena.width)
        ax_arena.set_ylim(0.0, arena.height)
        # Subtle arena coordinate hairlines for open-field Shy arena
        for frac in (0.25, 0.50, 0.75):
            ax_arena.axvline(arena.width * frac, color=theme.grid, lw=0.5, alpha=0.28, zorder=1)
            ax_arena.axhline(arena.height * frac, color=theme.grid, lw=0.5, alpha=0.28, zorder=1)

    else:
        # Vehicle 12 Daydreamer dark unwrapped arena
        pad = 5.0
        cx = 0.5 * (float(np.nanmin(uxs)) + float(np.nanmax(uxs)))
        cy = 0.5 * (float(np.nanmin(uys)) + float(np.nanmax(uys)))
        span = 0.5 * max(float(np.nanptp(uxs)), float(np.nanptp(uys)), 26.0) + pad
        ax_arena.set_xlim(cx - span, cx + span)
        ax_arena.set_ylim(cy - span, cy + span)
        # Faint full Hero #0 geometric mandala preview track
        ax_arena.plot(uxs[:, 0], uys[:, 0], color=theme.fg_dim, lw=0.75, ls=":", alpha=0.30, zorder=1.8)

    # Environmental non-focal targets artist (e.g. Chaser / Walker boxes in v9_shy)
    env_x, env_y, env_h = rollout["env_x"], rollout["env_y"], rollout["env_heading"]
    env_sc = None
    env_front_lc = None
    if env_x is not None and env_y is not None and env_x.shape[1] > 0:
        env_sc = ax_arena.scatter(
            [],
            [],
            s=85.0,
            marker="s",
            facecolors="#94a3b8" if theme.is_dark else "#cbd5e1",
            edgecolors="#ffffff" if theme.is_dark else "#18181b",
            linewidths=1.2,
            label="Env. Target (Chaser/Walker)",
            zorder=7.2,
        )
        env_front_lc = LineCollection(
            [], colors="#ffffff" if theme.is_dark else "#dc2626", linewidths=2.2, zorder=7.4
        )
        ax_arena.add_collection(env_front_lc)

    # --- PANEL B: Bespoke Ensemble Phase Portrait Per Experiment ---
    px_series, py_series, psi_wrap = _prepare_phase_portrait(ax_phase, rollout, theme)

    # Arena & Phase dynamic artists
    arena_wrap = None if is_unwrapped_dark else (arena.width if arena.edges == "wrap" else 1e9, arena.height if arena.edges == "wrap" else 1e9)
    comet_arena = CometArtist(
        ax_arena,
        theme,
        max_lw=2.0,
        head_size=24.0,
        arena_size=arena_wrap,
        zorder=6.0,
    )
    fov_spec = (60.0, 26.0) if exp_id == "v9_shy" else None
    hero_arena = HeroMarkerArtist(
        ax_arena,
        theme,
        size=92.0,
        arena_size=arena_wrap,
        fov_cone=fov_spec,
        zorder=9.5,
    )

    comet_phase = CometArtist(
        ax_phase,
        theme,
        max_lw=1.85,
        head_size=24.0,
        psi_wrap=psi_wrap,
        zorder=6.2,
    )
    hero_phase = HeroMarkerArtist(
        ax_phase,
        theme,
        size=92.0,
        psi_wrap=psi_wrap,
        zorder=9.5,
    )

    time_badge = ax_arena.text(
        0.032,
        0.965,
        "",
        transform=ax_arena.transAxes,
        va="top",
        ha="left",
        color=theme.fg,
        fontsize=7.6 if aspect != "16:9" else 8.0,
        fontweight="bold",
        bbox=badge_bbox(theme),
        zorder=12,
    )

    pos_x = uxs if is_unwrapped_dark else xs
    pos_y = uys if is_unwrapped_dark else ys
    tail_len = 22

    def update_ab(f_idx: int) -> None:
        sn = snapshots[f_idx]
        cids = np.asarray(sn.get("vehicle_codeword_ids", np.zeros(n_focal, dtype=int)), dtype=int)
        if len(cids) < n_focal:
            cids = np.pad(cids, (0, n_focal - len(cids)), constant_values=0)
        rgba_now = np.array([to_rgba(codeword_color(theme, int(c), 6)) for c in cids[:n_focal]], dtype=float)
        hero_fill = tuple(rgba_now[0])

        s0 = max(0, f_idx - tail_len)
        h0 = max(0, f_idx - 55)
        comet_arena.update(pos_x[s0 : f_idx + 1], pos_y[s0 : f_idx + 1], rgba_now)
        hero_arena.update(
            pos_x[h0 : f_idx + 1, 0],
            pos_y[h0 : f_idx + 1, 0],
            hero_fill,
            heading_rad=float(hdgs[f_idx, 0]),
        )

        comet_phase.update(px_series[s0 : f_idx + 1], py_series[s0 : f_idx + 1], rgba_now)
        hero_phase.update(px_series[h0 : f_idx + 1, 0], py_series[h0 : f_idx + 1, 0], hero_fill)

        if env_sc is not None and env_x is not None and env_y is not None:
            pts_e = np.column_stack([env_x[f_idx], env_y[f_idx]])
            env_sc.set_offsets(pts_e)
            if env_front_lc is not None and env_h is not None:
                f_segs = []
                for ex_i, ey_i, eh_i in zip(env_x[f_idx], env_y[f_idx], env_h[f_idx]):
                    fx, fy = ex_i + 1.8 * math.cos(eh_i), ey_i + 1.8 * math.sin(eh_i)
                    nx_e, ny_e = -1.6 * math.sin(eh_i), 1.6 * math.cos(eh_i)
                    f_segs.append([(fx - nx_e, fy - ny_e), (fx + nx_e, fy + ny_e)])
                env_front_lc.set_segments(f_segs)

        n_active_codes = len(set(int(c) for c in cids[:n_focal]))
        time_badge.set_text(f"t = {t_arr[f_idx]:.1f} s · {n_active_codes} active codewords")

    return update_ab, None


def _prepare_phase_portrait(
    ax: mpl.axes.Axes, rollout: dict[str, Any], theme: Theme
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Configure the geometric phase plane on ``ax`` and return ``(px_arr, py_arr, psi_wrap)``."""
    exp_id: str = rollout["experiment_id"]
    vdef: VehicleDefinition = rollout["focal_def"]
    world: World = rollout["world"]
    r_arr = rollout["r"]
    psi_arr = rollout["psi"]
    dr_arr = rollout["dr"]
    energy_arr = rollout["energy"]

    if exp_id == "v4a_orbiter":
        style_axes(
            ax,
            theme,
            tag="B",
            title=r"Cylindrical Phase Portrait $(r, \psi)$",
            xlabel=r"Distance to nearest lamp $r$",
            ylabel=r"Bearing $\psi \in [-\pi, \pi]$",
        )
        bg = _get_orbiter_ftle_bg(vdef, world)
        pf, fpc = bg["pf"], bg["fpc"]
        rg, pg = np.asarray(pf["r"]), np.asarray(pf["psi"])
        dr, dp = np.asarray(pf["dr"]), np.asarray(pf["dpsi"])
        ax.streamplot(rg, pg, dr, dp, color=theme.stream, linewidth=0.72, density=0.92, arrowsize=0.8, zorder=2)
        ax.contour(rg, pg, dr, levels=[0.0], colors=[theme.amber], linewidths=1.3, linestyles="--", alpha=0.85, zorder=3)
        dp_clean = np.where(np.abs(dp) < 8.0, dp, np.nan)
        ax.contour(rg, pg, dp_clean, levels=[0.0], colors=[theme.violet], linewidths=1.3, linestyles="-.", alpha=0.85, zorder=3)
        lc = fpc.get("limit_cycle")
        r_star = float(lc.get("r_mean", 19.9)) if lc else 19.9
        lc_col = theme.gold if theme.is_dark else theme.coral
        ax.axvline(r_star, color=lc_col, lw=1.3, ls=":", alpha=0.78, zorder=3.6)
        for sgn in (1.0, -1.0):
            plot_strogatz_equilibrium(ax, r_star, sgn * math.pi / 2, "stable_spiral", theme, size=58, zorder=7.5)
        ax.set_xlim(1.8, max(46.0, float(np.nanpercentile(r_arr, 98)) * 1.05))
        _format_psi_axis(ax)
        handles = [
            Line2D([0], [0], color=theme.amber, lw=1.3, ls="--", label=r"$\dot{r}=0$"),
            Line2D([0], [0], color=theme.violet, lw=1.3, ls="-.", label=r"$\dot{\psi}=0$"),
            Line2D([0], [0], color=lc_col, lw=1.3, ls=":", label=f"Orbit $r^*\\approx{r_star:.1f}$"),
        ]
        style_legend(ax.legend(handles=handles, loc="upper right"), theme)
        return r_arr, psi_arr, True

    if exp_id == "v5_moth":
        style_axes(
            ax,
            theme,
            tag="B",
            title=r"Irreversible Burn-Latch Fold $(r, \dot{r})$",
            xlabel=r"Distance to nearest lamp $r$",
            ylabel=r"Radial velocity $\dot{r}$ (units/s)",
        )
        r_pain = 8.4
        r_near = 17.3
        if "light" in world.stimuli and world.stimuli["light"].sources:
            s0 = world.stimuli["light"].sources[0]
            r_pain = s0.falloff * math.sqrt(max(2.0 * s0.intensity / 18.0 - 1.0, 0.0))
            r_near = s0.falloff * math.sqrt(max(2.0 * s0.intensity / 10.0 - 1.0, 0.0))

        ax.axhline(0.0, color=theme.fg_dim, lw=0.95, ls=":", alpha=0.65, zorder=2)
        ax.axvline(r_pain, color=theme.coral, lw=1.45, ls="--", alpha=0.88, label=f"Pain fold $r_{{\\mathrm{{pain}}}}\\approx{r_pain:.1f}$", zorder=3)
        ax.axvline(r_near, color=theme.amber, lw=1.15, ls="-.", alpha=0.75, label=f"Near zone $r_{{\\mathrm{{near}}}}\\approx{r_near:.1f}$", zorder=3)
        r_branch = np.linspace(r_pain, 48.0, 120)
        ax.plot(r_branch, np.full_like(r_branch, -4.5), color=theme.cyan, lw=1.3, ls="--", alpha=0.45, label="Attraction sheet (burned=0)", zorder=2.5)
        r_flee = np.linspace(2.5, r_near, 80)
        ax.plot(r_flee, np.full_like(r_flee, +4.5), color=theme.emerald, lw=1.3, ls="--", alpha=0.45, label="Repulsion sheet (burned=1)", zorder=2.5)
        ax.annotate(
            "Burn latch\n0 → 1",
            xy=(r_pain, 3.8),
            xytext=(r_pain + 6.5, 3.8),
            color=theme.coral,
            fontsize=7.4,
            fontweight="bold",
            va="center",
            arrowprops={"arrowstyle": "->", "color": theme.coral, "lw": 1.3},
            path_effects=text_halo(theme),
            zorder=8,
        )
        r_hi = max(46.0, float(np.nanpercentile(r_arr, 97)) * 1.05)
        ax.set_xlim(0.0, r_hi)
        v_bound = max(6.8, float(np.nanmax(np.abs(dr_arr))) * 1.15)
        ax.set_ylim(-v_bound, v_bound)
        style_legend(ax.legend(loc="lower right"), theme)
        return r_arr, dr_arr, False

    if exp_id == "v5_forager":
        style_axes(
            ax,
            theme,
            tag="B",
            title=r"Slow–Fast Relaxation Cycle $(r, E)$",
            xlabel=r"Distance to lamp $r$",
            ylabel=r"Internal energy $E(t)$",
        )
        ax.axhspan(30.0, 90.0, color=theme.cyan, alpha=0.06, zorder=1)
        ax.axhline(90.0, color=theme.emerald, lw=1.4, ls="--", alpha=0.88, label=r"$E_{\mathrm{high}}=90$ (full → explore)", zorder=3)
        ax.axhline(30.0, color=theme.amber, lw=1.4, ls="--", alpha=0.88, label=r"$E_{\mathrm{low}}=30$ (hungry → seek)", zorder=3)
        # Faint Hero #0 complete relaxation loop guide
        ax.plot(r_arr[:, 0], energy_arr[:, 0], color=theme.fg_dim, lw=0.85, ls=":", alpha=0.35, zorder=2)
        ax.set_xlim(0.0, max(48.0, float(np.nanpercentile(r_arr, 98)) * 1.06))
        ax.set_ylim(-2.0, 105.0)
        style_legend(ax.legend(loc="lower right"), theme)
        return r_arr, energy_arr, False

    if exp_id == "v7_wary_moth":
        w_red = rollout["w_red"]
        if w_red is None:
            # Extract from snapshot barcodes or fallback to phase_y
            w_red = rollout["phase_y"] if rollout["phase_y"] is not None else np.zeros_like(r_arr)
        style_axes(
            ax,
            theme,
            tag="B",
            title=r"Hebbian Synaptic Bifurcation $(r_{\mathrm{red}}, w_{\mathrm{red}})$",
            xlabel=r"Distance to hot red lamp $r_{\mathrm{red}}$",
            ylabel=r"Mnemotrix weight $w(\mathrm{sees\_red}\to\mathrm{danger})$",
        )
        ax.axhspan(ACTIVE_STRENGTH, 1.04, color=theme.emerald, alpha=0.09, zorder=1)
        ax.axhline(
            ACTIVE_STRENGTH,
            color=theme.gold if theme.is_dark else theme.coral,
            lw=1.6,
            ls="--",
            label=f"Bifurcation gate $w_c = {ACTIVE_STRENGTH}$",
            zorder=3.5,
        )
        ax.axvline(4.2, color=theme.coral, lw=1.15, ls=":", alpha=0.80, label="Burn zone ($r < 4.2$)", zorder=3)
        ax.axvline(25.5, color=theme.fg_dim, lw=1.0, ls="-.", alpha=0.62, label="Red sight ($r < 26$)", zorder=2.5)
        ax.text(
            0.97,
            0.84,
            "PREEMPTIVE AVOIDANCE\n($w > 0.7$ active)",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=theme.emerald,
            path_effects=text_halo(theme),
            zorder=8,
        )
        ax.text(
            0.97,
            0.26,
            "NAIVE PHOTOTAXIS\n($w \\leq 0.7$ silent)",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=theme.coral,
            path_effects=text_halo(theme),
            zorder=8,
        )
        r_red_x = rollout["phase_x"] if rollout["phase_x"] is not None else r_arr
        ax.set_xlim(0.0, max(48.0, float(np.nanpercentile(r_red_x, 98)) * 1.05))
        ax.set_ylim(-0.03, 1.04)
        style_legend(ax.legend(loc="center right"), theme)
        return r_red_x, w_red, False

    if exp_id == "v9_shy":
        px = rollout["phase_x"] if rollout["phase_x"] is not None else r_arr
        py = rollout["phase_y"] if rollout["phase_y"] is not None else psi_arr
        style_axes(
            ax,
            theme,
            tag="B",
            title=r"Target Symmetry & Reflex Space $(r_{\mathrm{target}}, \psi)$",
            xlabel=r"Distance to nearest stimulus target $r_{\mathrm{target}}$",
            ylabel=r"Target bearing $\psi \in [-\pi, \pi]$",
        )
        fov_half = math.pi / 6.0  # +-30 deg retina FOV
        ax.axhspan(-fov_half, fov_half, color=theme.cyan, alpha=0.08, zorder=1)
        ax.axhline(fov_half, color=theme.cyan, lw=1.05, ls="--", alpha=0.72, label=r"Retina FOV $\pm 30^\circ$")
        ax.axhline(-fov_half, color=theme.cyan, lw=1.05, ls="--", alpha=0.72)
        ax.axhline(0.0, color=theme.fg_dim, lw=0.8, ls=":", alpha=0.60)
        ax.axvline(30.0, color=theme.amber, lw=1.15, ls="-.", alpha=0.78, label="Camera range $r = 30$")
        # Highlight head-on symmetric foveal trigger box (|psi| < ~8 deg, r < 24)
        fovea_psi = math.radians(8.5)
        ax.fill_between(
            [0.0, 24.0],
            [-fovea_psi, -fovea_psi],
            [fovea_psi, fovea_psi],
            color=theme.coral,
            alpha=0.16,
            label="Foveal symmetry → flee",
            zorder=2,
        )
        ax.text(
            13.0,
            0.34,
            "look_left (ψ > 0)",
            color=theme.cyan,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            path_effects=text_halo(theme),
            zorder=8,
        )
        ax.text(
            13.0,
            -0.38,
            "look_right (ψ < 0)",
            color=theme.violet,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            path_effects=text_halo(theme),
            zorder=8,
        )
        ax.set_xlim(0.0, max(48.0, float(np.nanpercentile(px, 97)) * 1.06))
        _format_psi_axis(ax)
        style_legend(ax.legend(loc="upper right"), theme)
        return px, py, True

    # Default / v12_daydreamer: Complex Kuramoto Order Parameter Z(t) = R e^{i Phi}
    z_re = rollout["z_re"]
    z_im = rollout["z_im"]
    if z_re is None or z_im is None:
        z_re = rollout["phase_x"] if rollout["phase_x"] is not None else np.zeros_like(r_arr)
        z_im = rollout["phase_y"] if rollout["phase_y"] is not None else np.zeros_like(r_arr)

    style_axes(
        ax,
        theme,
        tag="B",
        title=r"Cortical Ring Order Parameter $Z(t) = R e^{i\Phi}$",
        xlabel=r"$\mathrm{Re}\,Z(t)$",
        ylabel=r"$\mathrm{Im}\,Z(t)$",
        grid=False,
    )
    ax.set_aspect("equal", adjustable="box")
    ring_r = 0.42
    th_c = np.linspace(0, 2 * math.pi, 180)
    ax.plot(ring_r * np.cos(th_c), ring_r * np.sin(th_c), color=theme.fg_dim, lw=0.95, ls=":", alpha=0.55, zorder=2)
    ax.plot(0.18 * np.cos(th_c), 0.18 * np.sin(th_c), color=theme.emerald, lw=0.95, ls="--", alpha=0.55, label="Critical wave band", zorder=2)

    ang8 = 2.0 * math.pi * np.arange(8) / 8.0
    rx8, ry8 = ring_r * np.cos(ang8), ring_r * np.sin(ang8)
    for k in range(8):
        for step_k in (1, 3):
            j = (k + step_k) % 8
            ax.plot([rx8[k], rx8[j]], [ry8[k], ry8[j]], color=theme.spine, lw=0.75, alpha=0.45, zorder=1.8)
    gestures = ["FWD", "L", "R", "STOP", "FWD", "L", "R", "STOP"]
    for k in range(8):
        lx, ly = (ring_r + 0.085) * math.cos(ang8[k]), (ring_r + 0.085) * math.sin(ang8[k])
        ax.text(
            lx,
            ly,
            f"I{k}·{gestures[k]}",
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=theme.fg_muted,
            path_effects=text_halo(theme),
            zorder=8,
        )
    ax.scatter([0.0], [0.0], s=60, marker="D", c=[theme.violet], edgecolors="#ffffff" if theme.is_dark else theme.fg, label="Caudate origin (Z=0)", zorder=7)
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(-0.56, 0.60)
    style_legend(ax.legend(loc="lower right"), theme)
    return z_re, z_im, False


# ---------------------------------------------------------------------------
# Tri-Panel Figure Builder (Static 4K Poster & Animated Frame Updater)
# ---------------------------------------------------------------------------


def build_tripanel_figure(
    experiment_id: str,
    *,
    theme_name: str = "cosmic",
    aspect: str = "16:9",
    neural_view: str = "composite",
    target: str = "poster",
    dpi: int | None = None,
    frame_index: int | None = None,
) -> tuple[Figure, FigureCanvasAgg, int, Callable[[int], None], dict[str, Any]]:
    """Construct a complete Axiom-1 Tri-Panel figure and return its frame ``update(f_idx)``."""
    from braitenberg.dynamics.shannon import EXPERIMENTS, validate_single_experiment_axiom

    if neural_view not in NEURAL_VIEWS:
        raise ValueError(f"Unknown neural_view '{neural_view}'. Choose from: {', '.join(NEURAL_VIEWS)}")

    spec = validate_single_experiment_axiom(EXPERIMENTS[experiment_id])
    rollout = get_experiment_rollout(experiment_id)
    validate_single_experiment_axiom(spec, scenario=rollout["scenario"])

    theme = get_theme(theme_name)
    fig, canvas, actual_dpi = _new_figure(aspect, target, theme, dpi)
    compact = aspect != "16:9"

    add_poster_header(
        fig,
        theme,
        spec.title,
        spec.subtitle,
        kicker=f"BRAITENBERG × STROGATZ × SHANNON · AXIOM 1 PURE EXPERIMENT [{spec.experiment_id.upper()}]",
        compact=compact,
    )

    ax_arena, ax_phase, slot_neural = _make_tripanel_slots(fig, aspect)
    update_ab, _ = _setup_arena_and_phase(ax_arena, ax_phase, rollout, theme, aspect)

    init_idx = frame_index if frame_index is not None else int(rollout["poster_frame"])
    init_idx = int(np.clip(init_idx, 0, rollout["n_frames"] - 1))

    neural_artist = render_shannon_neural_panel(
        fig,
        slot_neural,
        rollout["snapshots"][init_idx],
        theme,
        neural_view=neural_view,
        aspect=aspect,
    )
    _add_footer(fig, theme, experiment_id, rollout["n_focal"], aspect)

    def update_frame(f_idx: int) -> None:
        clamped = int(np.clip(f_idx, 0, rollout["n_frames"] - 1))
        update_ab(clamped)
        neural_artist.update(rollout["snapshots"][clamped])

    update_frame(init_idx)
    return fig, canvas, actual_dpi, update_frame, rollout


def build_experiment_poster(
    experiment_id: str,
    *,
    theme_name: str = "cosmic",
    aspect: str = "16:9",
    neural_view: str = "composite",
    dpi: int | None = None,
) -> Figure:
    """Build a static 4K-grade Tri-Panel poster for strictly one isolated ``experiment_id``."""
    theme = get_theme(theme_name)
    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _, _, _ = build_tripanel_figure(
            experiment_id,
            theme_name=theme_name,
            aspect=aspect,
            neural_view=neural_view,
            target="poster",
            dpi=dpi,
        )
        return fig


def render_posters(
    *,
    experiment: str = "all",
    neural_view: str = "composite",
    theme: str = "cosmic",
    aspect: str = "16:9",
    out_dir: str | Path = ".",
    dpi: int | None = None,
    fmt: str = "png",
    suffix: str = "",
) -> list[Path]:
    """Render static Tri-Panel poster(s) for one or all 6 canonical pure experiments."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    theme_obj = get_theme(theme)
    _, _, default_dpi = aspect_figsize(aspect, "poster")
    actual_dpi = dpi if dpi is not None else default_dpi

    exp_ids = _resolve_experiment_list(experiment)
    written: list[Path] = []
    for exp_id in exp_ids:
        fig = build_experiment_poster(
            exp_id,
            theme_name=theme,
            aspect=aspect,
            neural_view=neural_view,
            dpi=actual_dpi,
        )
        dest = out_path / f"{exp_id}_tripanel{suffix}.{fmt.lower()}"
        _save_figure(fig, dest, theme_obj, actual_dpi)
        written.append(dest)
    return written


# ---------------------------------------------------------------------------
# GIF & 60 FPS H.264 MP4 Encoder Pipeline
# ---------------------------------------------------------------------------


def _save_palette_gif(frames: list[Image.Image], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / max(1, fps) / 10.0)) * 10)
    sample_idx = sorted({0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1})
    w, h = frames[0].size
    strip = Image.new("RGB", (w, h * len(sample_idx)))
    for i, idx in enumerate(sample_idx):
        strip.paste(frames[idx], (0, i * h))
    palette_img = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def _render_single_animation(
    experiment_id: str,
    *,
    neural_view: str = "composite",
    theme_name: str = "cosmic",
    aspect: str = "16:9",
    fps: int = 60,
    seconds: float = 7.0,
    out_dir: Path,
    formats: Sequence[str] = ("gif", "mp4"),
    dpi: int | None = None,
    mp4_subdir: Path | None = None,
    mp4_aspect_tag: bool = False,
) -> list[Path]:
    theme = get_theme(theme_name)
    want_mp4 = "mp4" in formats
    want_gif = "gif" in formats
    ffmpeg_exe = _ffmpeg_bin() if want_mp4 else None

    target_kind = "mp4" if (want_mp4 and ffmpeg_exe) else "gif"
    effective_fps = max(1, fps) if (want_mp4 and ffmpeg_exe) else min(max(1, fps), 24)
    n_video_frames = max(2, int(round(effective_fps * seconds)))

    with mpl.rc_context(theme_rc_params(theme)):
        fig, canvas, _, update_fn, rollout = build_tripanel_figure(
            experiment_id,
            theme_name=theme_name,
            aspect=aspect,
            neural_view=neural_view,
            target=target_kind,
            dpi=dpi,
            frame_index=0,
        )
        canvas.draw()
        h_px, w_px = np.asarray(canvas.buffer_rgba()).shape[:2]
        n_rollout = rollout["n_frames"]

        written: list[Path] = []
        proc: subprocess.Popen | None = None
        mp4_path: Path | None = None

        if want_mp4 and ffmpeg_exe is not None:
            dest_dir = mp4_subdir if mp4_subdir is not None else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_{aspect.replace(':', 'x')}" if mp4_aspect_tag else ""
            mp4_path = dest_dir / f"{experiment_id}{tag}.mp4"
            cmd = [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w_px}x{h_px}",
                "-r",
                str(effective_fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "19",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        gif_stride = max(1, int(round(effective_fps / 20.0))) if want_gif else 1
        gif_fps = max(1, int(round(effective_fps / gif_stride)))
        gif_frames: list[Image.Image] = []
        max_gif_side = 1040
        scale = (
            min(1.0, max_gif_side / max(w_px, h_px))
            if (dpi is None and max(w_px, h_px) > max_gif_side)
            else 1.0
        )
        gif_size = (max(2, int(round(w_px * scale))), max(2, int(round(h_px * scale))))

        last_step = -1
        last_rgb: np.ndarray | None = None

        try:
            for v_idx in range(n_video_frames):
                r_step = min(
                    n_rollout - 1,
                    int(round(v_idx * (n_rollout - 1) / max(1, n_video_frames - 1))),
                )
                if r_step != last_step or last_rgb is None:
                    update_fn(r_step)
                    canvas.draw()
                    last_rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
                    last_step = r_step

                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(last_rgb.tobytes())
                if want_gif and (v_idx % gif_stride == 0):
                    img = Image.fromarray(last_rgb, mode="RGB")
                    if gif_size != (w_px, h_px):
                        img = img.resize(gif_size, Image.Resampling.LANCZOS)
                    gif_frames.append(img)
        finally:
            if proc is not None:
                if proc.stdin is not None:
                    proc.stdin.close()
                stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"ffmpeg failed ({ret}): {stderr_bytes.decode(errors='replace')}")
                if mp4_path is not None:
                    written.append(mp4_path)

        if want_gif and gif_frames:
            gif_path = out_dir / f"{experiment_id}.gif"
            _save_palette_gif(gif_frames, gif_path, gif_fps)
            written.append(gif_path)

        return written


def render_animations(
    *,
    experiment: str = "all",
    neural_view: str = "composite",
    theme: str = "cosmic",
    aspect: str = "16:9",
    fps: int = 60,
    seconds: float = 7.0,
    out_dir: str | Path = ".",
    formats: Sequence[str] = ("gif", "mp4"),
    dpi: int | None = None,
) -> list[Path]:
    """Render synchronized Tri-Panel animation(s) for one or all 6 canonical pure experiments."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_ids = _resolve_experiment_list(experiment)
    written: list[Path] = []
    for exp_id in exp_ids:
        written.extend(
            _render_single_animation(
                exp_id,
                neural_view=neural_view,
                theme_name=theme,
                aspect=aspect,
                fps=fps,
                seconds=seconds,
                out_dir=out_path,
                formats=formats,
                dpi=dpi,
            )
        )
    return written


def render_showcase(
    *,
    out_dir: str | Path = "experiments/strogatz-dynamics/media",
    fps: int = 60,
    seconds: float = 7.0,
    neural_view: str = "composite",
) -> list[Path]:
    """Generate the standardized 1-Experiment-Per-File gallery:
    - 6 Cosmic 4K (3840×2160) Tri-Panel Posters: ``<exp>_tripanel.png``
    - 6 Strogatz Textbook 4K Tri-Panel Posters: ``<exp>_tripanel_textbook.png``
    - 6 GitHub README Animated GIFs (16:9): ``<exp>.gif``
    - 18 Local 60fps H.264 MP4 Social Cuts: ``mp4/<exp>_{16x9,1x1,9x16}.mp4``
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mp4_dir = out_path / "mp4"
    written: list[Path] = []

    # Precompute each experiment rollout once, then render posters + GIF + 3 MP4 cuts
    for exp_id in CANONICAL_EXPERIMENTS:
        get_experiment_rollout(exp_id)

    written.extend(
        render_posters(
            experiment="all",
            neural_view=neural_view,
            theme="cosmic",
            aspect="16:9",
            out_dir=out_path,
            fmt="png",
            suffix="",
        )
    )
    written.extend(
        render_posters(
            experiment="all",
            neural_view=neural_view,
            theme="textbook",
            aspect="16:9",
            out_dir=out_path,
            fmt="png",
            suffix="_textbook",
        )
    )

    for exp_id in CANONICAL_EXPERIMENTS:
        written.extend(
            _render_single_animation(
                exp_id,
                neural_view=neural_view,
                theme_name="cosmic",
                aspect="16:9",
                fps=fps,
                seconds=seconds,
                out_dir=out_path,
                formats=("gif", "mp4"),
                mp4_subdir=mp4_dir,
                mp4_aspect_tag=True,
            )
        )
        for social_aspect in ("1:1", "9:16"):
            written.extend(
                _render_single_animation(
                    exp_id,
                    neural_view=neural_view,
                    theme_name="cosmic",
                    aspect=social_aspect,
                    fps=fps,
                    seconds=seconds,
                    out_dir=out_path,
                    formats=("mp4",),
                    mp4_subdir=mp4_dir,
                    mp4_aspect_tag=True,
                )
            )
    return written
