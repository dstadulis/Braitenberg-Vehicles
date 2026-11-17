"""Generative scientific visualization studio for Braitenberg Vehicles.

Renders the 4-Act nonlinear dynamics poster suite, the Master Strogatz Synthesis Atlas,
synchronized 60 FPS multi-aspect animations (``16:9``, ``1:1``, ``9:16``), and the
curated ``showcase`` gallery in both ``cosmic`` and ``textbook`` aesthetics.

Architecture rule (AGENTS.md): Never imports ``pygame`` and never opens GUI windows;
all rendering uses Matplotlib's headless ``Figure`` + ``FigureCanvasAgg`` / SVG canvas.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Sequence

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import numpy as np
from PIL import Image

from braitenberg.definition import ACTIVE_STRENGTH, VehicleDefinition, load_definition
from braitenberg.dynamics.themes import (
    Theme,
    add_poster_header,
    aspect_figsize,
    basin_colors,
    field_cmap,
    ftle_cmap,
    get_theme,
    phase_cmap,
    plot_comet_trails,
    plot_glow_line,
    plot_glow_scatter,
    plot_lamp_beacon,
    plot_strogatz_equilibrium,
    style_axes,
    style_legend,
    text_halo,
    theme_rc_params,
)
from braitenberg.scenario import Placement, Scenario, SimulationSettings, load_scenario
from braitenberg.sim import Simulation
from braitenberg.world import Arena, Source, Stimulus, World, load_world

REPO_ROOT = Path(__file__).resolve().parents[2]

POSTER_FILENAMES: dict[str, str] = {
    "1": "act1_polar_phase_portraits",
    "2": "act2_lagrangian_chaos_ftle",
    "3": "act3_hysteresis_and_plasticity",
    "4": "act4_neural_criticality",
    "atlas": "strogatz_atlas",
}

SCENE_STEMS: dict[str, str] = {
    "orbiter": "orbiter",
    "ftle-swarm": "ftle_swarm",
    "ftle_swarm": "ftle_swarm",
    "plasticity": "plasticity",
    "daydreamer": "daydreamer",
}


# ---------------------------------------------------------------------------
# Asset & World Helpers
# ---------------------------------------------------------------------------


def _vehicle_def(filename: str) -> VehicleDefinition:
    return load_definition(REPO_ROOT / "vehicles" / filename)


def _world(filename: str) -> World:
    return load_world(REPO_ROOT / "worlds" / filename)


def _scenario(filename: str) -> Scenario:
    return load_scenario(REPO_ROOT / "scenarios" / filename)


def _three_lamp_world() -> World:
    """Load worlds/three-lights-chaos.yaml if present, else build an equilateral 3-lamp chaos world."""
    candidate = REPO_ROOT / "worlds" / "three-lights-chaos.yaml"
    if candidate.exists():
        return load_world(candidate)
    cx, cy, radius = 50.0, 48.0, 22.5
    angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3]
    sources = [
        Source(
            position=(round(cx + radius * math.cos(a), 3), round(cy + radius * math.sin(a), 3)),
            intensity=10.0,
            falloff=15.0,
        )
        for a in angles
    ]
    return World(
        name="Three lights chaos",
        description="Equilateral three-lamp arena with competing preferred-distance rings.",
        arena=Arena(width=100.0, height=100.0, edges="wrap"),
        stimuli={"light": Stimulus(name="light", sources=sources)},
    )


def _primary_sources(world: World, stimulus_name: str | None = None) -> list[tuple[float, float]]:
    if not world.stimuli:
        return []
    key = stimulus_name if (stimulus_name in world.stimuli) else next(iter(world.stimuli))
    return [tuple(s.position) for s in world.stimuli[key].sources]


def _ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    hb = Path("/opt/homebrew/bin/ffmpeg")
    return str(hb) if hb.exists() else None


def _new_figure(aspect: str, target: str, theme: Theme, dpi: int | None = None) -> tuple[Figure, FigureCanvasAgg, int]:
    w_in, h_in, default_dpi = aspect_figsize(aspect, target)
    actual_dpi = dpi if dpi is not None else default_dpi
    # Guarantee even pixel dimensions for H.264 yuv420p compatibility
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


def _poster_axes_4(fig: Figure, aspect: str) -> list[mpl.axes.Axes]:
    """Create a harmonious 4-panel subplot grid adapted to 16:9, 1:1, or 9:16."""
    if aspect == "9:16":
        gs = fig.add_gridspec(
            4,
            1,
            left=0.13,
            right=0.935,
            bottom=0.048,
            top=0.888,
            hspace=0.37,
        )
        return [fig.add_subplot(gs[i, 0]) for i in range(4)]
    if aspect == "1:1":
        gs = fig.add_gridspec(
            2,
            2,
            left=0.088,
            right=0.955,
            bottom=0.075,
            top=0.855,
            wspace=0.28,
            hspace=0.35,
        )
        return [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    gs = fig.add_gridspec(
        2,
        2,
        left=0.062,
        right=0.962,
        bottom=0.092,
        top=0.835,
        wspace=0.24,
        hspace=0.38,
    )
    return [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]


def _add_footer(fig: Figure, theme: Theme, text: str, aspect: str) -> None:
    y = 0.014 if aspect == "9:16" else 0.020
    x = 0.13 if aspect == "9:16" else (0.088 if aspect == "1:1" else 0.062)
    fig.text(x, y, text, color=theme.fg_dim, fontsize=7.6, va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Fast In-Place Comet Artist for 60 FPS Animations
# ---------------------------------------------------------------------------


class CometArtist:
    """Zero-allocation per-frame updater for glowing comet-tailed particle swarms."""

    def __init__(
        self,
        ax: mpl.axes.Axes,
        theme: Theme,
        *,
        max_lw: float = 2.1,
        alpha: float = 0.92,
        head_size: float = 22.0,
        arena_size: tuple[float, float] | None = None,
        psi_wrap: bool = False,
        zorder: float = 6,
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
            self.halo_heads = ax.scatter([], [], s=head_size * 3.8, edgecolors="none", zorder=zorder + 0.1)

        self.core_heads = ax.scatter(
            [],
            [],
            s=head_size,
            edgecolors="#ffffff" if theme.is_dark else theme.fg,
            linewidths=0.45,
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
                valid &= np.abs(ys_arr[1:] - ys_arr[:-1]) < (math.pi * 0.85)

            ramp = np.linspace(0.06, 1.0, t_len - 1)[:, None]
            rgba = np.broadcast_to(base_rgba[None, :, :], (t_len - 1, n_p, 4)).copy()
            rgba[:, :, 3] = self.alpha * (ramp ** 1.65)
            lws = np.broadcast_to(0.35 + (self.max_lw - 0.35) * ramp, (t_len - 1, n_p))

            flat_segs = segs[valid]
            flat_rgba = rgba[valid]
            flat_lws = lws[valid]

            self.core_lc.set_segments(flat_segs)
            self.core_lc.set_color(flat_rgba)
            self.core_lc.set_linewidths(flat_lws)
            if self.halo_lc is not None:
                halo_rgba = flat_rgba.copy()
                halo_rgba[:, 3] *= 0.24
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
                    hh[:, 3] = 0.24 * self.alpha
                    self.halo_heads.set_facecolor(hh)


# ---------------------------------------------------------------------------
# Shared Plotting Building Blocks (Reused across Posters, Atlas & Animations)
# ---------------------------------------------------------------------------


def _format_psi_axis(ax: mpl.axes.Axes) -> None:
    ax.set_ylim(-math.pi, math.pi)
    ax.set_yticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])


def _integrate_polar_sample(
    r0: float,
    psi0: float,
    r_grid: np.ndarray,
    psi_grid: np.ndarray,
    dr: np.ndarray,
    dpsi: np.ndarray,
    steps: int = 380,
    dt: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast bilinear integration on a precomputed (r, psi) polar vector field for phase curves."""
    from braitenberg.dynamics.phase_plane import wrap_pi

    rs = np.empty(steps, dtype=float)
    psis = np.empty(steps, dtype=float)
    r_min, r_max = float(r_grid[0]), float(r_grid[-1])
    nr, npsi = len(r_grid), len(psi_grid)
    r, psi = float(r0), float(psi0)

    for k in range(steps):
        rs[k] = r
        psis[k] = psi
        if not (r_min <= r <= r_max):
            rs[k:] = np.nan
            psis[k:] = np.nan
            break
        # Normalized fractional indices
        ir = np.interp(r, r_grid, np.arange(nr - 1e-6))
        ip = np.interp(psi, psi_grid, np.arange(npsi - 1e-6))
        i0, j0 = min(int(ir), nr - 2), min(int(ip), npsi - 2)
        fr, fp = ir - i0, ip - j0
        vr = (
            (1 - fp) * ((1 - fr) * dr[j0, i0] + fr * dr[j0, i0 + 1])
            + fp * ((1 - fr) * dr[j0 + 1, i0] + fr * dr[j0 + 1, i0 + 1])
        )
        vp = (
            (1 - fp) * ((1 - fr) * dpsi[j0, i0] + fr * dpsi[j0, i0 + 1])
            + fp * ((1 - fr) * dpsi[j0 + 1, i0] + fr * dpsi[j0 + 1, i0 + 1])
        )
        r_next = r + vr * dt
        psi_next = float(wrap_pi(psi + vp * dt))
        # Break line segments cleanly across periodic +-pi cylinder seam
        if abs(psi_next - psi) > math.pi:
            if k + 1 < steps:
                rs[k + 1] = np.nan
                psis[k + 1] = np.nan
            r = r_next
            psi = psi_next
            continue
        r, psi = r_next, psi_next
    return rs, psis


def _draw_polar_portrait(
    ax: mpl.axes.Axes,
    vdef: VehicleDefinition,
    world: World,
    theme: Theme,
    *,
    tag: str,
    title: str,
    r_max: float = 44.0,
    sample_ics: Sequence[tuple[float, float]] = (),
    show_nullclines: bool = True,
    show_legend: bool = False,
) -> dict:
    """Render a complete cylindrical (r, psi) phase portrait on ``ax``."""
    from braitenberg.dynamics.phase_plane import find_fixed_points_and_cycles, polar_field

    r_grid = np.linspace(1.5, r_max, 90)
    psi_grid = np.linspace(-math.pi, math.pi, 91)
    pf = polar_field(vdef, world, r_grid=r_grid, psi_grid=psi_grid)
    dr = np.asarray(pf["dr"], dtype=float)
    dpsi = np.asarray(pf["dpsi"], dtype=float)
    speed = np.asarray(pf["speed"], dtype=float)

    style_axes(
        ax,
        theme,
        tag=tag,
        title=title,
        xlabel=r"Distance to lamp $r$",
        ylabel=r"Bearing $\psi$ (rad)",
    )
    ax.set_xlim(float(r_grid[0]), float(r_grid[-1]))
    _format_psi_axis(ax)

    # Subtle speed wash in cosmic mode
    if theme.is_dark:
        ax.pcolormesh(
            r_grid,
            psi_grid,
            speed,
            cmap=field_cmap(theme),
            alpha=0.16,
            shading="auto",
            zorder=1,
        )

    # Background streamplot
    ax.streamplot(
        r_grid,
        psi_grid,
        dr,
        dpsi,
        color=theme.stream,
        linewidth=0.75 if theme.is_dark else 0.85,
        density=1.05,
        arrowsize=0.85,
        zorder=2,
    )

    # Invariant manifold psi = 0
    ax.axhline(0.0, color=theme.fg_dim, lw=0.75, ls=":", alpha=0.55, zorder=2.5)

    # Nullclines: dr = 0 and dpsi = 0
    if show_nullclines:
        ax.contour(
            r_grid,
            psi_grid,
            dr,
            levels=[0.0],
            colors=[theme.amber],
            linewidths=1.35,
            linestyles="--",
            alpha=0.88,
            zorder=3.5,
        )
        # Mask dpsi discontinuities near +-pi wrap if any
        dpsi_clean = np.where(np.abs(dpsi) < 8.0, dpsi, np.nan)
        ax.contour(
            r_grid,
            psi_grid,
            dpsi_clean,
            levels=[0.0],
            colors=[theme.violet],
            linewidths=1.35,
            linestyles="-.",
            alpha=0.88,
            zorder=3.5,
        )

    # Sample trajectories
    for idx, (r0, psi0) in enumerate(sample_ics):
        rs, psis = _integrate_polar_sample(r0, psi0, r_grid, psi_grid, dr, dpsi)
        col = theme.cyan if idx % 2 == 0 else theme.emerald
        plot_glow_line(ax, rs, psis, col, theme, lw=1.55, alpha=0.82, zorder=4.5)

    # Fixed points and limit cycles
    analysis = find_fixed_points_and_cycles(vdef, world, r_bounds=(2.0, r_max))
    lc = analysis.get("limit_cycle")
    if lc is not None and lc.get("r_orbit") is not None:
        r_orb = np.asarray(lc["r_orbit"], dtype=float)
        psi_orb = np.asarray(lc["psi_orbit"], dtype=float)
        # Split across +-pi jumps if trajectory crosses cylinder seam
        jumps = np.where(np.abs(np.diff(psi_orb)) > math.pi)[0] + 1
        r_chunks = np.split(r_orb, jumps)
        p_chunks = np.split(psi_orb, jumps)
        for ci, (rc, pc) in enumerate(zip(r_chunks, p_chunks)):
            if len(rc) >= 2:
                plot_glow_line(
                    ax,
                    rc,
                    pc,
                    theme.coral if not theme.is_dark else theme.gold,
                    theme,
                    lw=2.5,
                    label="Limit cycle" if ci == 0 else None,
                    zorder=6.5,
                )
        # Also mirror symmetric branch (-psi) if orbit stays on one side (CW vs CCW circling!)
        if len(psi_orb) > 0 and (np.all(psi_orb > 0) or np.all(psi_orb < 0)):
            for rc, pc in zip(r_chunks, p_chunks):
                if len(rc) >= 2:
                    plot_glow_line(
                        ax,
                        rc,
                        -pc,
                        theme.coral if not theme.is_dark else theme.gold,
                        theme,
                        lw=2.5,
                        alpha=0.92,
                        zorder=6.5,
                    )

    for fp in analysis.get("fixed_points", []):
        fp_r = float(fp.get("r", 0.0))
        fp_psi = float(fp.get("psi", 0.0))
        fp_type = str(fp.get("type") or fp.get("classification") or fp.get("stability") or "saddle")
        if r_grid[0] <= fp_r <= r_grid[-1] and -math.pi <= fp_psi <= math.pi:
            plot_strogatz_equilibrium(ax, fp_r, fp_psi, fp_type, theme, size=56, zorder=8)

    if show_legend:
        handles = [
            Line2D([0], [0], color=theme.amber, lw=1.4, ls="--", label=r"$\dot{r}=0$ nullcline"),
            Line2D([0], [0], color=theme.violet, lw=1.4, ls="-.", label=r"$\dot{\psi}=0$ nullcline"),
        ]
        if lc is not None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=theme.coral if not theme.is_dark else theme.gold,
                    lw=2.3,
                    label=f"Limit cycle ($r^*\\approx{float(lc.get('r_mean', 18.4)):.1f}$)",
                )
            )
        leg = ax.legend(handles=handles, loc="upper right", framealpha=0.9)
        style_legend(leg, theme)

    pf["analysis"] = analysis
    return pf


def _orient_grid_2d(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Ensure a 2D field matches Matplotlib's (ny, nx) row-major convention."""
    a = np.asarray(arr)
    if a.shape == (ny, nx):
        return a
    if a.shape == (nx, ny):
        return a.T
    return a


# ---------------------------------------------------------------------------
# ACT 1 — Polar Phase Portraits & 1D Bifurcation Poster
# ---------------------------------------------------------------------------


def build_act1_poster(theme_name: str = "cosmic", aspect: str = "16:9", dpi: int | None = None) -> Figure:
    from braitenberg.dynamics.phase_plane import bifurcation_sweep

    theme = get_theme(theme_name)
    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _ = _new_figure(aspect, "poster", theme, dpi)
        compact = aspect != "16:9"
        add_poster_header(
            fig,
            theme,
            "Act I · Symmetry-Reduced Cylindrical Phase Portraits",
            (
                r"Around one isotropic lamp, $SO(2)$ invariance reduces $(x,y,\theta)$ to "
                r"$\dot{r} = -v\cos\psi,\; \dot{\psi} = \frac{v}{r}\sin\psi - \omega$"
                if not compact
                else r"$SO(2)$ reduction: $\dot{r} = -v\cos\psi,\; \dot{\psi} = \frac{v}{r}\sin\psi - \omega$"
            ),
            kicker="ACT I · CYLINDRICAL PHASE SPACE ON R+ × S1",
            compact=compact,
        )
        axes = _poster_axes_4(fig, aspect)
        world_one = _world("one-light.yaml")

        # Panel A: Vehicle 2a Fear
        _draw_polar_portrait(
            axes[0],
            _vehicle_def("vehicle2a-fear.yaml"),
            world_one,
            theme,
            tag="A",
            title="Vehicle 2a (Fear) · Repulsive Deflection",
            sample_ics=[(8.0, 0.35), (8.0, -0.35), (14.0, 0.9), (14.0, -0.9), (22.0, 0.2)],
        )
        axes[0].text(
            31.0,
            2.45,
            r"Escape $\psi \to \pm\pi$",
            color=theme.cyan,
            fontsize=8.3,
            fontweight="bold",
            path_effects=text_halo(theme),
            zorder=9,
        )

        # Panel B: Vehicle 3a Love
        _draw_polar_portrait(
            axes[1],
            _vehicle_def("vehicle3a-love.yaml"),
            world_one,
            theme,
            tag="B",
            title="Vehicle 3a (Love) · Stable Arrest Attractor",
            sample_ics=[(38.0, 2.2), (38.0, -2.2), (28.0, 1.2), (28.0, -1.2), (18.0, 0.6)],
        )
        axes[1].text(
            18.5,
            0.38,
            r"Attractor $\psi \to 0,\; v \to 0$",
            color=theme.emerald,
            fontsize=8.3,
            fontweight="bold",
            path_effects=text_halo(theme),
            zorder=9,
        )

        # Panel C: Vehicle 4a Orbiter
        _draw_polar_portrait(
            axes[2],
            _vehicle_def("vehicle4a-orbiter.yaml"),
            world_one,
            theme,
            tag="C",
            title="Vehicle 4a (Orbiter) · Supercritical Limit Cycle",
            sample_ics=[(5.0, 0.5), (5.0, -0.5), (39.0, 0.55), (39.0, -0.55), (32.0, 2.3)],
            show_legend=True,
        )

        # Panel D: 1D Bifurcation Curve
        ax_bif = axes[3]
        style_axes(
            ax_bif,
            theme,
            tag="D",
            title="Vehicle 4a · Orbital Radius Bifurcation Sweep",
            xlabel=r"Sensor peak preference $I_{\mathrm{peak}}$ (`response.at`)",
            ylabel=r"Asymptotic radius $r^*$ (arena units)",
        )
        sweep = bifurcation_sweep(
            _vehicle_def("vehicle4a-orbiter.yaml"),
            world_one,
            param="peak_at",
            values=np.linspace(1.2, 9.2, 26),
        )
        vals = np.asarray(sweep["values"], dtype=float)
        r_min = np.asarray(sweep["r_min"], dtype=float)
        r_max = np.asarray(sweep["r_max"], dtype=float)
        r_mean = np.asarray(sweep["r_mean"], dtype=float)

        # Theoretical analytical sensor inversion curve: I(r) = I0 / (1 + (r/falloff)^2) = peak_at
        i_dense = np.linspace(1.2, 9.5, 200)
        r_theory = 15.0 * np.sqrt(np.maximum(10.0 / i_dense - 1.0, 0.0))
        ax_bif.plot(
            i_dense,
            r_theory,
            color=theme.fg_dim,
            lw=1.1,
            ls="--",
            alpha=0.75,
            label=r"Iso-intensity $I(r)=I_{\mathrm{peak}}$",
            zorder=3,
        )

        valid = np.isfinite(r_mean)
        if np.any(valid):
            ax_bif.fill_between(
                vals[valid],
                r_min[valid],
                r_max[valid],
                color=theme.cyan,
                alpha=0.20,
                label=r"Orbital band $[r_{\min}, r_{\max}]$",
                zorder=4,
            )
            plot_glow_line(
                ax_bif,
                vals[valid],
                r_mean[valid],
                theme.cyan,
                theme,
                lw=2.4,
                label=r"Stable limit cycle $\langle r \rangle$",
                zorder=5,
            )

        # Canonical Vehicle 4a operating point at=4.0
        r_canon = float(np.interp(4.0, vals[valid], r_mean[valid])) if np.any(valid) else 18.4
        ax_bif.axvline(4.0, color=theme.amber, lw=1.1, ls=":", alpha=0.8, zorder=4)
        plot_glow_scatter(
            ax_bif,
            [4.0],
            [r_canon],
            theme.gold if theme.is_dark else theme.coral,
            theme,
            size=62,
            label=f"Canonical 4a ($I_{{\\mathrm{{peak}}}}=4,\\; r^*\\approx{r_canon:.1f}$)",
            zorder=7,
        )
        ax_bif.set_xlim(float(vals.min()), float(vals.max()))
        ax_bif.set_ylim(0.0, max(46.0, float(np.nanmax(r_max)) * 1.08 if np.any(valid) else 46.0))
        leg = ax_bif.legend(loc="upper right")
        style_legend(leg, theme)

        _add_footer(
            fig,
            theme,
            "Strogatz convention: filled disc = stable fixed point · open circle = unstable · crossed circle = saddle",
            aspect,
        )
        return fig


# ---------------------------------------------------------------------------
# ACT 2 — Lagrangian Coherent Structures, FTLE Ridges & Fractal Basins
# ---------------------------------------------------------------------------


def build_act2_poster(theme_name: str = "cosmic", aspect: str = "16:9", dpi: int | None = None) -> Figure:
    from braitenberg.dynamics.lagrangian import (
        compute_ftle_and_basins,
        poincare_section,
        simulate_swarm_ensemble,
    )

    theme = get_theme(theme_name)
    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _ = _new_figure(aspect, "poster", theme, dpi)
        compact = aspect != "16:9"
        add_poster_header(
            fig,
            theme,
            "Act II · Lagrangian Chaos, FTLE Ridges & Separatrix Splitting",
            (
                r"Multiple lamps break $SO(2)$ symmetry; forward FTLE ridges $\sigma_T(x_0,y_0)$ "
                r"expose repelling invariant manifolds in $\mathbb{R}^2 \times S^1$"
                if not compact
                else r"Broken $SO(2)$ symmetry: forward FTLE ridges $\sigma_T$ & fractal capture basins"
            ),
            kicker="ACT II · MULTI-BODY SYMMETRY BREAKING & LAGRANGIAN COHERENT STRUCTURES",
            compact=compact,
        )
        axes = _poster_axes_4(fig, aspect)
        world_3 = _three_lamp_world()
        v4a = _vehicle_def("vehicle4a-orbiter.yaml")
        sources = _primary_sources(world_3, "light")

        lcs = compute_ftle_and_basins(v4a, world_3, nx=110, ny=110, horizon=11.0, dt=0.05)
        xg = np.asarray(lcs["x"], dtype=float)
        yg = np.asarray(lcs["y"], dtype=float)
        nx, ny = (xg.shape[1], xg.shape[0]) if xg.ndim == 2 else (len(xg), len(yg))
        x1d = xg[0, :] if xg.ndim == 2 else xg
        y1d = yg[:, 0] if yg.ndim == 2 else yg
        ftle = _orient_grid_2d(lcs["ftle"], len(x1d), len(y1d))
        basin = _orient_grid_2d(lcs["basin"], len(x1d), len(y1d))
        extent = [float(x1d[0]), float(x1d[-1]), float(y1d[0]), float(y1d[-1])]

        # Panel A: Forward FTLE Ridges
        ax_ftle = axes[0]
        style_axes(
            ax_ftle,
            theme,
            tag="A",
            title=r"Forward FTLE Ridges $\sigma_T(x_0, y_0)$",
            xlabel=r"Arena $x$",
            ylabel=r"Arena $y$",
            grid=False,
        )
        vmax = float(np.nanpercentile(ftle, 99.2)) if np.any(np.isfinite(ftle)) else 1.0
        vmin = float(np.nanpercentile(ftle, 5.0)) if np.any(np.isfinite(ftle)) else 0.0
        im = ax_ftle.imshow(
            ftle,
            origin="lower",
            extent=extent,
            cmap=ftle_cmap(theme),
            vmin=vmin,
            vmax=max(vmax, vmin + 1e-4),
            interpolation="bilinear",
            aspect="auto" if aspect == "9:16" else "equal",
            zorder=2,
        )
        for sx, sy in sources:
            plot_lamp_beacon(ax_ftle, sx, sy, theme, size=72, zorder=8)
        cbar = fig.colorbar(im, ax=ax_ftle, fraction=0.044, pad=0.025)
        cbar.set_label(r"$\sigma_T\;(\mathrm{s}^{-1})$", color=theme.fg_muted, fontsize=8.3)
        cbar.ax.tick_params(colors=theme.fg_dim, labelsize=7.5)
        cbar.outline.set_edgecolor(theme.spine)

        # Panel B: Capture Basins + FTLE relief
        ax_bas = axes[1]
        style_axes(
            ax_bas,
            theme,
            tag="B",
            title="Competing Lamp Basins & Ridge Skeleton",
            xlabel=r"Arena $x$",
            ylabel=r"Arena $y$",
            grid=False,
        )
        b_cols = basin_colors(theme, max(3, len(sources)))
        rgba_basin = np.zeros((*basin.shape, 4), dtype=float)
        base_unassigned = to_rgba(theme.panel_bg, 0.85)
        rgba_basin[:, :] = base_unassigned
        for bid in range(max(3, len(sources))):
            mask = basin == bid
            rgba_basin[mask] = to_rgba(b_cols[bid % len(b_cols)], 0.52 if theme.is_dark else 0.36)
        # Modulate brightness by normalized FTLE so ridges glow sharply inside basin map
        ftle_norm = np.clip((ftle - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        ridge_tint = np.array(to_rgba(theme.gold if theme.is_dark else theme.fg))
        blend = (ftle_norm**2.1)[..., None] * 0.72
        rgba_basin = np.clip(rgba_basin * (1.0 - blend) + ridge_tint * blend, 0.0, 1.0)
        rgba_basin[..., 3] = 0.94

        ax_bas.imshow(
            rgba_basin,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            aspect="auto" if aspect == "9:16" else "equal",
            zorder=2,
        )
        for i, (sx, sy) in enumerate(sources):
            plot_lamp_beacon(ax_bas, sx, sy, theme, color=b_cols[i % len(b_cols)], size=72, zorder=8)
        entropy = float(lcs.get("basin_entropy", 0.0))
        ax_bas.text(
            0.03,
            0.95,
            f"Basin entropy $S_b = {entropy:.2f}$ bits",
            transform=ax_bas.transAxes,
            va="top",
            ha="left",
            color=theme.fg,
            fontsize=8.2,
            fontweight="bold",
            path_effects=text_halo(theme),
            zorder=9,
        )

        # Panel C: Ensemble Trajectories Splitting Across Saddle Manifolds
        ax_ens = axes[2]
        style_axes(
            ax_ens,
            theme,
            tag="C",
            title="Swarm Separatrix Splitting Across Saddle",
            xlabel=r"Arena $x$",
            ylabel=r"Arena $y$",
        )
        ax_ens.imshow(
            ftle,
            origin="lower",
            extent=extent,
            cmap=ftle_cmap(theme),
            vmin=vmin,
            vmax=max(vmax, vmin + 1e-4),
            alpha=0.35 if theme.is_dark else 0.24,
            aspect="auto" if aspect == "9:16" else "equal",
            zorder=1,
        )
        swarm = simulate_swarm_ensemble(v4a, world_3, n_particles=90, steps=420, dt=0.05, seed=3)
        sx_arr = np.asarray(swarm["x"], dtype=float)
        sy_arr = np.asarray(swarm["y"], dtype=float)
        bids = np.asarray(swarm["basin_id"], dtype=int)
        p_colors = [
            b_cols[b % len(b_cols)] if b >= 0 else theme.violet
            for b in bids
        ]
        plot_comet_trails(
            ax_ens,
            sx_arr,
            sy_arr,
            p_colors,
            theme,
            max_lw=1.45,
            alpha=0.82,
            arena_size=(world_3.arena.width, world_3.arena.height),
            head_size=14.0,
            zorder=5,
        )
        for i, (sx, sy) in enumerate(sources):
            plot_lamp_beacon(ax_ens, sx, sy, theme, color=b_cols[i % len(b_cols)], size=68, zorder=8)
        ax_ens.set_xlim(0.0, world_3.arena.width)
        ax_ens.set_ylim(0.0, world_3.arena.height)

        # Panel D: Poincaré Section
        ax_poin = axes[3]
        style_axes(
            ax_poin,
            theme,
            tag="D",
            title=r"Poincaré Surface of Section $\Sigma:\; x = L_x/2$",
            xlabel=r"Section coordinate $y \in [0, L_y]$",
            ylabel=r"Crossing bearing $\psi$ (rad)",
        )
        psec = poincare_section(v4a, world_3, n_ic=54, steps=2600, dt=0.05, seed=0)
        coords = np.asarray(psec["coord"], dtype=float)
        psis = np.asarray(psec.get("psi", psec.get("heading", np.zeros_like(coords))), dtype=float)
        ic_ids = np.asarray(psec.get("ic_id", np.zeros_like(coords, dtype=int)), dtype=int)
        if len(coords) > 0:
            palette = [theme.cyan, theme.coral, theme.amber, theme.emerald, theme.violet, theme.indigo]
            pt_cols = [palette[int(k) % len(palette)] for k in ic_ids]
            plot_glow_scatter(
                ax_poin,
                coords,
                psis,
                pt_cols,
                theme,
                size=16.0,
                alpha=0.82,
                edgecolor="none",
                zorder=5,
            )
        ax_poin.set_xlim(0.0, world_3.arena.height)
        _format_psi_axis(ax_poin)

        _add_footer(
            fig,
            theme,
            r"Unwrapped deformation gradient $\Delta = (D\Phi_T)^\top(D\Phi_T)$ eliminates periodic wrap-edge artifacts",
            aspect,
        )
        return fig


# ---------------------------------------------------------------------------
# ACT 3 — Hysteresis Folds, Slow-Fast Relaxation & Synaptic Bifurcations
# ---------------------------------------------------------------------------


def _extract_neuro_episodes() -> dict[str, dict]:
    """Retrieve canonical Act III trajectories via neuro.py, ensuring full multi-cycle spans."""
    from braitenberg.dynamics.neuro import characterize_hysteresis_and_learning, record_episode

    summary = characterize_hysteresis_and_learning()
    moth_ep = record_episode(_scenario("vehicle5-moth.yaml"), steps=3200)
    forager_ep = record_episode(_scenario("vehicle5-forager.yaml"), steps=10500)
    wary_ep = record_episode(_scenario("vehicle7-wary-moth.yaml"), steps=9500)
    driver_ep = record_episode(_scenario("vehicle11-cautious-driver.yaml"), steps=10500)
    return {
        "summary": summary,
        "moth": moth_ep,
        "forager": forager_ep,
        "wary": wary_ep,
        "driver": driver_ep,
    }


def _nearest_lamp_distance(x: np.ndarray, y: np.ndarray, world: World) -> np.ndarray:
    sources = _primary_sources(world, "light")
    if not sources:
        return np.zeros_like(x)
    dists = []
    for sx, sy in sources:
        dx, dy = world.arena.delta(x - sx, y - sy)
        dists.append(np.hypot(dx, dy))
    return np.minimum.reduce(dists)


def build_act3_poster(theme_name: str = "cosmic", aspect: str = "16:9", dpi: int | None = None) -> Figure:
    theme = get_theme(theme_name)
    data = _extract_neuro_episodes()
    moth = data["moth"]
    forager = data["forager"]
    wary = data["wary"]
    driver = data["driver"]

    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _ = _new_figure(aspect, "poster", theme, dpi)
        compact = aspect != "16:9"
        add_poster_header(
            fig,
            theme,
            "Act III · Hysteresis Folds, Slow–Fast Cycles & Synaptic Bifurcations",
            (
                r"Internal latches, energy stores $E(t)$, and Mnemotrix/Ergotrix weights crossing "
                r"$w_c = 0.7$ expand phase-space dimensionality"
                if not compact
                else r"Latches, slow–fast energy $E(t)$, and plastic bifurcation across $w_c = 0.7$"
            ),
            kicker="ACT III · MEMORY AS PHASE-SPACE DIMENSIONALITY EXPANSION",
            compact=compact,
        )
        axes = _poster_axes_4(fig, aspect)

        # --- Panel A: Vehicle 5 Moth Fold / Latch Catastrophe ---
        ax_moth = axes[0]
        style_axes(
            ax_moth,
            theme,
            tag="A",
            title="Vehicle 5 (Moth) · Irreversible Latch Catastrophe",
            xlabel=r"Distance to nearest lamp $r$",
            ylabel=r"Radial velocity $\dot{r}$ (approach $<0$, flee $>0$)",
        )
        moth_world = _scenario("vehicle5-moth.yaml").world
        r_moth = _nearest_lamp_distance(np.asarray(moth["x"]), np.asarray(moth["y"]), moth_world)
        dt_m = float(moth["t"][1] - moth["t"][0]) if len(moth["t"]) > 1 else 0.01
        dr_moth = np.gradient(r_moth, dt_m)
        # Smooth finite-difference jitter slightly for clean phase curve
        kernel = np.ones(9) / 9.0
        dr_smooth = np.convolve(dr_moth, kernel, mode="same")
        burned = np.asarray(moth["units"].get("burned", np.zeros_like(r_moth)), dtype=float)
        burn_idx = int(np.argmax(burned > 0.5)) if np.any(burned > 0.5) else len(r_moth) // 2

        ax_moth.axhline(0.0, color=theme.fg_dim, lw=0.8, ls=":", alpha=0.65)
        ax_moth.axvline(8.0, color=theme.coral, lw=1.1, ls="--", alpha=0.75, label=r"Pain threshold ($r\approx 8$)")
        plot_glow_line(
            ax_moth,
            r_moth[:burn_idx],
            dr_smooth[:burn_idx],
            theme.cyan,
            theme,
            lw=2.1,
            label="Pre-burn attraction (`burned = 0`)",
            zorder=5,
        )
        plot_glow_line(
            ax_moth,
            r_moth[burn_idx : min(len(r_moth), burn_idx + 1200)],
            dr_smooth[burn_idx : min(len(r_moth), burn_idx + 1200)],
            theme.coral,
            theme,
            lw=2.1,
            label="Post-burn repulsion (`burned = 1`)",
            zorder=6,
        )
        plot_glow_scatter(
            ax_moth,
            [r_moth[burn_idx]],
            [dr_smooth[burn_idx]],
            theme.gold if theme.is_dark else theme.coral,
            theme,
            size=68,
            label="Catastrophe latch jump",
            zorder=8,
        )
        ax_moth.set_xlim(0.0, min(42.0, float(np.nanmax(r_moth)) * 1.02))
        ax_moth.set_ylim(-6.8, 6.8)
        leg_a = ax_moth.legend(loc="lower right")
        style_legend(leg_a, theme)

        # --- Panel B: Vehicle 5 Forager Slow-Fast Relaxation Cycle ---
        ax_for = axes[1]
        style_axes(
            ax_for,
            theme,
            tag="B",
            title="Vehicle 5 (Forager) · Slow–Fast $(r, E)$ Relaxation Cycle",
            xlabel=r"Distance to lamp $r$",
            ylabel=r"Internal energy $E(t)$",
        )
        r_for = np.asarray(forager["r_primary"], dtype=float)
        e_for = np.asarray(forager["energy"], dtype=float)
        full_u = np.asarray(forager["units"].get("full", np.zeros_like(r_for)), dtype=float)

        ax_for.axhline(90.0, color=theme.emerald, lw=1.0, ls="--", alpha=0.75, label=r"Charged fold $E_{\mathrm{high}}=90$")
        ax_for.axhline(30.0, color=theme.amber, lw=1.0, ls="--", alpha=0.75, label=r"Hungry fold $E_{\mathrm{low}}=30$")

        # Plot seeking (full==0) vs exploring (full==1) segments cleanly
        changes = np.where(np.diff(full_u > 0.5))[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(r_for)]))
        drew_seek = drew_exp = False
        for s_i, e_i in zip(starts, ends):
            if e_i - s_i < 2:
                continue
            seg_end = min(len(r_for), e_i + 1)
            is_exploring = bool(full_u[s_i] > 0.5)
            col = theme.amber if is_exploring else theme.emerald
            lbl = None
            if is_exploring and not drew_exp:
                lbl = "Explore discharge (`full = 1`)"
                drew_exp = True
            elif not is_exploring and not drew_seek:
                lbl = "Phototaxis & recharge (`seeking`)"
                drew_seek = True
            plot_glow_line(ax_for, r_for[s_i:seg_end], e_for[s_i:seg_end], col, theme, lw=2.0, label=lbl, zorder=5)

        ax_for.set_xlim(0.0, max(45.0, float(np.nanmax(r_for)) * 1.05))
        ax_for.set_ylim(0.0, 105.0)
        leg_b = ax_for.legend(loc="lower left")
        style_legend(leg_b, theme)

        # --- Panel C: Vehicle 7 Wary Moth Mnemotrix Bifurcation at w_c = 0.7 ---
        ax_wary = axes[2]
        style_axes(
            ax_wary,
            theme,
            tag="C",
            title=r"Vehicle 7 (Wary Moth) · Mnemotrix Crossing $w_c = 0.7$",
            xlabel=r"Simulated time $t$ (s)",
            ylabel=r"Synaptic strength $w(t)$",
        )
        t_wary = np.asarray(wary["t"], dtype=float)
        strengths_w = wary["strengths"]
        w_red_key = next((k for k in strengths_w if "red" in k), next(iter(strengths_w), ""))
        w_white_key = next((k for k in strengths_w if "white" in k), "")
        w_red = np.asarray(strengths_w[w_red_key], dtype=float) if w_red_key else np.zeros_like(t_wary)
        w_white = np.asarray(strengths_w[w_white_key], dtype=float) if w_white_key else np.zeros_like(t_wary)

        ax_wary.axhspan(ACTIVE_STRENGTH, 1.03, color=theme.emerald, alpha=0.09, zorder=1)
        ax_wary.axhline(
            ACTIVE_STRENGTH,
            color=theme.gold if theme.is_dark else theme.coral,
            lw=1.5,
            ls="--",
            label=f"Critical threshold $w_c = {ACTIVE_STRENGTH}$",
            zorder=4,
        )
        plot_glow_line(
            ax_wary,
            t_wary,
            w_red,
            theme.coral,
            theme,
            lw=2.3,
            label="`sees_red -> danger` (hot lamp)",
            zorder=6,
        )
        plot_glow_line(
            ax_wary,
            t_wary,
            w_white,
            theme.cyan,
            theme,
            lw=1.8,
            label="`sees_white -> danger` (safe lamp)",
            zorder=5,
        )
        cross_idx = np.where(w_red > ACTIVE_STRENGTH)[0]
        if len(cross_idx) > 0:
            ci = int(cross_idx[0])
            plot_glow_scatter(
                ax_wary,
                [t_wary[ci]],
                [w_red[ci]],
                theme.gold if theme.is_dark else theme.fg,
                theme,
                size=62,
                zorder=8,
            )
            ax_wary.annotate(
                r"Bifurcation: flees red at $r\approx 26$",
                xy=(t_wary[ci], w_red[ci]),
                xytext=(min(t_wary[-1] * 0.42, t_wary[ci] + 6.0), 0.86),
                color=theme.fg,
                fontsize=8.0,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": theme.fg_muted, "lw": 1.0},
                path_effects=text_halo(theme),
                zorder=9,
            )
        ax_wary.set_xlim(float(t_wary[0]), float(t_wary[-1]))
        ax_wary.set_ylim(-0.03, 1.05)
        leg_c = ax_wary.legend(loc="lower right")
        style_legend(leg_c, theme)

        # --- Panel D: Vehicle 11 Cautious Driver Ergotrix Sequence Bifurcation ---
        ax_drv = axes[3]
        style_axes(
            ax_drv,
            theme,
            tag="D",
            title=r"Vehicle 11 (Cautious Driver) · Ergotrix Sequence Learning",
            xlabel=r"Simulated time $t$ (s)",
            ylabel=r"Ergotrix weight $w(t)$",
        )
        t_drv = np.asarray(driver["t"], dtype=float)
        s_drv = driver["strengths"]
        k_yr = next((k for k in s_drv if "yellow" in k and "expects_red" in k), "")
        k_gy = next((k for k in s_drv if "green" in k and "expects_yellow" in k), "")
        k_gr = next((k for k in s_drv if "green" in k and "expects_red" in k), "")

        ax_drv.axhspan(ACTIVE_STRENGTH, 1.03, color=theme.violet, alpha=0.09, zorder=1)
        ax_drv.axhline(
            ACTIVE_STRENGTH,
            color=theme.gold if theme.is_dark else theme.coral,
            lw=1.5,
            ls="--",
            label=f"Active wire gate $w_c = {ACTIVE_STRENGTH}$",
            zorder=4,
        )
        if k_yr:
            plot_glow_line(
                ax_drv,
                t_drv,
                s_drv[k_yr],
                theme.amber,
                theme,
                lw=2.3,
                label="`sees_yellow -> expects_red`",
                zorder=6,
            )
        if k_gy:
            plot_glow_line(
                ax_drv,
                t_drv,
                s_drv[k_gy],
                theme.emerald,
                theme,
                lw=1.8,
                ls="-",
                alpha=0.82,
                label="`sees_green -> expects_yellow`",
                zorder=5,
            )
        if k_gr:
            plot_glow_line(
                ax_drv,
                t_drv,
                s_drv[k_gr],
                theme.violet,
                theme,
                lw=1.5,
                ls=":",
                alpha=0.80,
                label="`sees_green -> expects_red`",
                zorder=4.5,
            )
        ax_drv.set_xlim(float(t_drv[0]), float(t_drv[-1]))
        ax_drv.set_ylim(-0.03, 1.05)
        leg_d = ax_drv.legend(loc="lower right")
        style_legend(leg_d, theme)

        _add_footer(
            fig,
            theme,
            "Piecewise-smooth hybrid dynamics: signal transmission switches strictly above ACTIVE_STRENGTH = 0.7",
            aspect,
        )
        return fig


# ---------------------------------------------------------------------------
# ACT 4 — Homeostatic Criticality & Complex Ring Order Parameter Z(t)
# ---------------------------------------------------------------------------


def _compute_ring_order_param(units_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute cortical matrix A(8, T) and Kuramoto complex order parameter Z(t) = R(t)exp(i*Phi(t))."""
    ideas = np.vstack([np.asarray(units_dict[f"idea_{k}"], dtype=float) for k in range(8)])  # (8, T)
    angles = np.exp(1j * 2.0 * math.pi * np.arange(8)[:, None] / 8.0)
    z = np.mean(ideas * angles, axis=0)
    return ideas, z, np.abs(z), np.angle(z)


def build_act4_poster(theme_name: str = "cosmic", aspect: str = "16:9", dpi: int | None = None) -> Figure:
    from braitenberg.dynamics.neuro import analyze_daydreamer_criticality, record_episode

    theme = get_theme(theme_name)
    crit = analyze_daydreamer_criticality(steps=1200, seed=0)
    ep_dream = record_episode(_scenario("vehicle12-daydreamer.yaml"), steps=1600)

    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _ = _new_figure(aspect, "poster", theme, dpi)
        compact = aspect != "16:9"
        add_poster_header(
            fig,
            theme,
            "Act IV · Homeostatic Criticality & Cortical Trains of Thought",
            (
                r"Vehicle 12's accumulating caudate controller balances an explosive 8-idea ring "
                r"between extinction and epileptic flood via $Z(t) = \frac{1}{8}\sum a_k e^{i2\pi k/8}$"
                if not compact
                else r"Caudate homeostatic criticality & complex ring order parameter $Z(t) = R(t)e^{i\Phi(t)}$"
            ),
            kicker="ACT IV · SELF-TUNED CRITICALITY IN ENDOGENOUS NEURAL RINGS",
            compact=compact,
        )
        axes = _poster_axes_4(fig, aspect)

        # --- Panel A: Caudate Criticality Phase Diagram ---
        ax_crit = axes[0]
        style_axes(
            ax_crit,
            theme,
            tag="A",
            title="Homeostatic Criticality Across Caudate Target",
            xlabel=r"Caudate target activity $\tau_{\mathrm{target}}$",
            ylabel=r"Mean active ideas $\langle A \rangle$ & wave coherence",
        )
        # Extract target sweep from crit if 1D/2D, or evaluate crisp target curve across [0.0 .. 7.5]
        v12_base = _vehicle_def("vehicle12-daydreamer.yaml")
        open_world = _world("open-field.yaml")
        targets_sweep = np.linspace(0.0, 7.2, 19)
        mean_act = np.zeros_like(targets_sweep)
        mean_r = np.zeros_like(targets_sweep)
        entropy_vals = np.zeros_like(targets_sweep)

        for idx, tgt in enumerate(targets_sweep):
            new_units = [
                replace(u, target=float(tgt)) if u.is_control else u
                for u in v12_base.units
            ]
            v_mod = replace(v12_base, units=new_units)
            sc = Scenario(
                name="sweep",
                world=open_world,
                vehicles=[Placement(v_mod, (50.0, 50.0), math.pi / 2, "v12")],
                simulation=SimulationSettings(dt=0.05, seed=0),
            )
            ep = record_episode(sc, steps=600)
            ideas_m, _, r_m, _ = _compute_ring_order_param(ep["units"])
            steady = ideas_m[:, 160:]
            mean_act[idx] = float(steady.sum(axis=0).mean())
            mean_r[idx] = float(r_m[160:].mean()) * 8.0  # scaled to [0, 8] axis for clear dual visual
            # State Shannon entropy over binary 8-bit patterns
            weights = (1 << np.arange(8))[:, None]
            codes = (steady.astype(int) * weights).sum(axis=0)
            _, counts = np.unique(codes, return_counts=True)
            probs = counts / counts.sum()
            entropy_vals[idx] = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

        ax_crit.axvspan(0.0, 0.25, color=theme.fg_dim, alpha=0.14, zorder=1)
        ax_crit.axvspan(0.35, 2.6, color=theme.emerald, alpha=0.11, zorder=1)
        ax_crit.axvspan(4.5, 7.2, color=theme.coral, alpha=0.10, zorder=1)
        plot_glow_line(ax_crit, targets_sweep, mean_act, theme.cyan, theme, lw=2.2, label=r"Mean activity $\langle A \rangle$", zorder=5)
        plot_glow_line(ax_crit, targets_sweep, entropy_vals, theme.emerald, theme, lw=2.1, label=r"Thought entropy $H$ (bits)", zorder=6)
        plot_glow_line(ax_crit, targets_sweep, mean_r, theme.violet, theme, lw=1.8, ls="--", label=r"Scaled coherence $8\langle |Z| \rangle$", zorder=5.5)
        ax_crit.axvline(1.0, color=theme.gold if theme.is_dark else theme.amber, lw=1.2, ls=":", zorder=4)
        ax_crit.text(1.12, 6.9, "Critical\nDaydreamer", color=theme.emerald, fontsize=7.8, fontweight="bold", path_effects=text_halo(theme))
        ax_crit.text(5.1, 6.9, "Epileptic\nFlood", color=theme.coral, fontsize=7.8, fontweight="bold", path_effects=text_halo(theme))
        ax_crit.set_xlim(0.0, 7.2)
        ax_crit.set_ylim(-0.2, 8.3)
        leg_a = ax_crit.legend(loc="center right")
        style_legend(leg_a, theme)

        # --- Panel B: Complex Ring Order Parameter Z(t) Polar Rose ---
        ax_z = axes[1]
        style_axes(
            ax_z,
            theme,
            tag="B",
            title=r"Cortical Mandala & Order Parameter $Z(t) = Re^{i\Phi}$",
            xlabel=r"$\mathrm{Re}\,Z(t)$",
            ylabel=r"$\mathrm{Im}\,Z(t)$",
        )
        ax_z.set_aspect("equal", adjustable="datalim")
        ideas_d, z_d, r_d, phi_d = _compute_ring_order_param(ep_dream["units"])

        # Draw outer cortical ring & k -> (k+1, k+3) star chords
        ring_rad = 0.42
        node_angles = 2.0 * math.pi * np.arange(8) / 8.0
        nx_ring = ring_rad * np.cos(node_angles)
        ny_ring = ring_rad * np.sin(node_angles)
        for k in range(8):
            for step_ahead in (1, 3):
                j = (k + step_ahead) % 8
                ax_z.plot(
                    [nx_ring[k], nx_ring[j]],
                    [ny_ring[k], ny_ring[j]],
                    color=theme.spine,
                    lw=0.85,
                    alpha=0.55,
                    zorder=2,
                )
        theta_c = np.linspace(0.0, 2.0 * math.pi, 180)
        ax_z.plot(ring_rad * np.cos(theta_c), ring_rad * np.sin(theta_c), color=theme.fg_dim, lw=0.8, ls=":", alpha=0.5, zorder=2)

        # Plot Z(t) rose trajectory colored by phase
        zx, zy = np.real(z_d[80:]), np.imag(z_d[80:])
        plot_glow_line(ax_z, zx, zy, theme.cyan, theme, lw=1.7, alpha=0.85, label="Daydreamer $Z(t)$", zorder=5)
        plot_glow_scatter(ax_z, [0.0], [0.0], theme.coral, theme, size=48, marker="x", linewidths=1.8, label="Silent / Flooded ($Z=0$)", zorder=7)

        # Cortical nodes around ring
        duty = ideas_d[:, 80:].mean(axis=1)
        node_cols = [phase_cmap(theme)(k / 8.0) for k in range(8)]
        plot_glow_scatter(ax_z, nx_ring, ny_ring, node_cols, theme, size=55 + 110 * duty, zorder=8)
        for k in range(8):
            lx, ly = (ring_rad + 0.085) * math.cos(node_angles[k]), (ring_rad + 0.085) * math.sin(node_angles[k])
            ax_z.text(lx, ly, f"$I_{k}$", color=theme.fg_muted, fontsize=7.6, ha="center", va="center", path_effects=text_halo(theme), zorder=9)

        ax_z.set_xlim(-0.56, 0.56)
        ax_z.set_ylim(-0.54, 0.54)
        leg_b = ax_z.legend(loc="lower right")
        style_legend(leg_b, theme)

        # --- Panel C: Spatiotemporal Raster & Caudate Threshold Breathing ---
        ax_ras = axes[2]
        style_axes(
            ax_ras,
            theme,
            tag="C",
            title="Idea Cascade & Caudate Threshold Breathing",
            xlabel=r"Simulated time $t$ (s)",
            ylabel="Active `idea_k` & $\\theta_{\\mathrm{eff}}(t)$",
        )
        t_d = np.asarray(ep_dream["t"], dtype=float)
        sl = slice(0, min(len(t_d), 900))  # first 45s crisp detail
        for k in range(8):
            active_t = np.where(ideas_d[k, sl] > 0.5, k + 0.5, np.nan)
            col = phase_cmap(theme)(k / 8.0)
            plot_glow_line(ax_ras, t_d[sl], active_t, col, theme, lw=3.2, alpha=0.92, zorder=5)

        eff_dict = ep_dream.get("effective_thresholds", {})
        eff_trace = np.asarray(
            eff_dict.get("idea_0", np.full_like(t_d, 2.0)),
            dtype=float,
        )
        # Overlay effective threshold scaled clearly onto top band or direct value
        plot_glow_line(
            ax_ras,
            t_d[sl],
            eff_trace[sl],
            theme.gold if theme.is_dark else theme.fg,
            theme,
            lw=1.8,
            ls="-",
            label=r"Effective threshold $\theta_{\mathrm{eff}}(t)$",
            zorder=7,
        )
        ax_ras.set_xlim(float(t_d[sl][0]), float(t_d[sl][-1]))
        ax_ras.set_ylim(-0.3, 8.4)
        leg_c = ax_ras.legend(loc="upper right")
        style_legend(leg_c, theme)

        # --- Panel D: Geometric Thought-Loop Tracks Across Seeds ---
        ax_trk = axes[3]
        style_axes(
            ax_trk,
            theme,
            tag="D",
            title="Endogenous Geometric Thought Loops in the Dark",
            xlabel=r"Unwrapped $x$ (arena units)",
            ylabel=r"Unwrapped $y$ (arena units)",
        )
        ax_trk.set_aspect("equal", adjustable="datalim")
        seed_specs = [
            (0, theme.cyan, "Seed 0 (periodic loop)"),
            (1, theme.coral, "Seed 1 (stuttering aperiodic)"),
            (2, theme.emerald, "Seed 2 (symmetric rosette)"),
        ]
        for sd, col, lbl in seed_specs:
            sc_sd = Scenario(
                name=f"s{sd}",
                world=open_world,
                vehicles=[Placement(v12_base, (0.0, 0.0), math.pi / 2, f"s{sd}")],
                simulation=SimulationSettings(dt=0.05, seed=sd),
            )
            ep_sd = record_episode(sc_sd, steps=1500)
            ux = np.asarray(ep_sd["ux"], dtype=float) - float(ep_sd["ux"][0])
            uy = np.asarray(ep_sd["uy"], dtype=float) - float(ep_sd["uy"][0])
            # Offset origins horizontally so all 3 geometric signatures display side-by-side cleanly!
            shift = (sd - 1) * 26.0
            plot_glow_line(ax_trk, ux + shift, uy, col, theme, lw=1.85, alpha=0.90, label=lbl, zorder=5 + sd * 0.2)
            plot_glow_scatter(ax_trk, [ux[-1] + shift], [uy[-1]], col, theme, size=32, zorder=7)

        leg_d = ax_trk.legend(loc="upper right")
        style_legend(leg_d, theme)

        _add_footer(
            fig,
            theme,
            "Zero external stimulus & zero stochastic noise: deterministic jitter order alone sculptures macroscopic trajectories",
            aspect,
        )
        return fig


# ---------------------------------------------------------------------------
# MASTER ATLAS — 4K 2x2 Synthesis Plate
# ---------------------------------------------------------------------------


def build_atlas_poster(theme_name: str = "cosmic", aspect: str = "16:9", dpi: int | None = None) -> Figure:
    from braitenberg.dynamics.lagrangian import compute_ftle_and_basins, simulate_swarm_ensemble
    from braitenberg.dynamics.neuro import record_episode

    theme = get_theme(theme_name)
    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _ = _new_figure(aspect, "poster", theme, dpi)
        compact = aspect != "16:9"
        add_poster_header(
            fig,
            theme,
            "Synthetic Psychology as Nonlinear Dynamics",
            (
                "After Valentino Braitenberg & Steven Strogatz — Cylindrical Limit Cycles, "
                "Lagrangian FTLE Ridges, Synaptic Bifurcations & Cortical Criticality"
                if not compact
                else "After Braitenberg & Strogatz — Limit Cycles, FTLE Chaos, Plasticity & Criticality"
            ),
            kicker="MASTER SYNTHESIS ATLAS · VEHICLES 1–12 IN PHASE SPACE",
            compact=compact,
        )
        axes = _poster_axes_4(fig, aspect)

        # Panel I: Act I Vehicle 4a Limit Cycle
        _draw_polar_portrait(
            axes[0],
            _vehicle_def("vehicle4a-orbiter.yaml"),
            _world("one-light.yaml"),
            theme,
            tag="I",
            title="SO(2) Polar Limit Cycle · Vehicle 4a Orbiter",
            sample_ics=[(4.5, 0.5), (4.5, -0.5), (39.0, 0.5), (39.0, -0.5), (30.0, 2.2)],
            show_legend=True,
        )

        # Panel II: Act II 3-Lamp Forward FTLE + Swarm Splitting
        ax_ii = axes[1]
        world_3 = _three_lamp_world()
        v4a = _vehicle_def("vehicle4a-orbiter.yaml")
        sources = _primary_sources(world_3, "light")
        lcs = compute_ftle_and_basins(v4a, world_3, nx=115, ny=115, horizon=11.5, dt=0.05)
        xg = np.asarray(lcs["x"], dtype=float)
        yg = np.asarray(lcs["y"], dtype=float)
        x1d = xg[0, :] if xg.ndim == 2 else xg
        y1d = yg[:, 0] if yg.ndim == 2 else yg
        ftle = _orient_grid_2d(lcs["ftle"], len(x1d), len(y1d))
        vmin, vmax = float(np.nanpercentile(ftle, 5.0)), float(np.nanpercentile(ftle, 99.2))

        style_axes(
            ax_ii,
            theme,
            tag="II",
            title=r"Three-Lamp FTLE Ridges $\sigma_T$ & Swarm Splitting",
            xlabel=r"Arena $x$",
            ylabel=r"Arena $y$",
            grid=False,
        )
        ax_ii.imshow(
            ftle,
            origin="lower",
            extent=[float(x1d[0]), float(x1d[-1]), float(y1d[0]), float(y1d[-1])],
            cmap=ftle_cmap(theme),
            vmin=vmin,
            vmax=max(vmax, vmin + 1e-4),
            interpolation="bilinear",
            aspect="auto" if aspect == "9:16" else "equal",
            zorder=2,
        )
        swarm = simulate_swarm_ensemble(v4a, world_3, n_particles=75, steps=360, dt=0.05, seed=2)
        b_cols = basin_colors(theme, max(3, len(sources)))
        p_cols = [b_cols[int(b) % len(b_cols)] if int(b) >= 0 else theme.violet for b in swarm["basin_id"]]
        plot_comet_trails(
            ax_ii,
            np.asarray(swarm["x"]),
            np.asarray(swarm["y"]),
            p_cols,
            theme,
            max_lw=1.35,
            alpha=0.76,
            arena_size=(world_3.arena.width, world_3.arena.height),
            head_size=13.0,
            zorder=5,
        )
        for sx, sy in sources:
            plot_lamp_beacon(ax_ii, sx, sy, theme, size=68, zorder=8)
        ax_ii.set_xlim(0.0, world_3.arena.width)
        ax_ii.set_ylim(0.0, world_3.arena.height)

        # Panel III: Act III Forager Slow-Fast Cycle + Synaptic Threshold Inset Concept
        ax_iii = axes[2]
        style_axes(
            ax_iii,
            theme,
            tag="III",
            title=r"Slow–Fast $(r, E)$ Relaxation & $w_c=0.7$ Plasticity",
            xlabel=r"Distance to lamp $r$",
            ylabel=r"Internal energy $E(t)$  /  $100 \cdot w_{\mathrm{red}}(t)$",
        )
        forager = record_episode(_scenario("vehicle5-forager.yaml"), steps=9800)
        wary = record_episode(_scenario("vehicle7-wary-moth.yaml"), steps=8500)
        r_for = np.asarray(forager["r_primary"], dtype=float)
        e_for = np.asarray(forager["energy"], dtype=float)
        plot_glow_line(
            ax_iii,
            r_for,
            e_for,
            theme.emerald,
            theme,
            lw=2.0,
            label="Forager $(r, E)$ limit cycle",
            zorder=5,
        )
        ax_iii.axhline(90.0, color=theme.emerald, lw=0.9, ls="--", alpha=0.65)
        ax_iii.axhline(30.0, color=theme.amber, lw=0.9, ls="--", alpha=0.65)
        # Overlay Wary Moth trajectory in (r_red, 100 * w_red) to unite slow-fast & synaptic memory!
        wary_world = _scenario("vehicle7-wary-moth.yaml").world
        dx_r, dy_r = wary_world.arena.delta(np.asarray(wary["x"]) - 70.0, np.asarray(wary["y"]) - 50.0)
        r_red = np.hypot(dx_r, dy_r)
        w_red_key = next((k for k in wary["strengths"] if "red" in k), "")
        if w_red_key:
            w_scaled = 100.0 * np.asarray(wary["strengths"][w_red_key], dtype=float)
            plot_glow_line(
                ax_iii,
                r_red,
                w_scaled,
                theme.coral,
                theme,
                lw=1.85,
                alpha=0.88,
                label=r"Wary Moth $100\,w_{\mathrm{red}}(r)$",
                zorder=6,
            )
            ax_iii.axhline(70.0, color=theme.gold if theme.is_dark else theme.coral, lw=1.2, ls=":", label=r"Synaptic gate $w_c = 0.7$")
        ax_iii.set_xlim(0.0, 52.0)
        ax_iii.set_ylim(0.0, 105.0)
        leg_iii = ax_iii.legend(loc="lower right")
        style_legend(leg_iii, theme)

        # Panel IV: Act IV Daydreamer Ring Order Parameter & Thought Loop
        ax_iv = axes[3]
        style_axes(
            ax_iv,
            theme,
            tag="IV",
            title=r"Homeostatic Criticality · Ring $Z(t)$ & Thought Geometry",
            xlabel=r"Normalized phase / arena coordinate",
            ylabel=r"Normalized coordinate",
        )
        ax_iv.set_aspect("equal", adjustable="datalim")
        ep_dream = record_episode(_scenario("vehicle12-daydreamer.yaml"), steps=1600)
        ideas_d, z_d, _, _ = _compute_ring_order_param(ep_dream["units"])
        ux = np.asarray(ep_dream["ux"], dtype=float)
        uy = np.asarray(ep_dream["uy"], dtype=float)
        ux_n = (ux - np.mean(ux)) / max(np.ptp(ux), 1.0) * 1.55 - 0.85
        uy_n = (uy - np.mean(uy)) / max(np.ptp(uy), 1.0) * 1.55
        plot_glow_line(ax_iv, ux_n, uy_n, theme.cyan, theme, lw=2.0, label="Daydreamer arena loop", zorder=5)

        # Right side of Panel IV: Cortical ring + Z(t) rose
        cx_shift = 0.92
        ring_r = 0.68
        ang = 2.0 * math.pi * np.arange(8) / 8.0
        rx, ry = cx_shift + ring_r * np.cos(ang), ring_r * np.sin(ang)
        for k in range(8):
            for step_ahead in (1, 3):
                j = (k + step_ahead) % 8
                ax_iv.plot([rx[k], rx[j]], [ry[k], ry[j]], color=theme.spine, lw=0.8, alpha=0.55, zorder=2)
        plot_glow_line(
            ax_iv,
            cx_shift + 1.45 * np.real(z_d[80:]),
            1.45 * np.imag(z_d[80:]),
            theme.violet,
            theme,
            lw=1.75,
            label=r"Order parameter $Z(t)$",
            zorder=6,
        )
        node_cols = [phase_cmap(theme)(k / 8.0) for k in range(8)]
        duty = ideas_d[:, 80:].mean(axis=1)
        plot_glow_scatter(ax_iv, rx, ry, node_cols, theme, size=48 + 90 * duty, zorder=8)
        ax_iv.set_xlim(-1.85, 1.85)
        ax_iv.set_ylim(-0.92, 0.92)
        leg_iv = ax_iv.legend(loc="lower left")
        style_legend(leg_iv, theme)

        _add_footer(
            fig,
            theme,
            "braitenberg.dynamics · From reactive taxis (Vehicles 1–4) to memory latches, Hebbian gates & homeostatic thought (Vehicles 5–12)",
            aspect,
        )
        return fig


# ---------------------------------------------------------------------------
# Public Poster Dispatcher
# ---------------------------------------------------------------------------


def render_posters(
    *,
    act: str = "all",
    theme: str = "cosmic",
    aspect: str = "16:9",
    out_dir: str | Path = ".",
    dpi: int | None = None,
    fmt: str = "png",
    suffix: str = "",
) -> list[Path]:
    """Render requested static poster(s) and return written file paths."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    acts = ["1", "2", "3", "4", "atlas"] if act == "all" else [act]
    builders: dict[str, Callable[[str, str, int | None], Figure]] = {
        "1": build_act1_poster,
        "2": build_act2_poster,
        "3": build_act3_poster,
        "4": build_act4_poster,
        "atlas": build_atlas_poster,
    }
    theme_obj = get_theme(theme)
    _, _, default_dpi = aspect_figsize(aspect, "poster")
    target_dpi = dpi if dpi is not None else default_dpi

    written: list[Path] = []
    for a in acts:
        fig = builders[a](theme, aspect, target_dpi)
        fname = f"{POSTER_FILENAMES[a]}{suffix}.{fmt}"
        dest = out_path / fname
        _save_figure(fig, dest, theme_obj, target_dpi)
        written.append(dest)
    return written


# ---------------------------------------------------------------------------
# Animation Scene Precomputation & Frame Builders
# ---------------------------------------------------------------------------

_SCENE_CACHE: dict[str, dict] = {}


def _anim_layout_two_panel(fig: Figure, aspect: str) -> tuple[mpl.axes.Axes, mpl.axes.Axes]:
    """Build synchronized 2-panel layout cleanly tailored to 16:9, 1:1, and 9:16."""
    if aspect == "9:16":
        gs = fig.add_gridspec(
            2,
            1,
            left=0.125,
            right=0.94,
            bottom=0.055,
            top=0.885,
            height_ratios=[1.12, 1.0],
            hspace=0.24,
        )
        return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    if aspect == "1:1":
        gs = fig.add_gridspec(
            2,
            1,
            left=0.10,
            right=0.95,
            bottom=0.075,
            top=0.865,
            height_ratios=[1.08, 1.0],
            hspace=0.28,
        )
        return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    gs = fig.add_gridspec(
        1,
        2,
        left=0.058,
        right=0.965,
        bottom=0.11,
        top=0.825,
        width_ratios=[1.0, 1.12],
        wspace=0.21,
    )
    return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])


def _precompute_orbiter_scene() -> dict:
    if "orbiter" in _SCENE_CACHE:
        return _SCENE_CACHE["orbiter"]

    from braitenberg.dynamics.phase_plane import find_fixed_points_and_cycles, polar_coords, polar_field

    v4a = replace(_vehicle_def("vehicle4a-orbiter.yaml"), general=replace(_vehicle_def("vehicle4a-orbiter.yaml").general, noise_std=0.0))
    world = World(
        name="Centered one light",
        arena=Arena(width=100.0, height=100.0, edges="wrap"),
        stimuli={"light": Stimulus(name="light", sources=[Source(position=(50.0, 50.0), intensity=10.0, falloff=15.0)])},
    )
    rng = np.random.default_rng(42)
    n_inner, n_outer = 65, 85
    n_total = n_inner + n_outer
    r0_in = rng.uniform(3.2, 12.2, size=n_inner)
    r0_out = rng.uniform(26.5, 45.5, size=n_outer)
    r0 = np.concatenate([r0_in, r0_out])
    ang0 = rng.uniform(0.0, 2.0 * math.pi, size=n_total)
    psi_init = rng.uniform(-math.pi * 0.92, math.pi * 0.92, size=n_total)

    placements = []
    for i in range(n_total):
        x0 = 50.0 + r0[i] * math.cos(ang0[i])
        y0 = 50.0 + r0[i] * math.sin(ang0[i])
        bearing_to_src = math.atan2(50.0 - y0, 50.0 - x0)
        hdg = bearing_to_src - psi_init[i]
        placements.append(Placement(v4a, (x0, y0), hdg, f"p{i}"))

    dt = 0.04
    steps = 480
    sim = Simulation(Scenario("orbiter_swarm", world, placements, SimulationSettings(dt=dt, seed=0)))
    xs = np.empty((steps, n_total), dtype=float)
    ys = np.empty((steps, n_total), dtype=float)
    rs = np.empty((steps, n_total), dtype=float)
    psis = np.empty((steps, n_total), dtype=float)

    for t_i in range(steps):
        for j, v in enumerate(sim.vehicles):
            xs[t_i, j] = v.x
            ys[t_i, j] = v.y
            r_val, psi_val = polar_coords(v.x, v.y, v.heading, (50.0, 50.0), world.arena)
            rs[t_i, j] = float(r_val)
            psis[t_i, j] = float(psi_val)
        sim.step()

    r_grid = np.linspace(1.5, 48.0, 85)
    psi_grid = np.linspace(-math.pi, math.pi, 85)
    pf = polar_field(v4a, world, r_grid=r_grid, psi_grid=psi_grid)
    fpc = find_fixed_points_and_cycles(v4a, world, r_bounds=(2.0, 46.0))

    res = {
        "world": world,
        "xs": xs,
        "ys": ys,
        "rs": rs,
        "psis": psis,
        "is_inner": np.arange(n_total) < n_inner,
        "pf": pf,
        "fpc": fpc,
        "dt": dt,
    }
    _SCENE_CACHE["orbiter"] = res
    return res


def _setup_orbiter_anim(fig: Figure, theme: Theme, aspect: str, n_frames: int) -> Callable[[int], None]:
    data = _precompute_orbiter_scene()
    world: World = data["world"]
    xs, ys, rs, psis = data["xs"], data["ys"], data["rs"], data["psis"]
    is_inner = data["is_inner"]
    pf = data["pf"]
    fpc = data["fpc"]

    compact = aspect != "16:9"
    add_poster_header(
        fig,
        theme,
        "Vehicle 4a Orbiter · Supercritical Limit-Cycle Collapse",
        (
            r"150 vehicles launched inside & outside $r^* \approx 18.4$ spiral onto the invariant closed orbit"
            if not compact
            else r"150 Orbiters collapsing onto the $(r,\psi)$ limit cycle"
        ),
        kicker="SYNCHRONIZED ARENA & CYLINDRICAL PHASE PORTRAIT",
        compact=compact,
    )
    ax_arena, ax_phase = _anim_layout_two_panel(fig, aspect)

    # Static Arena setup
    style_axes(ax_arena, theme, tag="A", title="Physical Arena · 150 Comet Particles", xlabel=r"$x$", ylabel=r"$y$", grid=False)
    gx, gy = np.meshgrid(np.linspace(0, 100, 120), np.linspace(0, 100, 120))
    field = world.stimulus_at("light", gx, gy) / 10.0
    ax_arena.imshow(field, origin="lower", extent=[0, 100, 0, 100], cmap=field_cmap(theme), alpha=0.45 if theme.is_dark else 0.30, aspect="equal", zorder=1)
    lc = fpc.get("limit_cycle")
    r_star = float(lc.get("r_mean", 18.4)) if lc else 18.4
    ring_th = np.linspace(0, 2 * math.pi, 200)
    plot_glow_line(ax_arena, 50.0 + r_star * np.cos(ring_th), 50.0 + r_star * np.sin(ring_th), theme.gold if theme.is_dark else theme.coral, theme, lw=1.5, ls="--", alpha=0.72, zorder=3)
    plot_lamp_beacon(ax_arena, 50.0, 50.0, theme, size=90, zorder=8)
    ax_arena.set_xlim(0, 100)
    ax_arena.set_ylim(0, 100)

    # Static Phase Plane setup
    style_axes(ax_phase, theme, tag="B", title=r"Cylindrical Phase Plane $(r, \psi)$", xlabel=r"Distance $r$", ylabel=r"Bearing $\psi$ (rad)")
    r_grid, psi_grid = np.asarray(pf["r"]), np.asarray(pf["psi"])
    dr, dpsi = np.asarray(pf["dr"]), np.asarray(pf["dpsi"])
    ax_phase.streamplot(r_grid, psi_grid, dr, dpsi, color=theme.stream, linewidth=0.72, density=0.95, arrowsize=0.8, zorder=2)
    ax_phase.contour(r_grid, psi_grid, dr, levels=[0.0], colors=[theme.amber], linewidths=1.3, linestyles="--", alpha=0.85, zorder=3)
    dpsi_clean = np.where(np.abs(dpsi) < 8.0, dpsi, np.nan)
    ax_phase.contour(r_grid, psi_grid, dpsi_clean, levels=[0.0], colors=[theme.violet], linewidths=1.3, linestyles="-.", alpha=0.85, zorder=3)
    if lc and lc.get("r_orbit") is not None:
        ro, po = np.asarray(lc["r_orbit"]), np.asarray(lc["psi_orbit"])
        jumps = np.where(np.abs(np.diff(po)) > math.pi)[0] + 1
        for rc, pc in zip(np.split(ro, jumps), np.split(po, jumps)):
            if len(rc) >= 2:
                plot_glow_line(ax_phase, rc, pc, theme.gold if theme.is_dark else theme.coral, theme, lw=2.2, alpha=0.85, zorder=4)
                if np.all(po > 0) or np.all(po < 0):
                    plot_glow_line(ax_phase, rc, -pc, theme.gold if theme.is_dark else theme.coral, theme, lw=2.2, alpha=0.85, zorder=4)
    ax_phase.set_xlim(float(r_grid[0]), float(r_grid[-1]))
    _format_psi_axis(ax_phase)

    base_rgba = np.array([to_rgba(theme.coral if inn else theme.cyan) for inn in is_inner], dtype=float)
    comet_arena = CometArtist(ax_arena, theme, max_lw=1.9, head_size=18.0, arena_size=(100.0, 100.0), zorder=6)
    comet_phase = CometArtist(ax_phase, theme, max_lw=1.7, head_size=16.0, psi_wrap=True, zorder=6)
    badge = ax_arena.text(0.03, 0.96, "", transform=ax_arena.transAxes, va="top", ha="left", color=theme.fg, fontsize=8.4, fontweight="bold", path_effects=text_halo(theme), zorder=10)

    total_steps = xs.shape[0]
    tail = 26

    def update(f_idx: int) -> None:
        step = min(total_steps - 1, int(round(f_idx * (total_steps - 1) / max(1, n_frames - 1))))
        s0 = max(0, step - tail)
        comet_arena.update(xs[s0 : step + 1], ys[s0 : step + 1], base_rgba)
        comet_phase.update(rs[s0 : step + 1], psis[s0 : step + 1], base_rgba)
        spread = float(np.std(rs[step]))
        badge.set_text(f"t = {step * data['dt']:.1f} s   σ(r) = {spread:.2f}")

    return update


def _precompute_ftle_scene() -> dict:
    if "ftle_swarm" in _SCENE_CACHE:
        return _SCENE_CACHE["ftle_swarm"]

    from braitenberg.dynamics.lagrangian import compute_ftle_and_basins, simulate_swarm_ensemble

    world = _three_lamp_world()
    v4a = _vehicle_def("vehicle4a-orbiter.yaml")
    lcs = compute_ftle_and_basins(v4a, world, nx=115, ny=115, horizon=11.5, dt=0.05)
    swarm = simulate_swarm_ensemble(v4a, world, n_particles=300, steps=480, dt=0.04, seed=7, noise_scale=0.0)
    res = {"world": world, "lcs": lcs, "swarm": swarm, "dt": 0.04}
    _SCENE_CACHE["ftle_swarm"] = res
    return res


def _setup_ftle_anim(fig: Figure, theme: Theme, aspect: str, n_frames: int) -> Callable[[int], None]:
    data = _precompute_ftle_scene()
    world: World = data["world"]
    lcs = data["lcs"]
    swarm = data["swarm"]
    sources = _primary_sources(world, "light")

    compact = aspect != "16:9"
    add_poster_header(
        fig,
        theme,
        "Lagrangian Chaos · 300-Vehicle Swarm Shearing Along FTLE Ridges",
        (
            r"Bioluminescent ensemble splits across repelling Finite-Time Lyapunov ($\sigma_T$) separatrices into 3 lamp attractors"
            if not compact
            else r"300 vehicles shearing apart along forward FTLE ($\sigma_T$) ridges"
        ),
        kicker="THREE-LAMP SYMMETRY BREAKING & LAGRANGIAN SEPARATRICES",
        compact=compact,
    )
    ax_arena, ax_right = _anim_layout_two_panel(fig, aspect)

    xg, yg = np.asarray(lcs["x"], dtype=float), np.asarray(lcs["y"], dtype=float)
    x1d = xg[0, :] if xg.ndim == 2 else xg
    y1d = yg[:, 0] if yg.ndim == 2 else yg
    ftle = _orient_grid_2d(lcs["ftle"], len(x1d), len(y1d))
    vmin, vmax = float(np.nanpercentile(ftle, 5.0)), float(np.nanpercentile(ftle, 99.2))

    style_axes(ax_arena, theme, tag="A", title=r"300-Vehicle Swarm on Forward FTLE $\sigma_T$", xlabel=r"Arena $x$", ylabel=r"Arena $y$", grid=False)
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
    b_cols = basin_colors(theme, max(3, len(sources)))
    for i, (sx, sy) in enumerate(sources):
        plot_lamp_beacon(ax_arena, sx, sy, theme, color=b_cols[i % len(b_cols)], size=82, zorder=8)
    ax_arena.set_xlim(0, world.arena.width)
    ax_arena.set_ylim(0, world.arena.height)

    # Right panel: Nearest-Lamp Cylindrical Projection (r_nearest, heading) + Basin Divergence
    style_axes(
        ax_right,
        theme,
        tag="B",
        title="Attractor Partition in Nearest-Lamp Phase Space",
        xlabel=r"Distance to nearest lamp $r_{\min}$",
        ylabel=r"Heading $\theta \in [-\pi, \pi]$",
    )
    ax_right.axvline(18.4, color=theme.gold if theme.is_dark else theme.amber, lw=1.3, ls="--", alpha=0.78, label=r"Preferred orbit $r^*\approx 18.4$")
    ax_right.set_xlim(0.0, 50.0)
    _format_psi_axis(ax_right)
    leg = ax_right.legend(loc="upper right")
    style_legend(leg, theme)

    xs = np.asarray(swarm["x"], dtype=float)
    ys = np.asarray(swarm["y"], dtype=float)
    hdgs = np.arctan2(np.sin(np.asarray(swarm["heading"], dtype=float)), np.cos(np.asarray(swarm["heading"], dtype=float)))
    r_min_arr = _nearest_lamp_distance(xs, ys, world)
    bids = np.asarray(swarm["basin_id"], dtype=int)
    base_rgba = np.array(
        [to_rgba(b_cols[int(b) % len(b_cols)] if int(b) >= 0 else theme.violet) for b in bids],
        dtype=float,
    )

    comet_arena = CometArtist(ax_arena, theme, max_lw=1.7, head_size=15.0, arena_size=(world.arena.width, world.arena.height), zorder=6)
    comet_phase = CometArtist(ax_right, theme, max_lw=1.4, head_size=14.0, psi_wrap=True, zorder=6)
    badge = ax_arena.text(0.03, 0.96, "", transform=ax_arena.transAxes, va="top", ha="left", color=theme.fg, fontsize=8.4, fontweight="bold", path_effects=text_halo(theme), zorder=10)

    total_steps = xs.shape[0]
    tail = 24

    def update(f_idx: int) -> None:
        step = min(total_steps - 1, int(round(f_idx * (total_steps - 1) / max(1, n_frames - 1))))
        s0 = max(0, step - tail)
        comet_arena.update(xs[s0 : step + 1], ys[s0 : step + 1], base_rgba)
        comet_phase.update(r_min_arr[s0 : step + 1], hdgs[s0 : step + 1], base_rgba)
        t_now = float(swarm["t"][step]) if "t" in swarm else step * data["dt"]
        badge.set_text(f"t = {t_now:.1f} s   N = {xs.shape[1]} vehicles")

    return update


def _precompute_plasticity_scene() -> dict:
    if "plasticity" in _SCENE_CACHE:
        return _SCENE_CACHE["plasticity"]

    from braitenberg.dynamics.neuro import record_episode

    forager = record_episode(_scenario("vehicle5-forager.yaml"), steps=9600)
    wary = record_episode(_scenario("vehicle7-wary-moth.yaml"), steps=9600)
    res = {"forager": forager, "wary": wary}
    _SCENE_CACHE["plasticity"] = res
    return res


def _setup_plasticity_anim(fig: Figure, theme: Theme, aspect: str, n_frames: int) -> Callable[[int], None]:
    data = _precompute_plasticity_scene()
    forager = data["forager"]
    wary = data["wary"]

    compact = aspect != "16:9"
    add_poster_header(
        fig,
        theme,
        "Memory & Synaptic Bifurcation · Forager Cycle & Wary Moth Learning",
        (
            r"Left: Forager slow–fast $(r,E)$ relaxation oscillation · Right: Wary Moth crossing $w_c = 0.7$ into avoidance"
            if not compact
            else r"Slow–fast $(r,E)$ relaxation & Mnemotrix crossing $w_c = 0.7$"
        ),
        kicker="HYBRID HYSTERESIS & HEBBIAN VECTOR-FIELD BIFURCATION",
        compact=compact,
    )
    ax_for, ax_wary = _anim_layout_two_panel(fig, aspect)

    # Panel A: Forager Slow-Fast (r, E)
    r_for = np.asarray(forager["r_primary"], dtype=float)
    e_for = np.asarray(forager["energy"], dtype=float)
    style_axes(ax_for, theme, tag="A", title="Vehicle 5 Forager · Slow–Fast $(r, E)$ Cycle", xlabel=r"Distance to lamp $r$", ylabel=r"Internal energy $E(t)$")
    ax_for.plot(r_for, e_for, color=theme.fg_dim, lw=0.9, alpha=0.28, zorder=2)
    ax_for.axhline(90.0, color=theme.emerald, lw=1.1, ls="--", alpha=0.75, label=r"$E_{\mathrm{high}}=90$ (`full`)")
    ax_for.axhline(30.0, color=theme.amber, lw=1.1, ls="--", alpha=0.75, label=r"$E_{\mathrm{low}}=30$ (`hungry`)")
    ax_for.set_xlim(0.0, max(46.0, float(np.nanmax(r_for)) * 1.05))
    ax_for.set_ylim(0.0, 106.0)
    leg_a = ax_for.legend(loc="lower left")
    style_legend(leg_a, theme)

    # Panel B: Wary Moth Synaptic Weight + Distance to Red Lamp
    t_w = np.asarray(wary["t"], dtype=float)
    s_w = wary["strengths"]
    k_red = next((k for k in s_w if "red" in k), next(iter(s_w), ""))
    k_wht = next((k for k in s_w if "white" in k), "")
    w_red = np.asarray(s_w[k_red], dtype=float) if k_red else np.zeros_like(t_w)
    w_wht = np.asarray(s_w[k_wht], dtype=float) if k_wht else np.zeros_like(t_w)

    style_axes(ax_wary, theme, tag="B", title=r"Vehicle 7 Wary Moth · $w(t)$ Bifurcation at $w_c = 0.7$", xlabel=r"Simulated time $t$ (s)", ylabel=r"Synaptic weight $w(t)$")
    ax_wary.axhspan(ACTIVE_STRENGTH, 1.04, color=theme.emerald, alpha=0.09, zorder=1)
    ax_wary.axhline(ACTIVE_STRENGTH, color=theme.gold if theme.is_dark else theme.coral, lw=1.5, ls="--", label=f"Active threshold $w_c = {ACTIVE_STRENGTH}$", zorder=3)
    ax_wary.plot(t_w, w_red, color=theme.coral, lw=0.9, alpha=0.22, zorder=2)
    ax_wary.set_xlim(float(t_w[0]), float(t_w[-1]))
    ax_wary.set_ylim(-0.03, 1.05)

    line_red_halo = ax_wary.plot([], [], color=theme.coral, lw=6.5, alpha=0.16, zorder=4.8)[0] if theme.glow_enabled else None
    line_red = ax_wary.plot([], [], color=theme.coral, lw=2.4, label="`sees_red -> danger`", zorder=5)[0]
    line_wht = ax_wary.plot([], [], color=theme.cyan, lw=1.8, label="`sees_white -> danger`", zorder=4.5)[0]
    dot_red = ax_wary.scatter([], [], s=62, c=[theme.gold if theme.is_dark else theme.coral], edgecolors="#ffffff" if theme.is_dark else theme.fg, zorder=7)
    leg_b = ax_wary.legend(loc="lower right")
    style_legend(leg_b, theme)

    comet_for = CometArtist(ax_for, theme, max_lw=2.8, head_size=52.0, zorder=6)
    status_txt = ax_wary.text(0.03, 0.95, "", transform=ax_wary.transAxes, va="top", ha="left", color=theme.fg, fontsize=8.4, fontweight="bold", path_effects=text_halo(theme), zorder=9)

    n_steps = min(len(r_for), len(t_w))
    tail = 420
    full_u = np.asarray(forager["units"].get("full", np.zeros_like(r_for)), dtype=float)

    def update(f_idx: int) -> None:
        step = min(n_steps - 1, int(round(f_idx * (n_steps - 1) / max(1, n_frames - 1))))
        s0 = max(0, step - tail)
        # Subsample tail by 4 for crisp smooth comet without 400-segment density
        col_for = np.array([to_rgba(theme.amber if full_u[step] > 0.5 else theme.emerald)], dtype=float)
        comet_for.update(r_for[s0 : step + 1 : 4], e_for[s0 : step + 1 : 4], col_for)

        sub = slice(0, step + 1, 6)
        line_red.set_data(t_w[sub], w_red[sub])
        if line_red_halo is not None:
            line_red_halo.set_data(t_w[sub], w_red[sub])
        line_wht.set_data(t_w[sub], w_wht[sub])
        dot_red.set_offsets(np.array([[t_w[step], w_red[step]]]))
        regime = "AVOIDING RED (w > 0.7)" if w_red[step] > ACTIVE_STRENGTH else "NAIVE ATTRACTION (w <= 0.7)"
        status_txt.set_text(f"t = {t_w[step]:.1f} s   {regime}")

    return update


def _precompute_daydreamer_scene() -> dict:
    if "daydreamer" in _SCENE_CACHE:
        return _SCENE_CACHE["daydreamer"]

    from braitenberg.dynamics.neuro import record_episode

    ep = record_episode(_scenario("vehicle12-daydreamer.yaml"), steps=1680)
    ideas, z, r_ord, phi = _compute_ring_order_param(ep["units"])
    res = {"ep": ep, "ideas": ideas, "z": z, "r_ord": r_ord, "phi": phi}
    _SCENE_CACHE["daydreamer"] = res
    return res


def _setup_daydreamer_anim(fig: Figure, theme: Theme, aspect: str, n_frames: int) -> Callable[[int], None]:
    data = _precompute_daydreamer_scene()
    ep = data["ep"]
    ideas = data["ideas"]
    z = data["z"]
    phi = data["phi"]

    compact = aspect != "16:9"
    add_poster_header(
        fig,
        theme,
        "Vehicle 12 Daydreamer · Endogenous Cortical Mandala & Thought Geometry",
        (
            r"8-idea cortical ring pulsing in the dark under caudate threshold control, drawing deterministic symmetries"
            if not compact
            else r"8-idea cortical ring, order parameter $Z(t)$ & dark-arena track"
        ),
        kicker="TRAINS OF THOUGHT · HOMEOSTATIC THRESHOLD CONTROL",
        compact=compact,
    )
    ax_trk, ax_ring = _anim_layout_two_panel(fig, aspect)

    # Start right as caudate ignites the first endogenous thought wave (step ~55)
    start_step = 55
    ux = np.asarray(ep["ux"], dtype=float)[start_step:]
    uy = np.asarray(ep["uy"], dtype=float)[start_step:]
    t_arr = np.asarray(ep["t"], dtype=float)[start_step:]
    ideas_s = ideas[:, start_step:]
    z_s = z[start_step:]
    phi_s = phi[start_step:]
    eff_s = np.asarray(ep.get("effective_thresholds", {}).get("idea_0", np.full_like(ep["t"], 2.0)), dtype=float)[start_step:]

    # Panel A: Physical Dark Arena Thought Geometry
    style_axes(ax_trk, theme, tag="A", title="Physical Dark Arena · Geometric Thought Track", xlabel=r"Unwrapped $x$", ylabel=r"Unwrapped $y$")
    ax_trk.set_aspect("equal", adjustable="datalim")
    pad = 5.0
    cx, cy = 0.5 * (ux.min() + ux.max()), 0.5 * (uy.min() + uy.max())
    half_span = 0.5 * max(np.ptp(ux), np.ptp(uy), 20.0) + pad
    ax_trk.set_xlim(cx - half_span, cx + half_span)
    ax_trk.set_ylim(cy - half_span, cy + half_span)
    ax_trk.plot(ux, uy, color=theme.fg_dim, lw=0.75, ls=":", alpha=0.25, zorder=2)

    trk_halo = ax_trk.plot([], [], color=theme.cyan, lw=6.0, alpha=0.16, zorder=4.8)[0] if theme.glow_enabled else None
    trk_core = ax_trk.plot([], [], color=theme.cyan, lw=2.2, zorder=5)[0]
    comet_v12 = CometArtist(ax_trk, theme, max_lw=3.0, head_size=56.0, zorder=7)

    # Panel B: 8-Neuron Circular Mandala + Complex Order Parameter Z(t)
    style_axes(ax_ring, theme, tag="B", title=r"8-Idea Cortical Mandala & Order Parameter $Z(t)$", xlabel=r"$\mathrm{Re}\,Z(t)$", ylabel=r"$\mathrm{Im}\,Z(t)$", grid=False)
    ax_ring.set_aspect("equal", adjustable="datalim")
    ring_rad = 0.44
    ang = 2.0 * math.pi * np.arange(8) / 8.0
    nx_r, ny_r = ring_rad * np.cos(ang), ring_rad * np.sin(ang)
    for k in range(8):
        for step_ahead in (1, 3):
            j = (k + step_ahead) % 8
            ax_ring.plot([nx_r[k], nx_r[j]], [ny_r[k], ny_r[j]], color=theme.spine, lw=1.0, alpha=0.60, zorder=2)
    gestures = ["FWD", "LEFT", "RIGHT", "STOP", "FWD", "LEFT", "RIGHT", "STOP"]
    for k in range(8):
        lx, ly = (ring_rad + 0.115) * math.cos(ang[k]), (ring_rad + 0.115) * math.sin(ang[k])
        ax_ring.text(lx, ly, f"I{k}·{gestures[k]}", color=theme.fg_muted, fontsize=7.4, fontweight="bold", ha="center", va="center", path_effects=text_halo(theme), zorder=9)

    th_circle = np.linspace(0, 2 * math.pi, 180)
    ax_ring.plot(ring_rad * np.cos(th_circle), ring_rad * np.sin(th_circle), color=theme.fg_dim, lw=0.8, ls=":", alpha=0.45, zorder=2)
    ax_ring.scatter([0.0], [0.0], s=95, marker="D", c=[theme.violet], edgecolors="#ffffff" if theme.is_dark else theme.fg, zorder=8)
    ax_ring.set_xlim(-0.66, 0.66)
    ax_ring.set_ylim(-0.62, 0.62)

    node_halo = ax_ring.scatter(nx_r, ny_r, s=np.full(8, 220.0), c=[theme.panel_bg] * 8, alpha=0.30, edgecolors="none", zorder=6.5) if theme.glow_enabled else None
    node_core = ax_ring.scatter(nx_r, ny_r, s=np.full(8, 120.0), c=[theme.panel_bg] * 8, edgecolors=theme.fg_muted, linewidths=1.3, zorder=7)
    z_vec_line = ax_ring.plot([0.0, 0.0], [0.0, 0.0], color=theme.gold if theme.is_dark else theme.coral, lw=2.5, zorder=8.5)[0]
    comet_z = CometArtist(ax_ring, theme, max_lw=2.2, head_size=36.0, zorder=7.5)
    caudate_badge = ax_ring.text(0.03, 0.96, "", transform=ax_ring.transAxes, va="top", ha="left", color=theme.fg, fontsize=8.3, fontweight="bold", path_effects=text_halo(theme), zorder=10)

    cmap_p = phase_cmap(theme)
    node_on_rgba = np.array([cmap_p(k / 8.0) for k in range(8)], dtype=float)
    node_off_rgba = np.tile(to_rgba(theme.panel_bg), (8, 1))
    n_steps = len(ux)

    def update(f_idx: int) -> None:
        step = min(n_steps - 1, int(round(f_idx * (n_steps - 1) / max(1, n_frames - 1))))
        trk_core.set_data(ux[: step + 1], uy[: step + 1])
        if trk_halo is not None:
            trk_halo.set_data(ux[: step + 1], uy[: step + 1])

        s0 = max(0, step - 35)
        phase_col = np.array([cmap_p((phi_s[step] + math.pi) / (2 * math.pi))], dtype=float)
        comet_v12.update(ux[s0 : step + 1], uy[s0 : step + 1], phase_col)

        act_k = ideas_s[:, step]
        colors_now = np.where(act_k[:, None] > 0.5, node_on_rgba, node_off_rgba)
        sizes_now = 95.0 + 135.0 * act_k
        node_core.set_facecolor(colors_now)
        node_core.set_sizes(sizes_now)
        if node_halo is not None:
            halo_c = colors_now.copy()
            halo_c[:, 3] = np.where(act_k > 0.5, 0.34, 0.0)
            node_halo.set_facecolor(halo_c)
            node_halo.set_sizes(sizes_now * 3.2)

        zx_now, zy_now = float(np.real(z_s[step])) * 1.15, float(np.imag(z_s[step])) * 1.15
        z_vec_line.set_data([0.0, zx_now], [0.0, zy_now])
        zs0 = max(0, step - 45)
        comet_z.update(np.real(z_s[zs0 : step + 1]) * 1.15, np.imag(z_s[zs0 : step + 1]) * 1.15, np.array([to_rgba(theme.violet)], dtype=float))
        caudate_badge.set_text(
            f"t = {t_arr[step]:.1f} s   |Z| = {abs(z_s[step]):.2f}   θ_eff = {eff_s[step]:.2f}"
        )

    return update


# ---------------------------------------------------------------------------
# GIF & 60 FPS H.264 MP4 Encoder Pipeline
# ---------------------------------------------------------------------------


def _save_palette_gif(frames: list[Image.Image], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / max(1, fps) / 10.0)) * 10)
    # Build global 256-color palette across 5 keyframes so mid-run colors stay vibrant
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
    scene_key: str,
    theme_name: str,
    aspect: str,
    fps: int,
    seconds: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int | None = None,
    mp4_subdir: Path | None = None,
    mp4_aspect_tag: bool = False,
) -> list[Path]:
    theme = get_theme(theme_name)
    stem = SCENE_STEMS[scene_key]
    want_mp4 = "mp4" in formats
    want_gif = "gif" in formats
    ffmpeg_exe = _ffmpeg_bin() if want_mp4 else None

    target_kind = "mp4" if (want_mp4 and ffmpeg_exe) else "gif"
    effective_fps = max(1, fps) if (want_mp4 and ffmpeg_exe) else min(max(1, fps), 25)
    n_frames = max(2, int(round(effective_fps * seconds)))

    setup_fns = {
        "orbiter": _setup_orbiter_anim,
        "ftle_swarm": _setup_ftle_anim,
        "plasticity": _setup_plasticity_anim,
        "daydreamer": _setup_daydreamer_anim,
    }

    with mpl.rc_context(theme_rc_params(theme)):
        fig, canvas, _ = _new_figure(aspect, target_kind, theme, dpi)
        update_fn = setup_fns[stem](fig, theme, aspect, n_frames)
        canvas.draw()
        h_px, w_px = np.asarray(canvas.buffer_rgba()).shape[:2]

        written: list[Path] = []
        proc: subprocess.Popen | None = None
        mp4_path: Path | None = None

        if want_mp4 and ffmpeg_exe is not None:
            dest_dir = mp4_subdir if mp4_subdir is not None else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_{aspect.replace(':', 'x')}" if mp4_aspect_tag else ""
            mp4_path = dest_dir / f"{stem}{tag}.mp4"
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
                "18",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        gif_stride = max(1, int(round(effective_fps / 25.0))) if want_gif else 1
        gif_fps = max(1, int(round(effective_fps / gif_stride)))
        gif_frames: list[Image.Image] = []
        # Scale high-res MP4 frames down cleanly when also saving a lightweight README GIF
        max_gif_side = 960
        scale = min(1.0, max_gif_side / max(w_px, h_px)) if (dpi is None and max(w_px, h_px) > max_gif_side) else 1.0
        gif_size = (max(2, int(round(w_px * scale))), max(2, int(round(h_px * scale))))

        try:
            for f_idx in range(n_frames):
                update_fn(f_idx)
                canvas.draw()
                rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(rgb.tobytes())
                if want_gif and (f_idx % gif_stride == 0):
                    img = Image.fromarray(rgb, mode="RGB")
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
            gif_path = out_dir / f"{stem}.gif"
            _save_palette_gif(gif_frames, gif_path, gif_fps)
            written.append(gif_path)

        return written


def render_animations(
    *,
    scene: str = "all",
    theme: str = "cosmic",
    aspect: str = "16:9",
    fps: int = 60,
    seconds: float = 8.0,
    out_dir: str | Path = ".",
    formats: Sequence[str] = ("gif", "mp4"),
    dpi: int | None = None,
) -> list[Path]:
    """Render requested animation scene(s) and return written file paths."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scenes = ["orbiter", "ftle-swarm", "plasticity", "daydreamer"] if scene == "all" else [scene]
    written: list[Path] = []
    for sc in scenes:
        written.extend(
            _render_single_animation(
                scene_key=sc,
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
    seconds: float = 8.0,
) -> list[Path]:
    """Generate the full curated gallery:
    - 4K Cosmic posters (top-level canonical filenames)
    - Textbook print plates (``*_textbook.png``)
    - Crisp palette-optimized README GIFs (``16:9`` cosmic)
    - 60 FPS H.264 MP4 cuts across ``16:9``, ``1:1``, and ``9:16`` under ``<out-dir>/mp4/``
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mp4_dir = out_path / "mp4"
    written: list[Path] = []

    # 1. Cosmic 4K posters + Textbook plates
    written.extend(render_posters(act="all", theme="cosmic", aspect="16:9", out_dir=out_path, fmt="png"))
    written.extend(
        render_posters(
            act="all",
            theme="textbook",
            aspect="16:9",
            out_dir=out_path,
            fmt="png",
            suffix="_textbook",
        )
    )

    # 2. 16:9 GIFs + 16:9 MP4s in a single pass, plus 1:1 & 9:16 MP4 cuts
    for sc in ("orbiter", "ftle-swarm", "plasticity", "daydreamer"):
        written.extend(
            _render_single_animation(
                scene_key=sc,
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
                    scene_key=sc,
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
