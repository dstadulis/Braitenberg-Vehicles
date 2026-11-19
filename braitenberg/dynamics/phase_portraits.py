"""Series I — Pure Axiom-1 Single-Experiment 4-Panel Deep Phase-Portrait Plates.

Every figure produced by this module strictly enforces **Axiom 1**
(:func:`braitenberg.dynamics.shannon.validate_single_experiment_axiom`):
1 file = strictly 1 isolated experiment and 1 homogeneous Braitenberg vehicle species
(zero cross-vehicle mixing!), expanding the Strogatz dynamical analysis into 8 bespoke
4-panel 4K plates across ``cosmic`` and ``textbook`` aesthetics and ``16:9``, ``1:1``,
and ``9:16`` aspect ratios:

1. ``v2a_fear``: Vehicle 2a (Fear) — cylindrical ``(r, psi)`` repulsion, physical arena
   scattering fan, radial drift ``dr/dt`` & turn-rate ``omega(r, psi)``, and Rutherford-like
   impact-parameter deflection ``b -> Delta theta(b)``.
2. ``v3a_love``: Vehicle 3a (Love) — cylindrical ``(r, psi)`` stable arrest attractor &
   off-axis saddles, arena capture vs. saddle escape, speed deceleration ``v(r) -> 0``,
   and equilibrium arrest distance ``r*(I_0)`` nonsmooth clipping bifurcation.
3. ``v4a_orbiter``: Vehicle 4a (Orbiter) — single-lamp ``(r, psi)`` limit cycle & spiral
   sinks, 1D ``peak.at`` escape bifurcation, 3-lamp forward FTLE ridges & trefoil basins,
   and 3-lamp Poincare surface of section ``Sigma: x = L_x/2``.
4. ``v5_moth``: Vehicle 5 (Moth) — irreversible burn-latch fold catastrophe in ``(r, dr/dt)``,
   pre-burn (``burned=0``) attraction ``(r, psi)`` portrait, post-burn (``burned=1``)
   repulsion ``(r, psi)`` portrait (topological flip of ``psi=0``), and two-lamp arena trajectory.
5. ``v5_forager``: Vehicle 5 (Forager) — slow-fast ``(r, E)`` relaxation limit cycle across
   ``[30, 90]``, arena rosette foraging track, ``E(t) / r(t) / full(t)`` hysteresis traces,
   and behavioral duty-cycle breakdown across lamp intensity.
6. ``v7_wary_moth``: Vehicle 7 (Wary Moth) — hybrid ``(r_red, w_red)`` trajectory crossing
   ``w_c = 0.7``, Mnemotrix acquisition & cooled extinction ``w_red(t)`` vs ``w_white(t)``,
   two-lamp selective avoidance arena paths, and minimum pass distance ``r_min`` envelope.
7. ``v11_cautious_driver``: Vehicle 11 (Cautious Driver) — Ergotrix sequence staircases
   crossing ``w_c = 0.7`` vs ``0`` reverse wires, longitudinal braking portrait ``(d_line, v)``,
   G/Y/R cycle alignment & anticipatory ``expects_red`` shift, and ``3x3`` directed synaptic matrix.
8. ``v12_daydreamer``: Vehicle 12 (Daydreamer) — caudate criticality sweep across ``tau_target``,
   cortical mandala & Kuramoto order parameter ``Z(t) = R e^{i Phi}``, 8-idea cascade &
   threshold breathing ``theta_eff(t)``, and endogenous dark-arena thought loops across seeds.

Architecture rule (AGENTS.md): Never imports ``pygame`` and never opens GUI windows.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from braitenberg.definition import ACTIVE_STRENGTH, VehicleDefinition, load_definition
from braitenberg.dynamics import themes as _themes
from braitenberg.dynamics.lagrangian import compute_ftle_and_basins, poincare_section
from braitenberg.dynamics.neuro import record_episode
from braitenberg.dynamics.phase_atlas import (
    _orient_grid_2d,
    draw_polar_portrait,
    format_psi_axis,
    integrate_polar_sample,
)
from braitenberg.dynamics.phase_plane import bifurcation_sweep, polar_coords, polar_field, wrap_pi
from braitenberg.dynamics.shannon import (
    PHASE_EXPERIMENTS,
    ExperimentSpec,
    validate_single_experiment_axiom,
)
from braitenberg.dynamics.themes import (
    Theme,
    add_poster_header,
    badge_bbox,
    basin_colors,
    field_cmap,
    ftle_cmap,
    get_theme,
    phase_cmap,
    plot_comet_trails,
    plot_glow_line,
    plot_glow_scatter,
    plot_lamp_beacon,
    style_axes,
    style_legend,
    text_halo,
    theme_rc_params,
)
from braitenberg.scenario import Placement, Scenario, SimulationSettings, load_scenario
from braitenberg.sim import Simulation
from braitenberg.world import Arena, Source, Stimulus, World, load_world

REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE_PORTRAIT_EXPERIMENTS: tuple[str, ...] = (
    "v2a_fear",
    "v3a_love",
    "v4a_orbiter",
    "v5_moth",
    "v5_forager",
    "v7_wary_moth",
    "v11_cautious_driver",
    "v12_daydreamer",
)

_PORTRAIT_CACHE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Figure, Grid & Pure-Simulation Helpers
# ---------------------------------------------------------------------------


def _resolve_figsize(aspect: str, target: str) -> tuple[float, float, int]:
    from braitenberg.dynamics import studio as _studio

    fn = getattr(_studio, "aspect_figsize", _themes.aspect_figsize)
    return fn(aspect, target)


def _new_figure(
    aspect: str, target: str, theme: Theme, dpi: int | None = None
) -> tuple[Figure, FigureCanvasAgg, int]:
    w_in, h_in, default_dpi = _resolve_figsize(aspect, target)
    actual_dpi = dpi if dpi is not None else default_dpi
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
    """Create a balanced 4-panel subplot grid adapted cleanly to 16:9, 1:1, or 9:16."""
    if aspect == "9:16":
        gs = fig.add_gridspec(
            4,
            1,
            left=0.132,
            right=0.942,
            bottom=0.056,
            top=0.892,
            hspace=0.38,
        )
        return [fig.add_subplot(gs[i, 0]) for i in range(4)]
    if aspect == "1:1":
        gs = fig.add_gridspec(
            2,
            2,
            left=0.110,
            right=0.950,
            bottom=0.080,
            top=0.858,
            wspace=0.33,
            hspace=0.36,
        )
        return [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    gs = fig.add_gridspec(
        2,
        2,
        left=0.062,
        right=0.964,
        bottom=0.090,
        top=0.835,
        wspace=0.24,
        hspace=0.38,
    )
    return [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]


def _add_axiom1_footer(
    fig: Figure,
    theme: Theme,
    spec: ExperimentSpec,
    detail: str,
    aspect: str,
    *,
    short_detail: str | None = None,
) -> None:
    y = 0.012 if aspect == "9:16" else 0.018
    x = 0.056 if aspect == "9:16" else (0.095 if aspect == "1:1" else 0.062)
    fs = 6.2 if aspect == "9:16" else (6.9 if aspect == "1:1" else 7.5)
    if aspect == "16:9":
        msg = (
            f"Axiom 1 Pure Single-Species Phase Portrait [{spec.experiment_id}] · "
            f"{spec.focal_vehicle_path} · {detail}"
        )
    elif aspect == "1:1":
        msg = f"Axiom 1 Pure [{spec.experiment_id}] · {short_detail or detail}"
    else:
        msg = f"Axiom 1 [{spec.experiment_id}] · {short_detail or detail}"
    fig.text(
        x,
        y,
        msg,
        color=theme.fg_dim,
        fontsize=fs,
        va="bottom",
        ha="left",
    )


def _tag_axiom1_figure(fig: Figure, spec: ExperimentSpec) -> Figure:
    fig.axiom1_metadata = {  # type: ignore[attr-defined]
        "series": "phase_portraits",
        "experiment_id": spec.experiment_id,
        "focal_vehicle_path": spec.focal_vehicle_path,
        "comparative_synthesis": False,
    }
    fig.series = "phase_portraits"  # type: ignore[attr-defined]
    fig.comparative_synthesis = False  # type: ignore[attr-defined]
    return fig


def _noiseless_copy(vdef: VehicleDefinition) -> VehicleDefinition:
    out = copy.deepcopy(vdef)
    out.general.noise_std = 0.0
    return out


def _plot_wrapped_line(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    theme: Theme,
    arena_size: tuple[float, float],
    *,
    lw: float = 1.9,
    alpha: float = 0.90,
    label: str | None = None,
    zorder: float = 5.0,
) -> None:
    """Plot a trajectory breaking segments cleanly across periodic arena wrap jumps."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2:
        return
    w, h = arena_size
    jumps = np.where((np.abs(np.diff(xa)) > 0.45 * w) | (np.abs(np.diff(ya)) > 0.45 * h))[0] + 1
    x_chunks = np.split(xa, jumps)
    y_chunks = np.split(ya, jumps)
    used_lbl = False
    for xc, yc in zip(x_chunks, y_chunks):
        if len(xc) >= 2:
            plot_glow_line(
                ax,
                xc,
                yc,
                color,
                theme,
                lw=lw,
                alpha=alpha,
                label=(label if not used_lbl else None),
                zorder=zorder,
            )
            used_lbl = True


def _resolve_portrait_list(experiment: str) -> list[str]:
    key = experiment.strip().lower()
    if key == "all":
        return list(PHASE_PORTRAIT_EXPERIMENTS)
    if key not in PHASE_PORTRAIT_EXPERIMENTS:
        raise ValueError(
            f"Unknown phase-portrait experiment '{experiment}'. Choose from: "
            f"{', '.join(PHASE_PORTRAIT_EXPERIMENTS)}, all"
        )
    return [key]


# ---------------------------------------------------------------------------
# 1. Vehicle 2a: Fear — Repulsive Scattering & Rutherford Deflection
# ---------------------------------------------------------------------------


def _get_v2a_fear_data() -> dict[str, Any]:
    if "v2a_fear" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v2a_fear"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v2a_fear"])
    v2a = _noiseless_copy(load_definition(REPO_ROOT / spec.focal_vehicle_path))

    # Open (bounce-free large) single-lamp scattering world for clean hyperbolic rays
    lamp_pos = (50.0, 50.0)
    arena_world = World(
        name="Fear scattering arena",
        arena=Arena(width=100.0, height=100.0, edges="bounce"),
        stimuli={"light": Stimulus(name="light", sources=[Source(lamp_pos, intensity=10.0, falloff=15.0)])},
    )

    b_fan = np.array([-30.0, -23.0, -17.0, -12.0, -7.5, -4.0, -1.8, 1.8, 4.0, 7.5, 12.0, 17.0, 23.0, 30.0])
    placements = [Placement(v2a, (8.0, 50.0 + float(b)), 0.0, f"Fear #{i}") for i, b in enumerate(b_fan)]
    sc_fan = Scenario("v2a_scattering_fan", arena_world, placements, SimulationSettings(dt=0.04, seed=0))
    sim_fan = Simulation(sc_fan)
    validate_single_experiment_axiom(spec, vehicles=sim_fan.vehicles, scenario=sc_fan)

    n_steps = 1250
    n_fan = len(b_fan)
    fx = np.full((n_steps, n_fan), np.nan, dtype=float)
    fy = np.full((n_steps, n_fan), np.nan, dtype=float)
    stopped_edge = np.zeros(n_fan, dtype=bool)
    for t_i in range(n_steps):
        for j, veh in enumerate(sim_fan.vehicles):
            if stopped_edge[j]:
                continue
            if veh.x < 1.8 or veh.x > 98.2 or veh.y < 1.8 or veh.y > 98.2:
                stopped_edge[j] = True
                continue
            fx[t_i, j] = veh.x
            fy[t_i, j] = veh.y
        sim_fan.step()

    # Radial & angular profile curves from exact polar_field
    r_prof = np.linspace(2.0, 46.0, 140)
    psi_prof = np.array([0.0, math.pi / 4.0, 2.0 * math.pi / 3.0, math.pi], dtype=float)
    pf_prof = polar_field(v2a, arena_world, r_grid=r_prof, psi_grid=psi_prof)

    # Rutherford-like impact parameter sweep b -> Delta theta(b) across intensities I_0 in {6, 10, 16}
    b_neg = np.linspace(-42.0, -0.8, 26)
    b_pos = np.linspace(0.8, 42.0, 26)
    b_sweep = np.concatenate([b_neg, b_pos])
    deflections: dict[float, np.ndarray] = {}
    wide_arena = Arena(width=400.0, height=400.0, edges="bounce")
    src_wide = (200.0, 200.0)

    for inten in (6.0, 10.0, 16.0):
        w_i = World(
            name=f"rutherford_{inten:g}",
            arena=wide_arena,
            stimuli={"light": Stimulus(name="light", sources=[Source(src_wide, intensity=inten, falloff=15.0)])},
        )
        pl_i = [Placement(v2a, (135.0, 200.0 + float(b)), 0.0, f"Fear b={b:.1f}") for b in b_sweep]
        sc_i = Scenario(f"v2a_rutherford_{inten:g}", w_i, pl_i, SimulationSettings(dt=0.05, seed=0))
        sim_i = Simulation(sc_i)
        validate_single_experiment_axiom(spec, vehicles=sim_i.vehicles, scenario=sc_i)
        for _ in range(950):
            sim_i.step()
        dth = np.array([float(wrap_pi(v.heading)) for v in sim_i.vehicles], dtype=float)
        deflections[inten] = dth

    res = {
        "spec": spec,
        "vdef": v2a,
        "world": arena_world,
        "b_fan": b_fan,
        "fan_x": fx,
        "fan_y": fy,
        "r_prof": r_prof,
        "pf_prof": pf_prof,
        "b_neg_count": len(b_neg),
        "b_sweep": b_sweep,
        "deflections": deflections,
    }
    _PORTRAIT_CACHE["v2a_fear"] = res
    return res


def _render_v2a_fear(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v2a_fear_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"

    # Panel A: Cylindrical (r, psi) phase portrait
    draw_polar_portrait(
        axes[0],
        data["vdef"],
        data["world"],
        theme,
        tag="A",
        title=r"Cylindrical Phase Portrait $(r, \psi)$",
        sample_ics=[(7.5, 0.25), (7.5, -0.25), (12.0, 0.65), (12.0, -0.65), (19.0, 1.05), (19.0, -1.05)],
    )
    axes[0].text(
        29.5,
        2.50,
        r"Escape $\psi \to +\pi$",
        color=theme.cyan,
        fontsize=7.8,
        fontweight="bold",
        path_effects=text_halo(theme),
        zorder=9,
    )
    axes[0].text(
        29.5,
        -2.62,
        r"Escape $\psi \to -\pi$",
        color=theme.coral,
        fontsize=7.8,
        fontweight="bold",
        path_effects=text_halo(theme),
        zorder=9,
    )

    # Panel B: Arena Hyperbolic Scattering Fan
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title="Arena Repulsive Scattering Fan",
        xlabel=r"Arena $x$",
        ylabel=r"Arena $y$",
        grid=False,
    )
    ax_b.set_aspect("equal", adjustable="box")
    gx, gy = np.meshgrid(np.linspace(0, 100, 110), np.linspace(0, 100, 110))
    field = data["world"].stimulus_at("light", gx, gy) / 10.0
    ax_b.imshow(
        field,
        origin="lower",
        extent=[0, 100, 0, 100],
        cmap=field_cmap(theme),
        interpolation="bilinear",
        alpha=0.44 if theme.is_dark else 0.30,
        aspect="equal",
        zorder=1,
    )
    ax_b.axhline(50.0, color=theme.fg_dim, lw=0.85, ls=":", alpha=0.6, zorder=2)
    b_fan = data["b_fan"]
    fx, fy = data["fan_x"], data["fan_y"]
    drew_pos = drew_neg = False
    for j, b_val in enumerate(b_fan):
        col = theme.cyan if b_val > 0 else theme.coral
        lbl = None
        if b_val > 0 and not drew_pos:
            lbl = r"Impact $b > 0$ ($\Delta\theta > 0$)"
            drew_pos = True
        elif b_val < 0 and not drew_neg:
            lbl = r"Impact $b < 0$ ($\Delta\theta < 0$)"
            drew_neg = True
        plot_glow_line(ax_b, fx[:, j], fy[:, j], col, theme, lw=1.75, alpha=0.88, label=lbl, zorder=5)
        valid_idx = np.flatnonzero(np.isfinite(fx[:, j]))
        if len(valid_idx) > 0:
            last_i = int(valid_idx[-1])
            plot_glow_scatter(ax_b, [fx[last_i, j]], [fy[last_i, j]], col, theme, size=22, zorder=6.5)

    plot_lamp_beacon(ax_b, 50.0, 50.0, theme, size=84, zorder=8)
    ax_b.set_xlim(0.0, 100.0)
    ax_b.set_ylim(0.0, 100.0)
    style_legend(ax_b.legend(loc="upper right", fontsize=7.0 if compact else 7.6), theme)

    # Panel C: Radial Drift & Turn-Rate Profiles
    ax_c = axes[2]
    style_axes(
        ax_c,
        theme,
        tag="C",
        title=r"Radial Drift $\dot{r}(r,\psi)$ & Repulsive Torque",
        xlabel=r"Distance to lamp $r$",
        ylabel=r"$\dot{r}$ (u/s) & scaled turn $-4\omega$ (rad/s)",
    )
    r_p = data["r_prof"]
    pf_p = data["pf_prof"]
    dr_p = np.asarray(pf_p["dr"], dtype=float)
    om_p = np.asarray(pf_p["turn_rate"], dtype=float)
    ax_c.axhline(0.0, color=theme.fg_dim, lw=0.85, ls=":", alpha=0.65, zorder=2)
    plot_glow_line(ax_c, r_p, dr_p[3], theme.emerald, theme, lw=2.2, label=r"$\dot{r}(r, \psi=\pi)$ flee", zorder=5.5)
    plot_glow_line(ax_c, r_p, dr_p[2], theme.amber, theme, lw=1.8, label=r"$\dot{r}(r, \psi=2\pi/3)$ oblique", zorder=5.0)
    plot_glow_line(ax_c, r_p, dr_p[0], theme.coral, theme, lw=2.0, label=r"$\dot{r}(r, \psi=0)$ head-on", zorder=5.2)
    plot_glow_line(
        ax_c,
        r_p,
        -4.0 * om_p[1],
        theme.violet,
        theme,
        lw=2.0,
        ls="--",
        label=r"$-4\omega(r, \psi=\pi/4)$ turn away",
        zorder=6.0,
    )
    ax_c.set_xlim(float(r_p[0]), float(r_p[-1]))
    ax_c.set_ylim(-10.2, 10.2)
    style_legend(ax_c.legend(loc="upper right", fontsize=7.1 if compact else 7.7), theme)

    # Panel D: Impact-Parameter Deflection Curve b -> Delta theta(b)
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title=r"Rutherford Deflection Curve $b \mapsto \Delta\theta(b)$",
        xlabel=r"Impact parameter $b = y_0 - y_{\mathrm{lamp}}$",
        ylabel=r"Scattering deflection $\Delta\theta$ (rad)",
    )
    ax_d.axhline(0.0, color=theme.fg_dim, lw=0.8, ls=":", alpha=0.6)
    ax_d.axvline(0.0, color=theme.amber, lw=1.1, ls="--", alpha=0.75, label=r"Head-on separatrix $b=0$")
    n_neg = data["b_neg_count"]
    b_sw = data["b_sweep"]
    curve_specs = [
        (6.0, theme.emerald, ":", 1.7, r"$I_0 = 6$ (weak)"),
        (10.0, theme.cyan, "-", 2.4, r"$I_0 = 10$ (canonical)"),
        (16.0, theme.coral, "-.", 1.9, r"$I_0 = 16$ (intense)"),
    ]
    for inten, col, ls_style, lw_v, lbl in curve_specs:
        dth = data["deflections"][inten]
        plot_glow_line(ax_d, b_sw[:n_neg], dth[:n_neg], col, theme, lw=lw_v, ls=ls_style, label=lbl, zorder=5.5)
        plot_glow_line(ax_d, b_sw[n_neg:], dth[n_neg:], col, theme, lw=lw_v, ls=ls_style, zorder=5.5)
    ax_d.set_xlim(-42.0, 42.0)
    format_psi_axis(ax_d)
    style_legend(ax_d.legend(loc="lower right", fontsize=7.1 if compact else 7.7), theme)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Uncrossed excitatory sensorimotor taxis: unstable head-on ray ψ = 0 & antisymmetric Rutherford scattering",
        aspect,
        short_detail="Uncrossed excitatory taxis: unstable ray ψ = 0 & antisymmetric Rutherford scattering",
    )
    return spec


# ---------------------------------------------------------------------------
# 2. Vehicle 3a: Love — Stable Arrest Attractor & Equilibrium Sweep
# ---------------------------------------------------------------------------


def _get_v3a_love_data() -> dict[str, Any]:
    if "v3a_love" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v3a_love"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v3a_love"])
    v3a = _noiseless_copy(load_definition(REPO_ROOT / spec.focal_vehicle_path))

    lamp_pos = (52.0, 50.0)
    world_love = World(
        name="Love capture arena",
        arena=Arena(width=100.0, height=100.0, edges="bounce"),
        stimuli={"light": Stimulus(name="light", sources=[Source(lamp_pos, intensity=10.0, falloff=15.0)])},
    )

    # Launch capture & saddle-escape ensemble around (52, 50)
    ics_arena: list[tuple[tuple[float, float], float]] = []
    # Capture fan approaching from left & oblique angles
    for y0 in (24.0, 33.0, 41.0, 47.0, 53.0, 59.0, 67.0, 76.0):
        ics_arena.append(((10.0, y0), 0.0))
    for ang_deg in (55.0, 125.0, -55.0, -125.0):
        rad = math.radians(ang_deg)
        x0 = lamp_pos[0] + 34.0 * math.cos(rad)
        y0 = lamp_pos[1] + 34.0 * math.sin(rad)
        hdg = math.atan2(lamp_pos[1] - y0, lamp_pos[0] - x0) + (0.35 if ang_deg > 0 else -0.35)
        ics_arena.append(((x0, y0), hdg))
    # Outer saddle-escape rays (facing nearly away / wide grazing)
    for x0, y0, hdg in (
        (36.0, 78.0, math.radians(35.0)),
        (36.0, 22.0, math.radians(-35.0)),
        (66.0, 76.0, math.radians(42.0)),
        (66.0, 24.0, math.radians(-42.0)),
    ):
        ics_arena.append(((x0, y0), hdg))

    placements = [Placement(v3a, pos, hdg, f"Love #{i}") for i, (pos, hdg) in enumerate(ics_arena)]
    sc_love = Scenario("v3a_capture_escape", world_love, placements, SimulationSettings(dt=0.05, seed=0))
    sim_love = Simulation(sc_love)
    validate_single_experiment_axiom(spec, vehicles=sim_love.vehicles, scenario=sc_love)

    n_steps = 460
    n_veh = len(placements)
    lx = np.full((n_steps, n_veh), np.nan, dtype=float)
    ly = np.full((n_steps, n_veh), np.nan, dtype=float)
    final_sp = np.zeros(n_veh, dtype=float)
    hit_wall = np.zeros(n_veh, dtype=bool)
    for t_i in range(n_steps):
        for j, veh in enumerate(sim_love.vehicles):
            if hit_wall[j]:
                continue
            if veh.x < 1.8 or veh.x > 98.2 or veh.y < 1.8 or veh.y > 98.2:
                hit_wall[j] = True
                final_sp[j] = veh.speed
                continue
            lx[t_i, j] = veh.x
            ly[t_i, j] = veh.y
            final_sp[j] = veh.speed
        sim_love.step()

    captured = final_sp < 0.05

    # Speed profiles v(r, psi=0) across lamp intensities I_0
    r_prof = np.linspace(1.5, 48.0, 200)
    speed_curves: dict[float, np.ndarray] = {}
    for inten in (6.0, 8.333, 10.0, 15.0, 22.0):
        w_i = World(
            name=f"love_v_{inten:g}",
            arena=Arena(width=120.0, height=120.0, edges="bounce"),
            stimuli={"light": Stimulus(name="light", sources=[Source((60.0, 60.0), intensity=inten, falloff=15.0)])},
        )
        pf_i = polar_field(v3a, w_i, r_grid=r_prof, psi_grid=np.array([0.0]))
        speed_curves[inten] = np.asarray(pf_i["speed"][0], dtype=float)

    # Equilibrium arrest distance sweep r*(I_0) vs lamp intensity I_0 in [5.0, 26.0]
    i_vals = np.linspace(5.0, 26.0, 28)
    r_sim = np.full_like(i_vals, np.nan)
    wide_arena = Arena(width=200.0, height=200.0, edges="bounce")
    for k_i, inten in enumerate(i_vals):
        w_s = World(
            name=f"love_sweep_{inten:g}",
            arena=wide_arena,
            stimuli={"light": Stimulus(name="light", sources=[Source((100.0, 100.0), intensity=float(inten), falloff=15.0)])},
        )
        sc_s = Scenario("v3a_rstar", w_s, [Placement(v3a, (42.0, 100.0), 0.0, "Love #0")], SimulationSettings(dt=0.04, seed=0))
        sim_s = Simulation(sc_s)
        validate_single_experiment_axiom(spec, vehicles=sim_s.vehicles, scenario=sc_s)
        for _ in range(2200):
            sim_s.step()
            if sim_s.vehicles[0].speed <= 1e-6:
                break
        veh0 = sim_s.vehicles[0]
        if veh0.speed <= 3e-3 and veh0.x < 100.0:
            r_sim[k_i] = 100.0 - float(veh0.x)

    res = {
        "spec": spec,
        "vdef": v3a,
        "world": world_love,
        "lamp_pos": lamp_pos,
        "lx": lx,
        "ly": ly,
        "captured": captured,
        "r_prof": r_prof,
        "speed_curves": speed_curves,
        "i_vals": i_vals,
        "r_sim": r_sim,
    }
    _PORTRAIT_CACHE["v3a_love"] = res
    return res


def _render_v3a_love(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v3a_love_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"

    # Panel A: Cylindrical (r, psi) phase portrait + arrest core
    pf_a = draw_polar_portrait(
        axes[0],
        data["vdef"],
        data["world"],
        theme,
        tag="A",
        title=r"Cylindrical $(r, \psi)$ Arrest & Off-Axis Saddles",
        sample_ics=[(38.0, 2.2), (38.0, -2.2), (30.0, 1.25), (30.0, -1.25), (20.0, 0.65), (20.0, -0.65)],
    )
    rg_a, pg_a, sp_a = np.asarray(pf_a["r"]), np.asarray(pf_a["psi"]), np.asarray(pf_a["speed"])
    axes[0].contourf(
        rg_a,
        pg_a,
        np.where(sp_a <= 1e-4, 1.0, np.nan),
        levels=[0.5, 1.5],
        colors=[theme.emerald],
        alpha=0.26,
        zorder=2.8,
    )
    axes[0].annotate(
        r"Arrest $v=0$" + "\n" + r"$(\psi=0,\; r^*\!\approx\!8.3)$",
        xy=(8.3, 0.10),
        xytext=(5.6, 0.72),
        color=theme.emerald,
        fontsize=7.3 if compact else 7.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": theme.emerald, "lw": 1.15},
        path_effects=text_halo(theme),
        zorder=9,
    )

    # Panel B: Physical Arena Capture vs Saddle Escape
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title="Arena Capture Ring vs. Saddle-Escape Rays",
        xlabel=r"Arena $x$",
        ylabel=r"Arena $y$",
        grid=False,
    )
    ax_b.set_aspect("equal", adjustable="box")
    gx, gy = np.meshgrid(np.linspace(0, 100, 110), np.linspace(0, 100, 110))
    field = data["world"].stimulus_at("light", gx, gy) / 10.0
    ax_b.imshow(
        field,
        origin="lower",
        extent=[0, 100, 0, 100],
        cmap=field_cmap(theme),
        interpolation="bilinear",
        alpha=0.44 if theme.is_dark else 0.30,
        aspect="equal",
        zorder=1,
    )
    sx, sy = data["lamp_pos"]
    r_star = 8.26
    th_c = np.linspace(0, 2 * math.pi, 160)
    ax_b.plot(
        sx + r_star * np.cos(th_c),
        sy + r_star * np.sin(th_c),
        color=theme.emerald,
        lw=1.35,
        ls="--",
        alpha=0.88,
        label=r"Arrest circle $r^*\approx 8.3$",
        zorder=3.5,
    )
    lx, ly, cap = data["lx"], data["ly"], data["captured"]
    drew_cap = drew_esc = False
    for j in range(lx.shape[1]):
        is_c = bool(cap[j])
        col = theme.emerald if is_c else theme.coral
        lbl = None
        if is_c and not drew_cap:
            lbl = "Captured into rest (v → 0)"
            drew_cap = True
        elif (not is_c) and not drew_esc:
            lbl = "Outer saddle escape"
            drew_esc = True
        plot_glow_line(ax_b, lx[:, j], ly[:, j], col, theme, lw=1.8 if is_c else 1.55, alpha=0.88, label=lbl, zorder=5)
        valid = np.flatnonzero(np.isfinite(lx[:, j]))
        if len(valid) > 0:
            li = int(valid[-1])
            plot_glow_scatter(ax_b, [lx[li, j]], [ly[li, j]], col, theme, size=34 if is_c else 20, zorder=7)

    plot_lamp_beacon(ax_b, sx, sy, theme, size=84, zorder=8)
    ax_b.set_xlim(0.0, 100.0)
    ax_b.set_ylim(0.0, 100.0)
    style_legend(ax_b.legend(loc="upper left", fontsize=6.9 if compact else 7.4), theme)

    # Panel C: Head-On Speed Deceleration Profiles v(r)
    ax_c = axes[2]
    style_axes(
        ax_c,
        theme,
        tag="C",
        title=r"Head-On Deceleration $v(r, \psi=0)$ to Rest",
        xlabel=r"Distance to lamp $r$",
        ylabel=r"Forward speed $v(r)$ (units/s)",
    )
    r_p = data["r_prof"]
    sc_specs = [
        (6.0, theme.coral, "--", 1.8, r"$I_0 = 6.0 < I_c$ (no arrest)"),
        (8.333, theme.amber, "-.", 1.8, r"$I_0 = 8.33$ (grazing)"),
        (10.0, theme.cyan, "-", 2.5, r"$I_0 = 10.0$ (canonical)"),
        (15.0, theme.emerald, "-", 2.0, r"$I_0 = 15.0$"),
        (22.0, theme.violet, ":", 1.9, r"$I_0 = 22.0$"),
    ]
    for inten, col, ls_s, lw_s, lbl in sc_specs:
        plot_glow_line(ax_c, r_p, data["speed_curves"][inten], col, theme, lw=lw_s, ls=ls_s, label=lbl, zorder=5.2)
    ax_c.set_xlim(float(r_p[0]), float(r_p[-1]))
    ax_c.set_ylim(-0.25, 5.4)
    style_legend(ax_c.legend(loc="lower right", fontsize=7.0 if compact else 7.6), theme)

    # Panel D: Equilibrium Arrest Distance Sweep r*(I_0)
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title=r"Equilibrium Arrest Distance $r^*(I_0)$ Sweep",
        xlabel=r"Lamp intensity $I_0$",
        ylabel=r"Resting distance $r^*$ (arena units)",
    )
    # Exact analytical curve: d_eye^2 = 15^2 * (0.6*I0 / 5 - 1); r* = 1.8 + sqrt(d_eye^2 - 1.8^2)
    i_dense = np.linspace(5.0, 26.0, 300)
    d2_eye = 225.0 * (0.12 * i_dense - 1.0)
    valid_th = d2_eye >= (1.8 * 1.8)
    i_crit = (1.0 + (1.8 * 1.8) / 225.0) / 0.12  # ~8.453
    ax_d.axvspan(5.0, i_crit, color=theme.coral, alpha=0.12, zorder=1)
    ax_d.axvline(i_crit, color=theme.coral, lw=1.2, ls="--", alpha=0.82, label=f"Arrest threshold $I_c\\approx{i_crit:.2f}$", zorder=3)
    ax_d.text(
        0.5 * (5.0 + i_crit),
        22.5,
        "Pass-through\n($v > 0$)",
        color=theme.coral,
        fontsize=7.3,
        fontweight="bold",
        ha="center",
        va="center",
        path_effects=text_halo(theme),
        zorder=8,
    )
    r_theory = np.where(valid_th, 1.8 + np.sqrt(np.maximum(d2_eye - 3.24, 0.0)), np.nan)
    ax_d.plot(i_dense, r_theory, color=theme.fg_dim, lw=1.5, ls="--", label=r"Analytical clipping root $r^*(I_0)$", zorder=4)

    i_vals, r_sim = data["i_vals"], data["r_sim"]
    fin = np.isfinite(r_sim)
    plot_glow_line(ax_d, i_vals[fin], r_sim[fin], theme.cyan, theme, lw=2.3, zorder=5)
    plot_glow_scatter(ax_d, i_vals[fin], r_sim[fin], theme.cyan, theme, size=32, label="Simulated rest equilibrium", zorder=6)
    plot_glow_scatter(
        ax_d,
        [10.0],
        [1.8 + math.sqrt(225.0 * 0.2 - 3.24)],
        theme.gold if theme.is_dark else theme.coral,
        theme,
        size=70,
        label=r"Canonical ($I_0=10,\; r^*\approx 8.26$)",
        zorder=8,
    )
    ax_d.set_xlim(5.0, 26.0)
    ax_d.set_ylim(0.0, 30.0)
    style_legend(ax_d.legend(loc="lower right", fontsize=7.0 if compact else 7.6), theme)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Uncrossed inhibitory taxis: motor saturation at zero creates a nonsmooth stable arrest equilibrium",
        aspect,
        short_detail="Uncrossed inhibitory taxis: motor clipping at zero creates stable arrest r*(I₀)",
    )
    return spec


# ---------------------------------------------------------------------------
# 3. Vehicle 4a: Orbiter — Single-Lamp Limit Cycle & 3-Lamp Lagrangian Chaos
# ---------------------------------------------------------------------------


def _get_v4a_orbiter_data() -> dict[str, Any]:
    if "v4a_orbiter" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v4a_orbiter"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v4a_orbiter"])
    v4a = load_definition(REPO_ROOT / spec.focal_vehicle_path)
    sc_swarm = load_scenario(REPO_ROOT / spec.scenario_path)
    sim_check = Simulation(sc_swarm)
    validate_single_experiment_axiom(spec, vehicles=sim_check.vehicles, scenario=sc_swarm)

    world_one = load_world(REPO_ROOT / "worlds" / "one-light.yaml")
    world_3 = load_world(REPO_ROOT / "worlds" / "three-lights-chaos.yaml")
    sweep = bifurcation_sweep(v4a, world_one, param="peak_at", values=np.linspace(1.2, 9.2, 30))
    lcs = compute_ftle_and_basins(v4a, world_3, nx=110, ny=110, horizon=11.0, dt=0.05)
    psec = poincare_section(v4a, world_3, n_ic=54, steps=2600, dt=0.05, seed=0)

    res = {
        "spec": spec,
        "vdef": v4a,
        "world_one": world_one,
        "world_3": world_3,
        "sweep": sweep,
        "lcs": lcs,
        "psec": psec,
    }
    _PORTRAIT_CACHE["v4a_orbiter"] = res
    return res


def _render_v4a_orbiter(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v4a_orbiter_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"

    # Panel A: Single-Lamp Cylindrical Limit Cycle
    draw_polar_portrait(
        axes[0],
        data["vdef"],
        data["world_one"],
        theme,
        tag="A",
        title=r"Single-Lamp $(r, \psi)$ Limit Cycle & Sinks",
        sample_ics=[(5.0, 0.5), (5.0, -0.5), (39.0, 0.55), (39.0, -0.55), (32.0, 2.3)],
        show_legend=True,
    )

    # Panel B: 1D Orbital Radius Bifurcation Sweep across peak.at
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title=r"Orbital Bifurcation Sweep Across $I_{\mathrm{peak}}$",
        xlabel=r"Sensor peak preference $I_{\mathrm{peak}}$ (peak.at)",
        ylabel=r"Asymptotic orbit radius $r^*$",
    )
    sweep = data["sweep"]
    vals = np.asarray(sweep["values"], dtype=float)
    r_min = np.asarray(sweep["r_min"], dtype=float)
    r_max = np.asarray(sweep["r_max"], dtype=float)
    r_mean = np.asarray(sweep["r_mean"], dtype=float)
    att_types = [str(t) for t in sweep.get("attractor_type", ["limit_cycle"] * len(vals))]

    i_dense = np.linspace(1.2, 9.2, 200)
    r_theory = 15.0 * np.sqrt(np.maximum(10.0 / i_dense - 1.0, 0.0))
    ax_b.plot(i_dense, r_theory, color=theme.fg_dim, lw=1.25, ls="--", alpha=0.82, label=r"Analytical $I(r)=I_{\mathrm{peak}}$", zorder=3)

    bounded = np.isfinite(r_mean) & (r_mean < 65.0) & np.array(["escape" not in t.lower() for t in att_types])
    if np.any(~bounded):
        crit_i = float(vals[bounded][0]) if np.any(bounded) else 2.4
        ax_b.axvspan(float(vals.min()), crit_i, color=theme.coral, alpha=0.12, zorder=1)
        ax_b.axvline(crit_i, color=theme.coral, lw=1.15, ls="-.", alpha=0.82, zorder=3.5)
        ax_b.text(
            0.5 * (float(vals.min()) + crit_i),
            43.5,
            "Escape\n($r\\to\\infty$)",
            color=theme.coral,
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="top",
            path_effects=text_halo(theme),
            zorder=8,
        )
    if np.any(bounded):
        ax_b.fill_between(vals[bounded], r_min[bounded], r_max[bounded], color=theme.cyan, alpha=0.22, label=r"Orbital band $[r_{\min}, r_{\max}]$", zorder=4)
        plot_glow_line(ax_b, vals[bounded], r_mean[bounded], theme.cyan, theme, lw=2.5, label=r"Stable orbit $r^*$", zorder=5)
        r_canon = float(np.interp(4.0, vals[bounded], r_mean[bounded]))
    else:
        r_canon = 19.9
    ax_b.axvline(4.0, color=theme.amber, lw=1.1, ls=":", alpha=0.82, zorder=4)
    plot_glow_scatter(
        ax_b,
        [4.0],
        [r_canon],
        theme.gold if theme.is_dark else theme.coral,
        theme,
        size=64,
        label=f"Canonical ($I_{{\\mathrm{{peak}}}}=4,\\; r^*\\approx{r_canon:.1f}$)",
        zorder=7,
    )
    ax_b.set_xlim(float(vals.min()), float(vals.max()))
    ax_b.set_ylim(0.0, 50.0)
    style_legend(ax_b.legend(loc="upper right", fontsize=7.1 if compact else 7.7), theme)

    # Panel C: 3-Lamp Forward FTLE Ridges & Trefoil Capture Basins
    ax_c = axes[2]
    world_3: World = data["world_3"]
    lcs = data["lcs"]
    xg, yg = np.asarray(lcs["x"], dtype=float), np.asarray(lcs["y"], dtype=float)
    x1d = xg[0, :] if xg.ndim == 2 else xg
    y1d = yg[:, 0] if yg.ndim == 2 else yg
    ftle = _orient_grid_2d(lcs["ftle"], len(x1d), len(y1d))
    basin = _orient_grid_2d(lcs["basin"], len(x1d), len(y1d))
    extent = [float(x1d[0]), float(x1d[-1]), float(y1d[0]), float(y1d[-1])]
    vmin, vmax = float(np.nanpercentile(ftle, 5.0)), float(np.nanpercentile(ftle, 99.2))

    style_axes(
        ax_c,
        theme,
        tag="C",
        title=r"3-Lamp FTLE Ridges $\sigma_T$ & Trefoil Basins",
        xlabel=r"Arena $x$",
        ylabel=r"Arena $y$",
        grid=False,
    )
    sources = [tuple(s.position) for s in world_3.stimuli["light"].sources]
    b_cols = basin_colors(theme, max(3, len(sources)))
    rgba_basin = np.zeros((*basin.shape, 4), dtype=float)
    rgba_basin[:, :] = to_rgba(theme.panel_bg, 0.85)
    for bid in range(max(3, len(sources))):
        rgba_basin[basin == bid] = to_rgba(b_cols[bid % len(b_cols)], 0.50 if theme.is_dark else 0.35)
    ftle_norm = np.clip((ftle - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    ridge_tint = np.array(to_rgba(theme.gold if theme.is_dark else theme.fg))
    blend = (ftle_norm**2.0)[..., None] * 0.74
    rgba_basin = np.clip(rgba_basin * (1.0 - blend) + ridge_tint * blend, 0.0, 1.0)
    rgba_basin[..., 3] = 0.95
    ax_c.imshow(rgba_basin, origin="lower", extent=extent, interpolation="bilinear", aspect="equal", zorder=2)
    for i, (sx, sy) in enumerate(sources):
        plot_lamp_beacon(ax_c, sx, sy, theme, color=b_cols[i % len(b_cols)], size=72, zorder=8)
    entropy = float(lcs.get("basin_entropy", 0.0))
    ax_c.text(
        0.035,
        0.955,
        f"Basin entropy $S_b = {entropy:.2f}$",
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        color=theme.fg,
        fontsize=7.8,
        fontweight="bold",
        bbox=badge_bbox(theme),
        zorder=9,
    )

    # Panel D: 3-Lamp Poincare Surface of Section Sigma: x = L_x / 2
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title=r"3-Lamp Poincaré Section $\Sigma:\; x = L_x/2$",
        xlabel=r"Section coordinate $y \in [0, L_y]$",
        ylabel=r"Crossing bearing $\psi$ (rad)",
    )
    psec = data["psec"]
    coords = np.asarray(psec["coord"], dtype=float)
    psis = np.asarray(psec.get("psi", psec.get("heading", np.zeros_like(coords))), dtype=float)
    ic_ids = np.asarray(psec.get("ic_id", np.zeros_like(coords, dtype=int)), dtype=int)
    if len(coords) > 0:
        palette = [theme.cyan, theme.coral, theme.amber, theme.emerald, theme.violet, theme.indigo]
        pt_cols = [palette[int(k) % len(palette)] for k in ic_ids]
        plot_glow_scatter(ax_d, coords, psis, pt_cols, theme, size=16.0, alpha=0.82, edgecolor="none", zorder=5)
    ax_d.set_xlim(0.0, world_3.arena.height)
    format_psi_axis(ax_d)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Non-monotonic bell sensor: planar SO(2) limit cycle around 1 lamp vs. chaotic FTLE scattering among 3 lamps",
        aspect,
        short_detail="Bell sensor: SO(2) limit cycle around 1 lamp vs. chaotic 3-lamp FTLE scattering",
    )
    return spec


# ---------------------------------------------------------------------------
# 4. Vehicle 5: Moth — Burn-Latch Fold & Pre/Post-Burn Topological Flip
# ---------------------------------------------------------------------------


def _moth_clamped_polar_field(
    vdef: VehicleDefinition,
    burned_state: int,
    r_grid: np.ndarray,
    psi_grid: np.ndarray,
    intensity: float = 10.0,
    falloff: float = 15.0,
) -> dict[str, np.ndarray]:
    """Exact cylindrical (r, psi) vector field of Vehicle 5 Moth on sheet ``burned = 0`` or ``1``."""
    R, PSI = np.meshgrid(r_grid, psi_grid, indexing="xy")
    hw = vdef.general.width / 2.0
    hl = vdef.general.length / 2.0
    cos_p = np.cos(PSI)
    sin_p = np.sin(PSI)

    def _eye_intensity(bx: float, by: float) -> np.ndarray:
        tx, ty = bx * hw, by * hl
        dx = -R + ty * cos_p - tx * sin_p
        dy = -ty * sin_p - tx * cos_p
        d2 = dx * dx + dy * dy
        return intensity / (1.0 + d2 / (falloff * falloff))

    i_left = _eye_intensity(-0.9, 0.9)
    i_right = _eye_intensity(0.9, 0.9)

    left_br = (i_left - i_right >= 0.05).astype(float)
    right_br = (i_right - i_left >= 0.05).astype(float)
    near = (i_left + i_right >= 10.0).astype(float)

    if burned_state == 0:
        seek_l, seek_r = left_br, right_br
        flee_l = np.zeros_like(left_br)
        flee_r = np.zeros_like(right_br)
    else:
        seek_l = np.zeros_like(left_br)
        seek_r = np.zeros_like(right_br)
        flee_l = left_br * near
        flee_r = right_br * near

    v_left = np.clip(3.0 + 3.0 * (seek_r + flee_l), 0.0, 6.0)
    v_right = np.clip(3.0 + 3.0 * (seek_l + flee_r), 0.0, 6.0)
    speed = 0.5 * (v_left + v_right)
    # Motor positions [-0.5, -0.6] and [0.5, -0.6] -> x = +-1.0 -> sxx = 2.0
    x_m = 0.5 * hw
    sxx = 2.0 * x_m * x_m
    omega = (v_right * x_m - v_left * x_m) / max(sxx, 1e-9)

    dr = -speed * cos_p
    dpsi = (speed / np.maximum(R, 1e-9)) * sin_p - omega
    return {"r": r_grid, "psi": psi_grid, "dr": dr, "dpsi": dpsi, "speed": speed}


def _get_v5_moth_data() -> dict[str, Any]:
    if "v5_moth" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v5_moth"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v5_moth"])
    v5_moth = load_definition(REPO_ROOT / spec.focal_vehicle_path)
    sc_moth = load_scenario(REPO_ROOT / spec.scenario_path)
    sim_check = Simulation(sc_moth)
    validate_single_experiment_axiom(spec, vehicles=sim_check.vehicles, scenario=sc_moth)

    ep = record_episode(sc_moth, steps=5400)
    r_grid = np.linspace(1.8, 42.0, 90)
    psi_grid = np.linspace(-math.pi, math.pi, 91)
    pf_pre = _moth_clamped_polar_field(v5_moth, 0, r_grid, psi_grid)
    pf_post = _moth_clamped_polar_field(v5_moth, 1, r_grid, psi_grid)

    res = {
        "spec": spec,
        "vdef": v5_moth,
        "world": sc_moth.world,
        "ep": ep,
        "pf_pre": pf_pre,
        "pf_post": pf_post,
    }
    _PORTRAIT_CACHE["v5_moth"] = res
    return res


def _render_v5_moth(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v5_moth_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"
    ep = data["ep"]
    world: World = data["world"]

    # Panel A: Irreversible Burn-Latch Fold Catastrophe in (r, dr/dt)
    ax_a = axes[0]
    style_axes(
        ax_a,
        theme,
        tag="A",
        title=r"Irreversible Burn-Latch Fold $(r, \dot{r})$",
        xlabel=r"Distance to nearest lamp $r$",
        ylabel=r"Radial velocity $\dot{r}$ (u/s)",
    )
    r_m = np.asarray(ep["r_nearest"], dtype=float)
    dt_m = float(ep["t"][1] - ep["t"][0])
    dr_raw = np.gradient(r_m, dt_m)
    dr_sm = np.convolve(dr_raw, np.ones(9) / 9.0, mode="same")
    burned = np.asarray(ep["units"]["burned"], dtype=float)
    burn_idx = int(np.argmax(burned > 0.5)) if np.any(burned > 0.5) else len(r_m) // 2
    r_burn_actual = float(r_m[burn_idx])

    ax_a.axhline(0.0, color=theme.fg_dim, lw=0.85, ls=":", alpha=0.65)
    ax_a.axvline(r_burn_actual, color=theme.coral, lw=1.25, ls="--", alpha=0.80, label=f"Pain fold ($r\\approx{r_burn_actual:.1f}$)")
    ax_a.axvline(15.0, color=theme.amber, lw=1.1, ls="-.", alpha=0.72, label=r"Near boundary ($r\approx 15$)")
    plot_glow_line(ax_a, r_m[:burn_idx], dr_sm[:burn_idx], theme.cyan, theme, lw=2.2, label="Pre-burn sheet (burned=0)", zorder=5)
    plot_glow_line(
        ax_a,
        r_m[burn_idx : min(len(r_m), burn_idx + 1400)],
        dr_sm[burn_idx : min(len(r_m), burn_idx + 1400)],
        theme.coral,
        theme,
        lw=2.2,
        label="Post-burn sheet (burned=1)",
        zorder=6,
    )
    plot_glow_scatter(
        ax_a,
        [r_burn_actual],
        [dr_sm[burn_idx]],
        theme.gold if theme.is_dark else theme.coral,
        theme,
        size=70,
        label="1-bit latch jump (0 → 1)",
        zorder=8,
    )
    ax_a.set_xlim(0.0, min(44.0, float(np.nanmax(r_m)) * 1.04))
    ax_a.set_ylim(-6.8, 6.8)
    style_legend(ax_a.legend(loc="lower right", fontsize=6.9 if compact else 7.5), theme)

    # Panel B & Panel C: Pre-burn (burned=0) vs Post-burn (burned=1) Cylindrical (r, psi) Portraits
    for panel_idx, (tag_lbl, pf_sheet, sheet_title, is_post) in enumerate(
        (
            ("B", data["pf_pre"], r"Pre-Burn Sheet ($\mathrm{burned}=0$) · Attraction", False),
            ("C", data["pf_post"], r"Post-Burn Sheet ($\mathrm{burned}=1$) · Repulsion", True),
        ),
        start=1,
    ):
        ax_p = axes[panel_idx]
        style_axes(
            ax_p,
            theme,
            tag=tag_lbl,
            title=sheet_title,
            xlabel=r"Distance to lamp $r$",
            ylabel=r"Bearing $\psi$ (rad)",
        )
        rg, pg = pf_sheet["r"], pf_sheet["psi"]
        dr_s, dp_s = pf_sheet["dr"], pf_sheet["dpsi"]
        ax_p.streamplot(rg, pg, dr_s, dp_s, color=theme.stream, linewidth=0.75, density=1.0, arrowsize=0.82, zorder=2)
        ax_p.contour(rg, pg, dr_s, levels=[0.0], colors=[theme.amber], linewidths=1.25, linestyles="--", alpha=0.85, zorder=3)
        ax_p.axhline(
            0.0,
            color=theme.coral if is_post else theme.cyan,
            lw=1.7,
            ls="-" if not is_post else "--",
            alpha=0.90,
            zorder=4,
        )
        if not is_post:
            r_pain_1 = 15.0 * math.sqrt(max(20.0 / 18.0 - 1.0, 0.0))  # 5.0 around single lamp
            ax_p.axvspan(float(rg[0]), r_pain_1, color=theme.coral, alpha=0.18, zorder=1.5)
            ax_p.axvline(r_pain_1, color=theme.coral, lw=1.35, ls="--", zorder=4.5)
            for r0, p0 in ((36.0, 1.9), (36.0, -1.9), (26.0, 1.1), (26.0, -1.1)):
                rs_c, ps_c = integrate_polar_sample(r0, p0, rg, pg, dr_s, dp_s, steps=360, dt=0.04)
                plot_glow_line(ax_p, rs_c, ps_c, theme.cyan, theme, lw=1.65, alpha=0.86, zorder=5)
            ax_p.text(
                7.2,
                0.58,
                r"Attracting $\psi = 0$" + "\n" + r"($\dot{r} < 0 \to$ burn fold)",
                color=theme.cyan,
                fontsize=7.5,
                fontweight="bold",
                path_effects=text_halo(theme),
                zorder=9,
            )
        else:
            r_near_1 = 15.0  # I_L + I_R >= 10 around single I0=10, falloff=15 lamp
            ax_p.axvspan(float(rg[0]), r_near_1, color=theme.amber, alpha=0.11, zorder=1.5)
            ax_p.axvline(r_near_1, color=theme.amber, lw=1.35, ls="-.", zorder=4.5)
            for r0, p0 in ((5.0, 0.25), (5.0, -0.25), (8.5, 0.65), (8.5, -0.65), (12.0, 1.1), (12.0, -1.1)):
                rs_c, ps_c = integrate_polar_sample(r0, p0, rg, pg, dr_s, dp_s, steps=320, dt=0.04)
                plot_glow_line(ax_p, rs_c, ps_c, theme.coral, theme, lw=1.65, alpha=0.88, zorder=5)
            ax_p.text(
                20.0,
                1.98,
                r"Flipped! $\psi = 0$ repels" + "\n" + r"inside $r < r_{\mathrm{near}}$",
                color=theme.coral,
                fontsize=7.5,
                fontweight="bold",
                path_effects=text_halo(theme),
                zorder=9,
            )
        ax_p.set_xlim(float(rg[0]), float(rg[-1]))
        format_psi_axis(ax_p)

    # Panel D: Two-Lamp Arena Trajectory Before & After Burn Jump
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title="Two-Lamp Arena Trajectory Across Burn Jump",
        xlabel=r"Arena $x$",
        ylabel=r"Arena $y$",
        grid=False,
    )
    ax_d.set_aspect("equal", adjustable="box")
    W, H = world.arena.width, world.arena.height
    gx, gy = np.meshgrid(np.linspace(0, W, 110), np.linspace(0, H, 110))
    field = world.stimulus_at("light", gx, gy) / 12.0
    ax_d.imshow(
        np.clip(field, 0.0, 1.0),
        origin="lower",
        extent=[0, W, 0, H],
        cmap=field_cmap(theme),
        interpolation="bilinear",
        alpha=0.44 if theme.is_dark else 0.30,
        aspect="equal",
        zorder=1,
    )
    th_c = np.linspace(0, 2 * math.pi, 160)
    for src in world.stimuli["light"].sources:
        sx, sy = src.position
        ax_d.plot(sx + 6.5 * np.cos(th_c), sy + 6.5 * np.sin(th_c), color=theme.coral, lw=1.15, ls="--", alpha=0.80, zorder=3)
        ax_d.plot(sx + 16.5 * np.cos(th_c), sy + 16.5 * np.sin(th_c), color=theme.amber, lw=0.95, ls=":", alpha=0.65, zorder=2.8)
        plot_lamp_beacon(ax_d, sx, sy, theme, size=76, zorder=8)

    xm, ym = np.asarray(ep["x"], dtype=float), np.asarray(ep["y"], dtype=float)
    _plot_wrapped_line(ax_d, xm[: burn_idx + 1], ym[: burn_idx + 1], theme.cyan, theme, (W, H), lw=2.2, label="Pre-burn seek (burned=0)", zorder=5.5)
    _plot_wrapped_line(ax_d, xm[burn_idx:], ym[burn_idx:], theme.coral, theme, (W, H), lw=1.85, alpha=0.88, label="Post-burn flee (burned=1)", zorder=5.2)
    plot_glow_scatter(
        ax_d,
        [xm[burn_idx]],
        [ym[burn_idx]],
        theme.gold if theme.is_dark else theme.fg,
        theme,
        size=74,
        label="Burn event",
        zorder=9,
    )
    ax_d.set_xlim(0.0, W)
    ax_d.set_ylim(0.0, H)
    style_legend(ax_d.legend(loc="upper right", fontsize=7.0 if compact else 7.6), theme)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Self-excitatory memory unit (burned -> burned) permanently inverts the stability of ψ = 0 across every lamp",
        aspect,
        short_detail="Self-excitatory latch (burned -> burned) permanently inverts stability of ψ = 0",
    )
    return spec


# ---------------------------------------------------------------------------
# 5. Vehicle 5: Forager — Slow-Fast Relaxation Cycle & Duty Breakdown
# ---------------------------------------------------------------------------


def _get_v5_forager_data() -> dict[str, Any]:
    if "v5_forager" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v5_forager"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v5_forager"])
    v5_for = load_definition(REPO_ROOT / spec.focal_vehicle_path)
    base_sc = load_scenario(REPO_ROOT / spec.scenario_path)
    sc_for = replace(base_sc, simulation=SimulationSettings(dt=0.05, seed=base_sc.simulation.seed))
    sim_check = Simulation(sc_for)
    validate_single_experiment_axiom(spec, vehicles=sim_check.vehicles, scenario=sc_for)

    ep = record_episode(sc_for, steps=3400)  # 170 s -> multiple clean relaxation cycles

    # Duty-cycle phase breakdown across lamp intensity I_0 in {6, 8, 10, 14, 18}
    intensities = np.array([6.0, 8.0, 10.0, 14.0, 18.0], dtype=float)
    frac_explore = np.zeros_like(intensities)
    frac_recharge = np.zeros_like(intensities)
    frac_seek = np.zeros_like(intensities)

    for idx, inten in enumerate(intensities):
        w_i = World(
            name=f"forager_I_{inten:g}",
            arena=Arena(width=100.0, height=100.0, edges="wrap"),
            stimuli={"light": Stimulus(name="light", sources=[Source((60.0, 50.0), intensity=float(inten), falloff=15.0)])},
        )
        sc_i = Scenario(f"forager_{inten:g}", w_i, [Placement(v5_for, (20.0, 65.0), 0.0, "Forager #0")], SimulationSettings(dt=0.05, seed=0))
        sim_i = Simulation(sc_i)
        validate_single_experiment_axiom(spec, vehicles=sim_i.vehicles, scenario=sc_i)
        ep_i = record_episode(sc_i, steps=3000)
        sl = slice(400, 3000)
        full_i = ep_i["units"]["full"][sl] >= 0.5
        r_i = ep_i["r_primary"][sl]
        rech_i = (~full_i) & (r_i <= 16.0)
        seek_i = (~full_i) & (r_i > 16.0)
        frac_explore[idx] = 100.0 * float(np.mean(full_i))
        frac_recharge[idx] = 100.0 * float(np.mean(rech_i))
        frac_seek[idx] = 100.0 * float(np.mean(seek_i))

    res = {
        "spec": spec,
        "vdef": v5_for,
        "world": sc_for.world,
        "ep": ep,
        "intensities": intensities,
        "frac_explore": frac_explore,
        "frac_recharge": frac_recharge,
        "frac_seek": frac_seek,
    }
    _PORTRAIT_CACHE["v5_forager"] = res
    return res


def _render_v5_forager(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v5_forager_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"
    ep = data["ep"]
    world: World = data["world"]

    r_f = np.asarray(ep["r_primary"], dtype=float)
    e_f = np.asarray(ep["energy"], dtype=float)
    t_f = np.asarray(ep["t"], dtype=float)
    full_u = np.asarray(ep["units"]["full"], dtype=float)

    # Panel A: Slow-Fast (r, E) Relaxation Limit Cycle
    ax_a = axes[0]
    style_axes(
        ax_a,
        theme,
        tag="A",
        title=r"Slow–Fast $(r, E)$ Relaxation Limit Cycle",
        xlabel=r"Distance to lamp $r$",
        ylabel=r"Internal energy $E(t)$",
    )
    ax_a.axhspan(30.0, 90.0, color=theme.cyan, alpha=0.06, zorder=1)
    ax_a.axhline(90.0, color=theme.emerald, lw=1.25, ls="--", alpha=0.85, label=r"Charged fold $E_{\mathrm{high}}=90$")
    ax_a.axhline(30.0, color=theme.amber, lw=1.25, ls="--", alpha=0.85, label=r"Hungry fold $E_{\mathrm{low}}=30$")

    changes = np.where(np.diff(full_u > 0.5))[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(r_f)]))
    drew_seek = drew_exp = False
    for s_i, e_i in zip(starts, ends):
        if e_i - s_i < 2:
            continue
        seg_end = min(len(r_f), e_i + 1)
        is_exp = bool(full_u[s_i] > 0.5)
        col = theme.amber if is_exp else theme.emerald
        lbl = None
        if is_exp and not drew_exp:
            lbl = "Explore discharge (full=1)"
            drew_exp = True
        elif (not is_exp) and not drew_seek:
            lbl = "Seek & recharge (full=0)"
            drew_seek = True
        plot_glow_line(ax_a, r_f[s_i:seg_end], e_f[s_i:seg_end], col, theme, lw=2.0, label=lbl, zorder=5)
    ax_a.set_xlim(0.0, max(48.0, float(np.nanmax(r_f)) * 1.05))
    ax_a.set_ylim(-2.0, 105.0)
    style_legend(ax_a.legend(loc="lower left", fontsize=7.0 if compact else 7.6), theme)

    # Panel B: Arena Rosette Foraging Trajectory
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title="Arena Rosette Foraging & Zigzag Track",
        xlabel=r"Arena $x$",
        ylabel=r"Arena $y$",
        grid=False,
    )
    ax_b.set_aspect("equal", adjustable="box")
    W, H = world.arena.width, world.arena.height
    gx, gy = np.meshgrid(np.linspace(0, W, 110), np.linspace(0, H, 110))
    field = world.stimulus_at("light", gx, gy) / 10.0
    ax_b.imshow(
        field,
        origin="lower",
        extent=[0, W, 0, H],
        cmap=field_cmap(theme),
        interpolation="bilinear",
        alpha=0.44 if theme.is_dark else 0.30,
        aspect="equal",
        zorder=1,
    )
    sx, sy = world.stimuli["light"].sources[0].position
    r_sees = 15.0 * math.sqrt(max(20.0 / 3.0 - 1.0, 0.0))  # ~35.7
    th_c = np.linspace(0, 2 * math.pi, 180)
    ax_b.plot(sx + r_sees * np.cos(th_c), sy + r_sees * np.sin(th_c), color=theme.emerald, lw=1.05, ls=":", alpha=0.70, label=r"Sight perimeter $r_{\mathrm{sees}}$", zorder=2.8)
    xf, yf = np.asarray(ep["x"], dtype=float), np.asarray(ep["y"], dtype=float)
    drew_s2 = drew_e2 = False
    for s_i, e_i in zip(starts, ends):
        if e_i - s_i < 2:
            continue
        seg_end = min(len(xf), e_i + 1)
        is_exp = bool(full_u[s_i] > 0.5)
        col = theme.amber if is_exp else theme.emerald
        lbl = None
        if is_exp and not drew_e2:
            lbl = "Zigzag explore (full=1)"
            drew_e2 = True
        elif (not is_exp) and not drew_s2:
            lbl = "Phototaxis (full=0)"
            drew_s2 = True
        _plot_wrapped_line(ax_b, xf[s_i:seg_end], yf[s_i:seg_end], col, theme, (W, H), lw=1.75, alpha=0.88, label=lbl, zorder=5)
    plot_lamp_beacon(ax_b, sx, sy, theme, size=82, zorder=8)
    ax_b.set_xlim(0.0, W)
    ax_b.set_ylim(0.0, H)
    style_legend(ax_b.legend(loc="lower left", fontsize=7.0 if compact else 7.6), theme)

    # Panel C: Time-Series of E(t), r(t), and full Hysteresis State
    ax_c = axes[2]
    style_axes(
        ax_c,
        theme,
        tag="C",
        title=r"Energy $E(t)$, Distance $r(t)$ & Hysteresis Bouts",
        xlabel=r"Simulated time $t$ (s)",
        ylabel=r"Energy $E(t)$ & distance $r(t)$",
    )
    for s_i, e_i in zip(starts, ends):
        if e_i > s_i and full_u[s_i] > 0.5:
            ax_c.axvspan(float(t_f[s_i]), float(t_f[min(len(t_f) - 1, e_i)]), color=theme.amber, alpha=0.13, zorder=1)
    ax_c.axhline(90.0, color=theme.emerald, lw=0.95, ls="--", alpha=0.7)
    ax_c.axhline(30.0, color=theme.amber, lw=0.95, ls="--", alpha=0.7)
    plot_glow_line(ax_c, t_f, e_f, theme.emerald, theme, lw=2.2, label=r"Internal energy $E(t)$", zorder=5.5)
    plot_glow_line(ax_c, t_f, r_f, theme.cyan, theme, lw=1.75, alpha=0.88, label=r"Lamp distance $r(t)$", zorder=5.0)
    ax_c.fill_between([], [], [], color=theme.amber, alpha=0.25, label="full = 1 explore bout")
    ax_c.set_xlim(float(t_f[0]), float(t_f[-1]))
    ax_c.set_ylim(0.0, 106.0)
    style_legend(ax_c.legend(loc="upper right", fontsize=7.0 if compact else 7.6), theme)

    # Panel D: Duty-Cycle & Phase Breakdown Across Lamp Intensity
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title=r"Foraging Duty-Cycle Breakdown vs. Lamp $I_0$",
        xlabel=r"Lamp intensity $I_0$",
        ylabel=r"Time fraction of cycle (%)",
    )
    x_idx = np.arange(len(data["intensities"]), dtype=float)
    f_exp = data["frac_explore"]
    f_rec = data["frac_recharge"]
    f_sk = data["frac_seek"]
    bar_w = 0.56
    ax_d.bar(x_idx, f_rec, width=bar_w, color=theme.emerald, alpha=0.82, edgecolor=theme.spine, linewidth=0.8, label="Recharging ($r\\leq 16$, full=0)", zorder=3)
    ax_d.bar(x_idx, f_sk, width=bar_w, bottom=f_rec, color=theme.cyan, alpha=0.78, edgecolor=theme.spine, linewidth=0.8, label="Seeking / searching (full=0)", zorder=3)
    ax_d.bar(x_idx, f_exp, width=bar_w, bottom=f_rec + f_sk, color=theme.amber, alpha=0.85, edgecolor=theme.spine, linewidth=0.8, label="Exploring (full=1)", zorder=3)
    for k_b, val_e in enumerate(f_exp):
        ax_d.text(
            x_idx[k_b],
            f_rec[k_b] + f_sk[k_b] + 0.5 * val_e,
            f"{val_e:.0f}%",
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color="#060818" if theme.is_dark else theme.fg,
            zorder=6,
        )
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels([f"$I_0={int(v)}$" + ("*" if int(v) == 10 else "") for v in data["intensities"]])
    ax_d.set_ylim(0.0, 114.0)
    style_legend(ax_d.legend(loc="upper right", ncol=1, fontsize=6.8 if compact else 7.4), theme)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Schmitt-trigger energy latch [30, 90] + 3-neuron delay oscillator generates endogenous foraging cycles",
        aspect,
        short_detail="Schmitt-trigger latch [30, 90] + 3-neuron oscillator drives endogenous foraging cycles",
    )
    return spec


# ---------------------------------------------------------------------------
# 6. Vehicle 7: Wary Moth — Mnemotrix Learning, Avoidance & Extinction
# ---------------------------------------------------------------------------


def _get_v7_wary_moth_data() -> dict[str, Any]:
    if "v7_wary_moth" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v7_wary_moth"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v7_wary_moth"])
    sc_wm = load_scenario(REPO_ROOT / spec.scenario_path)
    sim_check = Simulation(sc_wm)
    validate_single_experiment_axiom(spec, vehicles=sim_check.vehicles, scenario=sc_wm)

    # 220 seconds covers BOTH Hot Red [0..120s] (acquisition) AND Cooled Red [120..220s] (extinction)!
    ep = record_episode(sc_wm, steps=22000)
    world = sc_wm.world
    red_pos = world.stimuli["red"].sources[0].position
    white_pos = world.stimuli["white"].sources[0].position
    r_red, _ = polar_coords(ep["x"], ep["y"], ep["heading"], red_pos, arena=world.arena)
    r_white, _ = polar_coords(ep["x"], ep["y"], ep["heading"], white_pos, arena=world.arena)
    w_red = np.asarray(ep["strengths"]["sees_red->danger"], dtype=float)
    w_white = np.asarray(ep["strengths"]["sees_white->danger"], dtype=float)
    t_arr = np.asarray(ep["t"], dtype=float)

    # Extract local minimum approach distances per encounter pass inside r_red <= 26.2
    r_smooth = np.convolve(r_red, np.ones(41) / 41.0, mode="same")
    pass_times: list[float] = []
    pass_rmin: list[float] = []
    pass_w: list[float] = []
    min_sep_steps = 280  # >= 2.8 s between distinct approach passes
    last_step_m = -min_sep_steps
    for idx in range(50, len(t_arr) - 100):
        if r_smooth[idx] <= 26.2 and r_smooth[idx] <= r_smooth[idx - 1] and r_smooth[idx] < r_smooth[idx + 1]:
            win_lo = max(0, idx - 60)
            win_hi = min(len(t_arr), idx + 61)
            if r_smooth[idx] > float(np.min(r_smooth[win_lo:win_hi])) + 1e-6:
                continue
            true_i = win_lo + int(np.argmin(r_red[win_lo:win_hi]))
            # Sample synaptic strength ~1.5 s prior to closest approach (before any new r < 4 burn on this pass)
            pre_i = max(0, true_i - 150)
            if idx - last_step_m >= min_sep_steps:
                pass_times.append(float(t_arr[true_i]))
                pass_rmin.append(float(r_red[true_i]))
                pass_w.append(float(w_red[pre_i]))
                last_step_m = idx
            elif float(r_red[true_i]) < pass_rmin[-1]:
                pass_times[-1] = float(t_arr[true_i])
                pass_rmin[-1] = float(r_red[true_i])
                pass_w[-1] = float(w_red[pre_i])
                last_step_m = idx

    res = {
        "spec": spec,
        "world": world,
        "ep": ep,
        "t": t_arr,
        "r_red": r_red,
        "r_white": r_white,
        "w_red": w_red,
        "w_white": w_white,
        "pass_times": np.asarray(pass_times, dtype=float),
        "pass_rmin": np.asarray(pass_rmin, dtype=float),
        "pass_w": np.asarray(pass_w, dtype=float),
    }
    _PORTRAIT_CACHE["v7_wary_moth"] = res
    return res


def _render_v7_wary_moth(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v7_wary_moth_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"
    ep = data["ep"]
    world: World = data["world"]
    t_arr = data["t"]
    r_red = data["r_red"]
    w_red = data["w_red"]
    w_white = data["w_white"]

    hot_mask = t_arr <= 120.0
    cool_mask = t_arr >= 120.0
    sub_hot = np.flatnonzero(hot_mask)[::6]
    sub_cool = np.flatnonzero(cool_mask)[::6]

    # Panel A: Coupled State-Parameter Trajectory (r_red, w_red)
    ax_a = axes[0]
    style_axes(
        ax_a,
        theme,
        tag="A",
        title=r"Hybrid State–Weight Portrait $(r_{\mathrm{red}}, w_{\mathrm{red}})$",
        xlabel=r"Distance to red lamp $r_{\mathrm{red}}$",
        ylabel=r"Mnemotrix weight $w(\mathrm{sees\_red}\to\mathrm{danger})$",
    )
    ax_a.axhspan(ACTIVE_STRENGTH, 1.04, color=theme.emerald, alpha=0.09, zorder=1)
    ax_a.axhline(
        ACTIVE_STRENGTH,
        color=theme.gold if theme.is_dark else theme.coral,
        lw=1.55,
        ls="--",
        label=f"Active gate $w_c = {ACTIVE_STRENGTH}$",
        zorder=3.5,
    )
    ax_a.axvline(4.0, color=theme.coral, lw=1.15, ls=":", alpha=0.82, label="Burn core ($r < 4$)", zorder=3)
    ax_a.axvline(26.0, color=theme.amber, lw=1.05, ls="-.", alpha=0.72, label=r"Red sight ($r \approx 26$)", zorder=3)
    plot_glow_line(ax_a, r_red[sub_hot], w_red[sub_hot], theme.coral, theme, lw=1.9, alpha=0.88, label="Hot schedule (0–120 s)", zorder=5.2)
    plot_glow_line(ax_a, r_red[sub_cool], w_red[sub_cool], theme.cyan, theme, lw=1.65, alpha=0.82, ls="-", label="Cooled extinction (120–220 s)", zorder=4.8)
    ax_a.set_xlim(0.0, 62.0)
    ax_a.set_ylim(-0.03, 1.05)
    style_legend(ax_a.legend(loc="center right", fontsize=6.8 if compact else 7.4), theme)

    # Panel B: Mnemotrix Acquisition & Cooled Extinction Over Time
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title=r"Acquisition & Extinction Across Lamp Schedule",
        xlabel=r"Simulated time $t$ (s)",
        ylabel=r"Synaptic strength $w(t)$",
    )
    ax_b.axvspan(0.0, 120.0, color=theme.coral, alpha=0.08, zorder=1)
    ax_b.axvspan(120.0, float(t_arr[-1]), color=theme.cyan, alpha=0.07, zorder=1)
    ax_b.axvline(120.0, color=theme.fg_dim, lw=1.2, ls=":", zorder=3)
    ax_b.axhline(ACTIVE_STRENGTH, color=theme.gold if theme.is_dark else theme.coral, lw=1.45, ls="--", label=f"Gate $w_c = {ACTIVE_STRENGTH}$", zorder=4)
    sub_all = slice(0, len(t_arr), 8)
    plot_glow_line(ax_b, t_arr[sub_all], w_red[sub_all], theme.coral, theme, lw=2.3, label="sees_red → danger", zorder=6)
    plot_glow_line(ax_b, t_arr[sub_all], w_white[sub_all], theme.cyan, theme, lw=1.8, label="sees_white → danger", zorder=5)
    ax_b.text(
        62.0,
        0.28,
        "HOT RED LAMP\n(acquisition)",
        color=theme.coral,
        fontsize=7.3,
        fontweight="bold",
        ha="center",
        va="center",
        path_effects=text_halo(theme),
        zorder=8,
    )
    ax_b.text(
        125.5,
        0.95,
        "COOLED LAMP\n(extinction)",
        color=theme.cyan,
        fontsize=7.3,
        fontweight="bold",
        ha="left",
        va="top",
        path_effects=text_halo(theme),
        zorder=8,
    )
    ax_b.set_xlim(0.0, float(t_arr[-1]))
    ax_b.set_ylim(-0.04, 1.06)
    style_legend(ax_b.legend(loc="center right", fontsize=7.0 if compact else 7.6), theme)

    # Panel C: Arena Selective Avoidance (Visiting White, Turning Away from Hot Red)
    ax_c = axes[2]
    style_axes(
        ax_c,
        theme,
        tag="C",
        title="Arena Selective Avoidance (Visits White, Flees Red)",
        xlabel=r"Arena $x$",
        ylabel=r"Arena $y$",
        grid=False,
    )
    ax_c.set_aspect("equal", adjustable="box")
    W, H = world.arena.width, world.arena.height
    gx, gy = np.meshgrid(np.linspace(0, W, 110), np.linspace(0, H, 110))
    field = world.stimulus_at("light", gx, gy) / 12.0
    ax_c.imshow(
        np.clip(field, 0.0, 1.0),
        origin="lower",
        extent=[0, W, 0, H],
        cmap=field_cmap(theme),
        interpolation="bilinear",
        alpha=0.42 if theme.is_dark else 0.28,
        aspect="equal",
        zorder=1,
    )
    th_c = np.linspace(0, 2 * math.pi, 160)
    ax_c.plot(70.0 + 26.0 * np.cos(th_c), 50.0 + 26.0 * np.sin(th_c), color=theme.coral, lw=1.2, ls="--", alpha=0.80, zorder=3)
    ax_c.plot(70.0 + 4.0 * np.cos(th_c), 50.0 + 4.0 * np.sin(th_c), color=theme.gold if theme.is_dark else theme.coral, lw=1.2, ls=":", alpha=0.90, zorder=3.2)
    plot_lamp_beacon(ax_c, 30.0, 50.0, theme, color="#e2e8f0" if theme.is_dark else theme.cyan, size=80, zorder=8)
    plot_lamp_beacon(ax_c, 70.0, 50.0, theme, color=theme.coral, size=82, zorder=8)
    ax_c.text(30.0, 39.0, "Safe White", color=theme.fg, fontsize=7.3, fontweight="bold", ha="center", path_effects=text_halo(theme), zorder=9)
    ax_c.text(70.0, 39.0, "Hot Red", color=theme.coral, fontsize=7.3, fontweight="bold", ha="center", path_effects=text_halo(theme), zorder=9)

    xw, yw = np.asarray(ep["x"], dtype=float), np.asarray(ep["y"], dtype=float)
    cross_steps = np.flatnonzero(w_red > ACTIVE_STRENGTH)
    bif_idx = int(cross_steps[0]) if len(cross_steps) > 0 else 1800
    _plot_wrapped_line(ax_c, xw[:bif_idx:4], yw[:bif_idx:4], theme.coral, theme, (W, H), lw=1.8, alpha=0.85, label=r"Naive ($w \leq 0.7$, burned)", zorder=5)
    _plot_wrapped_line(
        ax_c,
        xw[bif_idx:11800:4],
        yw[bif_idx:11800:4],
        theme.emerald,
        theme,
        (W, H),
        lw=1.85,
        alpha=0.90,
        label=r"Learned ($w > 0.7$, avoids red)",
        zorder=5.5,
    )
    ax_c.set_xlim(0.0, W)
    ax_c.set_ylim(0.0, H)
    style_legend(ax_c.legend(loc="lower left", fontsize=7.0 if compact else 7.6), theme)

    # Panel D: Minimum Approach Distance r_min Per Pass
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title=r"Minimum Red-Lamp Approach Distance $r_{\min}$ per Pass",
        xlabel=r"Encounter time $t_{\mathrm{pass}}$ (s)",
        ylabel=r"Closest approach $r_{\min}$ (units)",
    )
    ax_d.axvline(120.0, color=theme.fg_dim, lw=1.15, ls=":", zorder=2.5)
    ax_d.axhline(4.0, color=theme.coral, lw=1.2, ls="--", alpha=0.82, label=r"Pain threshold ($r=4$)", zorder=3)
    ax_d.axhline(26.0, color=theme.emerald, lw=1.2, ls="-.", alpha=0.80, label=r"Red sight ($r\approx 26$)", zorder=3)
    pt, pr, pw = data["pass_times"], data["pass_rmin"], data["pass_w"]
    if len(pt) > 0:
        plot_glow_line(ax_d, pt, pr, theme.amber, theme, lw=1.8, alpha=0.78, zorder=4.5)
        learned_m = pw > ACTIVE_STRENGTH
        if np.any(~learned_m):
            plot_glow_scatter(ax_d, pt[~learned_m], pr[~learned_m], theme.coral, theme, size=48, label=r"Approach $w \leq 0.7$ (plunges)", zorder=6)
        if np.any(learned_m):
            plot_glow_scatter(ax_d, pt[learned_m], pr[learned_m], theme.emerald, theme, size=52, label=r"Approach $w > 0.7$ (repelled)", zorder=6.5)
    ax_d.set_xlim(0.0, float(t_arr[-1]))
    ax_d.set_ylim(0.0, 38.5)
    style_legend(ax_d.legend(loc="upper right", fontsize=6.9 if compact else 7.5), theme)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Continuous Mnemotrix associative plasticity: crossing w_c = 0.7 activates danger at r ≈ 26; cooling extinguishes fear",
        aspect,
        short_detail="Mnemotrix crossing w_c = 0.7 repels red at r ≈ 26; cooling extinguishes fear",
    )
    return spec


# ---------------------------------------------------------------------------
# 7. Vehicle 11: Cautious Driver — Ergotrix Sequence & Preemptive Braking
# ---------------------------------------------------------------------------


def _get_v11_cautious_driver_data() -> dict[str, Any]:
    if "v11_cautious_driver" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v11_cautious_driver"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v11_cautious_driver"])
    v11 = load_definition(REPO_ROOT / spec.focal_vehicle_path)
    sc_11 = load_scenario(REPO_ROOT / spec.scenario_path)
    sim_check = Simulation(sc_11)
    validate_single_experiment_axiom(spec, vehicles=sim_check.vehicles, scenario=sc_11)

    ep = record_episode(sc_11, steps=7000)  # 70 s (> 5 cycles of 13.5 s)

    # Controlled longitudinal braking comparison arriving at the stop line during Yellow (t in [4.5, 9.0]):
    # Start at x = 25.0 at t = 0: reaches x = 42.0 (d_line = 8.0) at t = 17/5 = 3.4s; if started at x = 18.0,
    # reaches d_line = 8.0 at t = 24/5 = 4.8s (right inside Yellow [4.5, 9.0)!)
    world = sc_11.world
    sc_naive = Scenario(
        "v11_naive_yellow_arrival",
        world,
        [Placement(v11, (18.0, 10.0), 0.0, "Driver Naive")],
        SimulationSettings(dt=0.01, seed=0),
    )
    sim_naive = Simulation(sc_naive)
    validate_single_experiment_axiom(spec, vehicles=sim_naive.vehicles, scenario=sc_naive)
    naive_d, naive_v, naive_c = [], [], []
    for _ in range(1350):
        sim_naive.step()
        veh = sim_naive.vehicles[0]
        naive_d.append(50.0 - float(veh.x))
        naive_v.append(float(veh.speed))
        tm = sim_naive.time % 13.5
        naive_c.append(0 if tm < 4.5 else (1 if tm < 9.0 else 2))

    # Learned probe: exact same Vehicle 11 connectome initialized with learned forward sequence weights (1.0)
    sc_learned = Scenario(
        "v11_learned_yellow_arrival",
        world,
        [Placement(v11, (18.0, 10.0), 0.0, "Driver Learned")],
        SimulationSettings(dt=0.01, seed=0),
    )
    sim_learned = Simulation(sc_learned)
    validate_single_experiment_axiom(spec, vehicles=sim_learned.vehicles, scenario=sc_learned)
    veh_l = sim_learned.vehicles[0]
    for (u_name, j_in) in list(veh_l.strengths.keys()):
        src_name = v11.units[[u.name for u in v11.units].index(u_name)].inputs[j_in].source
        if (src_name, u_name) in (
            ("sees_green", "expects_yellow"),
            ("sees_yellow", "expects_red"),
            ("sees_red", "expects_green"),
        ):
            veh_l.strengths[(u_name, j_in)] = 1.0
    learned_d, learned_v = [], []
    for _ in range(1450):
        sim_learned.step()
        learned_d.append(50.0 - float(veh_l.x))
        learned_v.append(float(veh_l.speed))

    res = {
        "spec": spec,
        "vdef": v11,
        "world": world,
        "ep": ep,
        "naive_d": np.asarray(naive_d, dtype=float),
        "naive_v": np.asarray(naive_v, dtype=float),
        "learned_d": np.asarray(learned_d, dtype=float),
        "learned_v": np.asarray(learned_v, dtype=float),
    }
    _PORTRAIT_CACHE["v11_cautious_driver"] = res
    return res


def _render_v11_cautious_driver(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v11_cautious_driver_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"
    ep = data["ep"]
    t_arr = np.asarray(ep["t"], dtype=float)
    s_dict = ep["strengths"]

    # Panel A: Ergotrix Temporal Sequence Staircases vs 0 Reverse Wires
    ax_a = axes[0]
    style_axes(
        ax_a,
        theme,
        tag="A",
        title=r"Ergotrix Sequence Staircases Crossing $w_c = 0.7$",
        xlabel=r"Simulated time $t$ (s)",
        ylabel=r"Ergotrix synaptic strength $w(t)$",
    )
    ax_a.axhspan(ACTIVE_STRENGTH, 1.04, color=theme.violet, alpha=0.09, zorder=1)
    ax_a.axhline(ACTIVE_STRENGTH, color=theme.gold if theme.is_dark else theme.coral, lw=1.5, ls="--", label=f"Active gate $w_c = {ACTIVE_STRENGTH}$", zorder=4)
    sub = slice(0, len(t_arr), 5)
    plot_glow_line(ax_a, t_arr[sub], s_dict["sees_yellow->expects_red"][sub], theme.amber, theme, lw=2.4, label="sees_yellow → expects_red", zorder=6.2)
    plot_glow_line(ax_a, t_arr[sub], s_dict["sees_green->expects_yellow"][sub], theme.emerald, theme, lw=1.9, ls="-", alpha=0.86, label="sees_green → expects_yellow", zorder=5.8)
    plot_glow_line(ax_a, t_arr[sub], s_dict["sees_red->expects_green"][sub], theme.coral, theme, lw=1.8, ls="-.", alpha=0.84, label="sees_red → expects_green", zorder=5.5)
    plot_glow_line(ax_a, t_arr[sub], s_dict["sees_green->expects_red"][sub], theme.cyan, theme, lw=1.6, ls=":", alpha=0.82, label="3 reverse wires (≡ 0.0)", zorder=5.0)
    ax_a.set_xlim(0.0, float(t_arr[-1]))
    ax_a.set_ylim(-0.04, 1.06)
    style_legend(ax_a.legend(loc="center right", fontsize=6.9 if compact else 7.5), theme)

    # Panel B: Longitudinal Braking Phase Portrait (d_line, v)
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title=r"Longitudinal Braking Portrait $(d_{\mathrm{line}}, v)$ on Yellow",
        xlabel=r"Distance ahead to traffic post $d_{\mathrm{line}} = x_{\mathrm{post}} - x$",
        ylabel=r"Road speed $v$ (units/s)",
    )
    ax_b.axvspan(0.0, 8.3, color=theme.amber, alpha=0.13, zorder=1)
    ax_b.axvline(8.3, color=theme.amber, lw=1.25, ls="--", label=r"Stop-line zone ($d_{\mathrm{line}}\leq 8.3$)", zorder=3)
    ax_b.axvline(0.0, color=theme.coral, lw=1.2, ls=":", label=r"Traffic post ($d_{\mathrm{line}}=0$)", zorder=3)
    nd, nv = data["naive_d"][8:], data["naive_v"][8:]
    ld, lv = data["learned_d"][8:], data["learned_v"][8:]
    plot_glow_line(ax_b, nd, nv + 0.08, theme.coral, theme, lw=2.1, ls="--", label="Naive ($w=0$, runs yellow!)", zorder=5.2)
    plot_glow_line(ax_b, ld, lv, theme.emerald, theme, lw=2.5, label=r"Learned ($w>0.7$, halts at yellow)", zorder=6.2)
    plot_glow_scatter(ax_b, [7.9], [0.0], theme.emerald, theme, size=66, zorder=8)
    ax_b.annotate(
        "Preemptive yellow stop\n(waits for green)",
        xy=(8.0, 0.22),
        xytext=(29.5, 4.05),
        color=theme.emerald,
        fontsize=7.0 if compact else 7.3,
        fontweight="bold",
        bbox=badge_bbox(theme),
        arrowprops={"arrowstyle": "->", "color": theme.emerald, "lw": 1.2},
        zorder=9,
    )
    ax_b.set_xlim(32.5, -16.0)  # Left-to-right = driving forward toward & past the post!
    ax_b.set_ylim(-0.35, 5.65)
    style_legend(ax_b.legend(loc="center left", bbox_to_anchor=(0.015, 0.36), fontsize=6.5 if compact else 7.2), theme)

    # Panel C: Traffic-Light Cycle Alignment & Anticipatory expects_red Shift
    ax_c = axes[2]
    style_axes(
        ax_c,
        theme,
        tag="C",
        title=r"G/Y/R Cycle Alignment & Anticipatory $\mathrm{expects\_red}$",
        xlabel=r"Simulated time $t$ (s)",
        ylabel="Signal / unit state",
    )
    t_max_c = 67.5  # 5 exact cycles of 13.5 s
    for cyc in range(5):
        t0 = cyc * 13.5
        ax_c.axvspan(t0, t0 + 4.5, color=theme.emerald, alpha=0.09, zorder=1)
        ax_c.axvspan(t0 + 4.5, t0 + 9.0, color=theme.amber, alpha=0.12, zorder=1)
        ax_c.axvspan(t0 + 9.0, t0 + 13.5, color=theme.coral, alpha=0.11, zorder=1)

    mask_c = t_arr <= t_max_c
    tc = t_arr[mask_c][::4]
    u_sr = np.asarray(ep["units"]["sees_red"][mask_c][::4], dtype=float)
    u_sy = np.asarray(ep["units"]["sees_yellow"][mask_c][::4], dtype=float)
    u_er = np.asarray(ep["units"]["expects_red"][mask_c][::4], dtype=float)
    u_st = np.asarray(ep["units"]["stop"][mask_c][::4], dtype=float)

    plot_glow_line(ax_c, tc, 3.0 + 0.65 * u_sy, theme.amber, theme, lw=1.8, zorder=5)
    plot_glow_line(ax_c, tc, 2.0 + 0.65 * u_sr, theme.coral, theme, lw=1.8, zorder=5)
    plot_glow_line(ax_c, tc, 1.0 + 0.68 * u_er, theme.violet, theme, lw=2.2, zorder=6)
    plot_glow_line(ax_c, tc, 0.0 + 0.68 * u_st, theme.cyan, theme, lw=2.0, zorder=6)

    cross_er = np.flatnonzero(s_dict["sees_yellow->expects_red"] > ACTIVE_STRENGTH)
    if len(cross_er) > 0:
        t_bif = float(t_arr[int(cross_er[0])])
        ax_c.axvline(t_bif, color=theme.gold if theme.is_dark else theme.fg, lw=1.4, ls="--", zorder=7)
        if aspect == "1:1":
            ax_c.text(
                t_bif + 1.0,
                3.74,
                f"$w > 0.7$ ($t\\approx{t_bif:.1f}$s):\nexpects_red on YELLOW!",
                color=theme.gold if theme.is_dark else theme.fg,
                fontsize=6.6,
                fontweight="bold",
                va="bottom",
                path_effects=text_halo(theme),
                zorder=9,
            )
        else:
            ax_c.text(
                t_bif + 1.0,
                3.78,
                f"$w > 0.7$ ($t\\approx{t_bif:.1f}$s): expects_red fires on YELLOW!",
                color=theme.gold if theme.is_dark else theme.fg,
                fontsize=7.2,
                fontweight="bold",
                path_effects=text_halo(theme),
                zorder=9,
            )

    ax_c.set_xlim(0.0, t_max_c)
    ax_c.set_ylim(-0.2, 4.52 if aspect == "1:1" else 4.15)
    ax_c.set_yticks([0.34, 1.34, 2.34, 3.34])
    ax_c.set_yticklabels(["stop", "expects_red", "sees_red", "sees_yellow"])

    # Panel D: Directed 3x3 (6 Ergotrix + 3 Fixed) Asymptotic Weight Matrix
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title=(
            r"Directed Synaptic Matrix ($6$ Ergotrix + $3$ Innate)"
            if aspect == "1:1"
            else r"Asymptotic Directed Synaptic Matrix ($6$ Ergotrix + $3$ Innate)"
        ),
        xlabel="Target expectation unit",
        ylabel="Source color detector",
        grid=False,
    )
    mat = np.array(
        [
            [1.0, float(s_dict["sees_green->expects_yellow"][-1]), float(s_dict["sees_green->expects_red"][-1])],
            [float(s_dict["sees_yellow->expects_green"][-1]), 1.0, float(s_dict["sees_yellow->expects_red"][-1])],
            [float(s_dict["sees_red->expects_green"][-1]), float(s_dict["sees_red->expects_yellow"][-1]), 1.0],
        ],
        dtype=float,
    )
    # Custom RGBA grid distinguishing Innate Fixed (diagonal), Learned Active (>0.7), and Silent Reverse (0.0)
    rgba_mat = np.zeros((3, 3, 4), dtype=float)
    for r_i in range(3):
        for c_i in range(3):
            if r_i == c_i:
                rgba_mat[r_i, c_i] = to_rgba(theme.cyan, 0.34 if theme.is_dark else 0.24)
            elif mat[r_i, c_i] > ACTIVE_STRENGTH:
                rgba_mat[r_i, c_i] = to_rgba(theme.emerald, 0.65 if theme.is_dark else 0.45)
            else:
                rgba_mat[r_i, c_i] = to_rgba(theme.panel_bg, 0.90)

    ax_d.imshow(rgba_mat, origin="upper", aspect="auto", zorder=2)
    for r_i in range(3):
        for c_i in range(3):
            if r_i == c_i:
                txt = "1.00\n(innate)"
                col_t = theme.cyan
            elif mat[r_i, c_i] > ACTIVE_STRENGTH:
                txt = f"{mat[r_i, c_i]:.2f}\n(LEARNED > 0.7)"
                col_t = "#ffffff" if theme.is_dark else theme.fg
            else:
                txt = f"{mat[r_i, c_i]:.2f}\n(silent)"
                col_t = theme.fg_dim
            ax_d.text(c_i, r_i, txt, ha="center", va="center", fontsize=7.5, fontweight="bold", color=col_t, zorder=5)

    ax_d.set_xticks([0, 1, 2])
    ax_d.set_xticklabels(["expects_green", "expects_yellow", "expects_red"])
    ax_d.set_yticks([0, 1, 2])
    ax_d.set_yticklabels(["sees_green", "sees_yellow", "sees_red"])

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Discrete-event Ergotrix window (within = 5.5 s) extracts directed cyclic arrow of time G → Y → R → G",
        aspect,
        short_detail="Ergotrix window (5.5 s) extracts directed arrow of time G → Y → R → G",
    )
    return spec


# ---------------------------------------------------------------------------
# 8. Vehicle 12: Daydreamer — Caudate Criticality & Cortical Mandala
# ---------------------------------------------------------------------------


def _get_v12_daydreamer_data() -> dict[str, Any]:
    if "v12_daydreamer" in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE["v12_daydreamer"]

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS["v12_daydreamer"])
    v12_base = load_definition(REPO_ROOT / spec.focal_vehicle_path)
    sc_dream = load_scenario(REPO_ROOT / spec.scenario_path)
    sim_check = Simulation(sc_dream)
    validate_single_experiment_axiom(spec, vehicles=sim_check.vehicles, scenario=sc_dream)

    ep_dream = record_episode(sc_dream, steps=1600)
    open_world = load_world(REPO_ROOT / "worlds" / "open-field.yaml")

    targets_sweep = np.linspace(0.0, 7.2, 19)
    mean_act = np.zeros_like(targets_sweep)
    mean_r = np.zeros_like(targets_sweep)
    entropy_vals = np.zeros_like(targets_sweep)

    angles = np.exp(1j * 2.0 * math.pi * np.arange(8)[:, None] / 8.0)
    for idx, tgt in enumerate(targets_sweep):
        new_units = [replace(u, target=float(tgt)) if u.is_control else u for u in v12_base.units]
        v_mod = replace(v12_base, units=new_units)
        sc = Scenario(
            name=f"v12_sweep_{tgt:.2f}",
            world=open_world,
            vehicles=[Placement(v_mod, (50.0, 50.0), math.pi / 2, "Daydreamer")],
            simulation=SimulationSettings(dt=0.05, seed=0),
        )
        ep = record_episode(sc, steps=600)
        ideas_m = np.vstack([np.asarray(ep["units"][f"idea_{k}"], dtype=float) for k in range(8)])
        r_m = np.abs(np.mean(ideas_m * angles, axis=0))
        steady = ideas_m[:, 160:]
        mean_act[idx] = float(steady.sum(axis=0).mean())
        mean_r[idx] = float(r_m[160:].mean()) * 8.0
        weights = (1 << np.arange(8))[:, None]
        codes = (steady.astype(int) * weights).sum(axis=0)
        _, counts = np.unique(codes, return_counts=True)
        probs = counts / counts.sum()
        entropy_vals[idx] = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

    seed_tracks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for sd in (0, 1, 2):
        sc_sd = Scenario(
            name=f"v12_seed_{sd}",
            world=open_world,
            vehicles=[Placement(v12_base, (0.0, 0.0), math.pi / 2, f"Daydreamer s{sd}")],
            simulation=SimulationSettings(dt=0.05, seed=sd),
        )
        sim_sd = Simulation(sc_sd)
        validate_single_experiment_axiom(spec, vehicles=sim_sd.vehicles, scenario=sc_sd)
        ep_sd = record_episode(sc_sd, steps=1500)
        ux = np.asarray(ep_sd["ux"], dtype=float) - float(ep_sd["ux"][0])
        uy = np.asarray(ep_sd["uy"], dtype=float) - float(ep_sd["uy"][0])
        seed_tracks.append((sd, ux, uy))

    res = {
        "spec": spec,
        "ep_dream": ep_dream,
        "targets_sweep": targets_sweep,
        "mean_act": mean_act,
        "mean_r": mean_r,
        "entropy_vals": entropy_vals,
        "seed_tracks": seed_tracks,
    }
    _PORTRAIT_CACHE["v12_daydreamer"] = res
    return res


def _render_v12_daydreamer(fig: Figure, axes: list[mpl.axes.Axes], theme: Theme, aspect: str) -> ExperimentSpec:
    data = _get_v12_daydreamer_data()
    spec: ExperimentSpec = validate_single_experiment_axiom(data["spec"])
    compact = aspect != "16:9"
    ep_dream = data["ep_dream"]

    ideas_d = np.vstack([np.asarray(ep_dream["units"][f"idea_{k}"], dtype=float) for k in range(8)])
    angles = np.exp(1j * 2.0 * math.pi * np.arange(8)[:, None] / 8.0)
    z_d = np.mean(ideas_d * angles, axis=0)

    # Panel A: Caudate Criticality Sweep
    ax_a = axes[0]
    style_axes(
        ax_a,
        theme,
        tag="A",
        title=r"Caudate Criticality Sweep Across $\tau_{\mathrm{target}}$",
        xlabel=r"Caudate target activity $\tau_{\mathrm{target}}$",
        ylabel=r"Activity $\langle A \rangle$ & Entropy $H$",
    )
    tgts = data["targets_sweep"]
    ax_a.axvspan(0.0, 0.25, color=theme.fg_dim, alpha=0.14, zorder=1)
    ax_a.axvspan(0.35, 2.6, color=theme.emerald, alpha=0.11, zorder=1)
    ax_a.axvspan(4.5, 7.2, color=theme.coral, alpha=0.10, zorder=1)
    plot_glow_line(ax_a, tgts, data["mean_act"], theme.cyan, theme, lw=2.2, label=r"Mean activity $\langle A \rangle$", zorder=5)
    plot_glow_line(ax_a, tgts, data["entropy_vals"], theme.emerald, theme, lw=2.1, label=r"Thought entropy $H$ (bits)", zorder=6)
    plot_glow_line(ax_a, tgts, data["mean_r"], theme.violet, theme, lw=1.8, ls="--", label=r"Scaled coherence $8\langle |Z| \rangle$", zorder=5.5)
    ax_a.axvline(1.0, color=theme.gold if theme.is_dark else theme.amber, lw=1.2, ls=":", zorder=4)
    ax_a.text(1.48, 5.85, "Critical\nDaydreamer", color=theme.emerald, fontsize=7.3 if compact else 7.6, fontweight="bold", ha="center", va="top", path_effects=text_halo(theme))
    ax_a.text(6.10, 8.85, "Epileptic\nFlood", color=theme.coral, fontsize=7.3 if compact else 7.6, fontweight="bold", ha="center", va="top", path_effects=text_halo(theme))
    ax_a.set_xlim(0.0, 7.2)
    ax_a.set_ylim(-0.2, 9.25)
    style_legend(ax_a.legend(loc="upper center", bbox_to_anchor=(0.45, 0.99), fontsize=6.6 if compact else 7.3), theme)

    # Panel B: Cortical Mandala & Kuramoto Ring Order Parameter Z(t)
    ax_b = axes[1]
    style_axes(
        ax_b,
        theme,
        tag="B",
        title=r"Cortical Mandala & Order Parameter $Z(t) = Re^{i\Phi}$",
        xlabel=r"$\mathrm{Re}\,Z(t)$",
        ylabel=r"$\mathrm{Im}\,Z(t)$",
    )
    ax_b.set_aspect("equal", adjustable="box")
    ring_rad = 0.42
    node_angles = 2.0 * math.pi * np.arange(8) / 8.0
    nx_ring = ring_rad * np.cos(node_angles)
    ny_ring = ring_rad * np.sin(node_angles)
    for k in range(8):
        for step_ahead in (1, 3):
            j = (k + step_ahead) % 8
            ax_b.plot([nx_ring[k], nx_ring[j]], [ny_ring[k], ny_ring[j]], color=theme.spine, lw=0.85, alpha=0.55, zorder=2)
    th_c = np.linspace(0.0, 2.0 * math.pi, 180)
    ax_b.plot(ring_rad * np.cos(th_c), ring_rad * np.sin(th_c), color=theme.fg_dim, lw=0.8, ls=":", alpha=0.5, zorder=2)
    plot_glow_line(ax_b, np.real(z_d[80:]), np.imag(z_d[80:]), theme.cyan, theme, lw=1.7, alpha=0.85, label="Daydreamer $Z(t)$", zorder=5)
    plot_glow_scatter(ax_b, [0.0], [0.0], theme.coral, theme, size=48, marker="x", linewidths=1.8, label="Silent / Flooded ($Z=0$)", zorder=7)
    duty = ideas_d[:, 80:].mean(axis=1)
    node_cols = [phase_cmap(theme)(k / 8.0) for k in range(8)]
    plot_glow_scatter(ax_b, nx_ring, ny_ring, node_cols, theme, size=55 + 110 * duty, zorder=8)
    for k in range(8):
        lx, ly = (ring_rad + 0.085) * math.cos(node_angles[k]), (ring_rad + 0.085) * math.sin(node_angles[k])
        ax_b.text(lx, ly, f"$I_{k}$", color=theme.fg_muted, fontsize=7.5, ha="center", va="center", path_effects=text_halo(theme), zorder=9)
    ax_b.set_xlim(-0.58, 1.32)
    ax_b.set_ylim(-0.70, 0.62)
    style_legend(ax_b.legend(loc="lower right", fontsize=7.2 if compact else 7.8), theme)

    # Panel C: 8-Idea Cascade & Caudate Threshold Breathing
    ax_c = axes[2]
    style_axes(
        ax_c,
        theme,
        tag="C",
        title=r"8-Idea Cascade & Caudate Threshold $\theta_{\mathrm{eff}}(t)$",
        xlabel=r"Simulated time $t$ (s)",
        ylabel=r"Active idea $k$ & $\theta_{\mathrm{eff}}(t)$",
    )
    t_d = np.asarray(ep_dream["t"], dtype=float)
    sl = slice(0, min(len(t_d), 900))
    for k in range(8):
        active_t = np.where(ideas_d[k, sl] > 0.5, k + 0.5, np.nan)
        plot_glow_line(ax_c, t_d[sl], active_t, phase_cmap(theme)(k / 8.0), theme, lw=3.2, alpha=0.92, zorder=5)
    eff_trace = np.asarray(ep_dream.get("effective_thresholds", {}).get("idea_0", np.full_like(t_d, 2.0)), dtype=float)
    plot_glow_line(
        ax_c,
        t_d[sl],
        eff_trace[sl],
        theme.gold if theme.is_dark else theme.fg,
        theme,
        lw=1.8,
        label=r"Effective threshold $\theta_{\mathrm{eff}}(t)$",
        zorder=7,
    )
    ax_c.set_xlim(float(t_d[sl][0]), float(t_d[sl][-1]))
    ax_c.set_ylim(-0.3, 8.4)
    style_legend(ax_c.legend(loc="upper right", fontsize=7.1 if compact else 7.7), theme)

    # Panel D: Endogenous Geometric Thought Loops Across Seeds
    ax_d = axes[3]
    style_axes(
        ax_d,
        theme,
        tag="D",
        title="Endogenous Dark-Arena Thought Loops Across Seeds",
        xlabel=r"Unwrapped $x$ (arena units)",
        ylabel=r"Unwrapped $y$ (arena units)",
    )
    ax_d.set_aspect("equal", adjustable="box")
    seed_styles = {
        0: (theme.cyan, "Seed 0 (periodic loop)"),
        1: (theme.coral, "Seed 1 (stuttering aperiodic)"),
        2: (theme.emerald, "Seed 2 (symmetric rosette)"),
    }
    for sd, ux, uy in data["seed_tracks"]:
        col, lbl = seed_styles[sd]
        shift = (sd - 1) * 26.0
        plot_glow_line(ax_d, ux + shift, uy, col, theme, lw=1.85, alpha=0.90, label=lbl, zorder=5 + sd * 0.2)
        plot_glow_scatter(ax_d, [ux[-1] + shift], [uy[-1]], col, theme, size=32, zorder=7)
    ax_d.set_xlim(-56.0, 44.0)
    ax_d.set_ylim(-8.0, 48.0)
    style_legend(ax_d.legend(loc="upper right", fontsize=7.0 if compact else 7.6), theme)

    _add_axiom1_footer(
        fig,
        theme,
        spec,
        "Zero external stimulus & zero noise: caudate integral control stabilizes itinerant trains of thought",
        aspect,
        short_detail="Zero stimulus & zero noise: caudate control stabilizes itinerant trains of thought",
    )
    return spec


# ---------------------------------------------------------------------------
# Public Dispatcher & Showcase Builder for Series I (Phase Portraits)
# ---------------------------------------------------------------------------

_RENDERERS: dict[str, Callable[[Figure, list[mpl.axes.Axes], Theme, str], ExperimentSpec]] = {
    "v2a_fear": _render_v2a_fear,
    "v3a_love": _render_v3a_love,
    "v4a_orbiter": _render_v4a_orbiter,
    "v5_moth": _render_v5_moth,
    "v5_forager": _render_v5_forager,
    "v7_wary_moth": _render_v7_wary_moth,
    "v11_cautious_driver": _render_v11_cautious_driver,
    "v12_daydreamer": _render_v12_daydreamer,
}


def build_phase_portrait_poster(
    experiment_id: str,
    *,
    theme_name: str = "cosmic",
    aspect: str = "16:9",
    dpi: int | None = None,
) -> Figure:
    """Build a pure Axiom-1 4-panel Strogatz phase-portrait plate for strictly 1 vehicle."""
    if experiment_id not in PHASE_EXPERIMENTS or experiment_id not in _RENDERERS:
        raise ValueError(
            f"Unknown phase-portrait experiment '{experiment_id}'. Choose from: "
            f"{', '.join(PHASE_PORTRAIT_EXPERIMENTS)}"
        )

    spec = validate_single_experiment_axiom(PHASE_EXPERIMENTS[experiment_id])
    theme = get_theme(theme_name)
    with mpl.rc_context(theme_rc_params(theme)):
        fig, _, _ = _new_figure(aspect, "poster", theme, dpi)
        compact = aspect != "16:9"
        add_poster_header(
            fig,
            theme,
            spec.title,
            spec.subtitle,
            kicker=f"STROGATZ DEEP PHASE PORTRAIT · AXIOM 1 PURE EXPERIMENT [{spec.experiment_id.upper()}]",
            compact=compact,
        )
        axes = _poster_axes_4(fig, aspect)
        validated_spec = _RENDERERS[experiment_id](fig, axes, theme, aspect)
        return _tag_axiom1_figure(fig, validated_spec)


def render_phase_portraits(
    *,
    experiment: str = "all",
    theme: str = "cosmic",
    aspect: str = "16:9",
    out_dir: str | Path = ".",
    dpi: int | None = None,
    fmt: str = "png",
    suffix: str = "",
    skip_existing: bool = False,
) -> list[Path]:
    """Render 1-experiment-per-file 4-panel deep phase portraits (``<id>_phase[suffix].<fmt>``)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    theme_obj = get_theme(theme)
    _, _, default_dpi = _resolve_figsize(aspect, "poster")
    actual_dpi = dpi if dpi is not None else default_dpi

    exp_ids = _resolve_portrait_list(experiment)
    written: list[Path] = []
    for exp_id in exp_ids:
        dest = out_path / f"{exp_id}_phase{suffix}.{fmt.lower()}"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            written.append(dest)
            continue
        fig = build_phase_portrait_poster(
            exp_id,
            theme_name=theme,
            aspect=aspect,
            dpi=actual_dpi,
        )
        _save_figure(fig, dest, theme_obj, actual_dpi)
        written.append(dest)
    return written


def render_phase_portraits_showcase(
    *,
    out_dir: str | Path = "experiments/strogatz-dynamics/media",
    skip_existing: bool = False,
) -> list[Path]:
    """Populate Series I under ``<out-dir>/phase_portraits/`` (8 Cosmic + 8 Textbook 4K PNGs)."""
    portraits_dir = Path(out_dir) / "phase_portraits"
    portraits_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(
        render_phase_portraits(
            experiment="all",
            theme="cosmic",
            aspect="16:9",
            out_dir=portraits_dir,
            fmt="png",
            suffix="",
            skip_existing=skip_existing,
        )
    )
    written.extend(
        render_phase_portraits(
            experiment="all",
            theme="textbook",
            aspect="16:9",
            out_dir=portraits_dir,
            fmt="png",
            suffix="_textbook",
            skip_existing=skip_existing,
        )
    )
    return written
