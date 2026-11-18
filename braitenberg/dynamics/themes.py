"""Dual aesthetic engine for Braitenberg dynamical systems visualizations.

Provides two publication- and social-media-grade visual themes:
- ``cosmic`` (default): Deep abyssal space-indigo background, multi-pass neon glow
  strokes, bioluminescent cyan/coral/amber/emerald/violet accents, and custom
  abyssal-to-cyan-to-solar-gold colormaps for FTLE ridges and stimulus fields.
- ``textbook``: Classic Steven Strogatz *Nonlinear Dynamics and Chaos* print plate
  with warm ivory paper, crisp charcoal ink, editorial serif typography, and
  canonical fixed-point markers (filled disc = stable, open circle = unstable,
  cross-ticked open circle = saddle).

Never imports ``pygame`` (per AGENTS.md architecture rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.patheffects as pe
import numpy as np


ASPECT_SPECS: dict[str, dict[str, tuple[float, float, int]]] = {
    "16:9": {
        "poster": (12.8, 7.2, 300),   # 3840 x 2160 (4K UHD)
        "mp4": (12.8, 7.2, 150),      # 1920 x 1080 (1080p60)
        "gif": (12.8, 7.2, 82),       # 1050 x 590 (exact 12.8x7.2 layout parity!)
    },
    "1:1": {
        "poster": (9.0, 9.0, 240),    # 2160 x 2160 (4K-class square)
        "mp4": (9.0, 9.0, 120),       # 1080 x 1080 (Instagram / X square)
        "gif": (9.0, 9.0, 80),        # 720 x 720 (exact 9.0x9.0 layout parity!)
    },
    "9:16": {
        "poster": (7.2, 12.8, 300),   # 2160 x 3840 (4K Vertical)
        "mp4": (7.2, 12.8, 150),      # 1080 x 1920 (Reels / Shorts / TikTok)
        "gif": (7.2, 12.8, 82),       # 590 x 1050 (exact 7.2x12.8 layout parity!)
    },
}


@dataclass(frozen=True)
class Theme:
    """Complete color, typography, and glow specification for a visual theme."""

    name: str
    is_dark: bool
    fig_bg: str
    ax_bg: str
    panel_bg: str
    fg: str
    fg_muted: str
    fg_dim: str
    spine: str
    grid: str
    grid_alpha: float
    cyan: str
    coral: str
    amber: str
    emerald: str
    violet: str
    indigo: str
    gold: str
    stream: str
    stream_alpha: float
    font_family: str
    mathtext_fontset: str
    glow_enabled: bool

    @property
    def palette(self) -> tuple[str, ...]:
        return (self.cyan, self.coral, self.amber, self.emerald, self.violet, self.indigo)


COSMIC_THEME = Theme(
    name="cosmic",
    is_dark=True,
    fig_bg="#060818",
    ax_bg="#0a0f29",
    panel_bg="#0e1638",
    fg="#f1f5f9",
    fg_muted="#cbd5e1",
    fg_dim="#94a3b8",
    spine="#293b63",
    grid="#1e315c",
    grid_alpha=0.34,
    cyan="#38bdf8",
    coral="#fb7185",
    amber="#fbbf24",
    emerald="#34d399",
    violet="#c084fc",
    indigo="#818cf8",
    gold="#fde047",
    stream="#47669c",
    stream_alpha=0.52,
    font_family="DejaVu Sans",
    mathtext_fontset="stixsans",
    glow_enabled=True,
)


TEXTBOOK_THEME = Theme(
    name="textbook",
    is_dark=False,
    fig_bg="#fcfbf7",
    ax_bg="#faf8f1",
    panel_bg="#f3efe4",
    fg="#18181b",
    fg_muted="#3f3f46",
    fg_dim="#52525b",
    spine="#18181b",
    grid="#d6d3c9",
    grid_alpha=0.55,
    cyan="#0369a1",
    coral="#be123c",
    amber="#b45309",
    emerald="#047857",
    violet="#6d28d9",
    indigo="#3730a3",
    gold="#d97706",
    stream="#71717a",
    stream_alpha=0.55,
    font_family="DejaVu Serif",
    mathtext_fontset="stix",
    glow_enabled=False,
)


THEMES: dict[str, Theme] = {
    "cosmic": COSMIC_THEME,
    "textbook": TEXTBOOK_THEME,
}


def get_theme(name: str = "cosmic") -> Theme:
    """Return a validated :class:`Theme` by name (``'cosmic'`` or ``'textbook'``)."""
    key = name.strip().lower()
    if key not in THEMES:
        raise ValueError(f"Unknown theme '{name}'. Choose from: {', '.join(THEMES)}")
    return THEMES[key]


def aspect_figsize(aspect: str = "16:9", target: str = "poster") -> tuple[float, float, int]:
    """Return ``(width_inches, height_inches, default_dpi)`` for an aspect ratio & output target."""
    if aspect not in ASPECT_SPECS:
        raise ValueError(f"Unknown aspect '{aspect}'. Choose from: {', '.join(ASPECT_SPECS)}")
    spec = ASPECT_SPECS[aspect]
    return spec.get(target, spec["poster"])


def theme_rc_params(theme: Theme) -> dict[str, object]:
    """Return a context-safe dictionary of Matplotlib rcParams for ``theme``."""
    return {
        "figure.facecolor": theme.fig_bg,
        "axes.facecolor": theme.ax_bg,
        "savefig.facecolor": theme.fig_bg,
        "axes.edgecolor": theme.spine,
        "axes.labelcolor": theme.fg_muted,
        "axes.titlecolor": theme.fg,
        "xtick.color": theme.fg_dim,
        "ytick.color": theme.fg_dim,
        "text.color": theme.fg,
        "font.family": theme.font_family,
        "mathtext.fontset": theme.mathtext_fontset,
        "axes.linewidth": 0.9 if theme.is_dark else 1.15,
        "xtick.major.width": 0.8 if theme.is_dark else 1.0,
        "ytick.major.width": 0.8 if theme.is_dark else 1.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.8,
        "legend.frameon": True,
        "legend.facecolor": theme.panel_bg,
        "legend.edgecolor": theme.spine,
        "legend.fontsize": 8.2,
    }


# --- Custom Colormaps ---


def ftle_cmap(theme: Theme) -> LinearSegmentedColormap:
    """Colormap for Finite-Time Lyapunov Exponent (FTLE) ridges."""
    if theme.is_dark:
        stops = [
            (0.00, "#060818"),
            (0.22, "#0b1938"),
            (0.45, "#0e486b"),
            (0.68, "#189ab4"),
            (0.84, "#38bdf8"),
            (0.94, "#fbbf24"),
            (1.00, "#fef9c3"),
        ]
    else:
        stops = [
            (0.00, "#fcfbf7"),
            (0.28, "#ebe7dc"),
            (0.55, "#93c5fd"),
            (0.78, "#1d4ed8"),
            (0.92, "#991b1b"),
            (1.00, "#450a0a"),
        ]
    return LinearSegmentedColormap.from_list(f"braitenberg_ftle_{theme.name}", stops, N=256)


def field_cmap(theme: Theme) -> LinearSegmentedColormap:
    """Colormap for ambient stimulus intensity fields (inspired by render.py HEAT_STOPS)."""
    if theme.is_dark:
        stops = [
            (0.00, "#080c24"),
            (0.28, "#17174a"),
            (0.58, "#581c5c"),
            (0.82, "#b84a2c"),
            (1.00, "#fde68a"),
        ]
    else:
        stops = [
            (0.00, "#faf8f1"),
            (0.35, "#f3ebd7"),
            (0.70, "#f5d096"),
            (1.00, "#e59849"),
        ]
    return LinearSegmentedColormap.from_list(f"braitenberg_field_{theme.name}", stops, N=256)


def basin_colors(theme: Theme, n: int = 4) -> list[str]:
    """Distinct attractor basin colors for up to ``n`` stimulus sources."""
    base = (
        [theme.cyan, theme.coral, theme.amber, theme.emerald, theme.violet, theme.indigo]
        if theme.is_dark
        else ["#2563eb", "#dc2626", "#d97706", "#059669", "#7c3aed", "#4f46e5"]
    )
    return [base[i % len(base)] for i in range(max(1, n))]


def phase_cmap(theme: Theme) -> LinearSegmentedColormap:
    """Cyclic colormap for neural ring phase ``Phi in [-pi, pi]``."""
    if theme.is_dark:
        stops = [
            (0.00, "#38bdf8"),
            (0.25, "#c084fc"),
            (0.50, "#fb7185"),
            (0.75, "#fbbf24"),
            (1.00, "#38bdf8"),
        ]
    else:
        stops = [
            (0.00, "#0369a1"),
            (0.25, "#6d28d9"),
            (0.50, "#be123c"),
            (0.75, "#b45309"),
            (1.00, "#0369a1"),
        ]
    return LinearSegmentedColormap.from_list(f"braitenberg_phase_{theme.name}", stops, N=256)


# --- Shannon Codeword Palettes, Entropy & Activity Colormaps ---


CODEWORD_PALETTES: dict[str, tuple[str, ...]] = {
    "cosmic": (
        "#38bdf8",  # C0: Electric Sky Cyan
        "#fb7185",  # C1: Bioluminescent Coral Rose
        "#fbbf24",  # C2: Solar Amber Gold
        "#34d399",  # C3: Neon Emerald
        "#c084fc",  # C4: Luminous Orchid Violet
        "#f97316",  # C5: Tangerine Plasma
        "#2dd4bf",  # C6: Seafoam Teal
        "#e879f9",  # C7: Electric Fuchsia
    ),
    "textbook": (
        "#0284c7",  # C0: Cerulean Ink
        "#e11d48",  # C1: Crimson Vermilion
        "#d97706",  # C2: Ochre Amber
        "#059669",  # C3: Viridian Emerald
        "#7c3aed",  # C4: Royal Violet
        "#ea580c",  # C5: Burnt Sienna
        "#0d9488",  # C6: Deep Teal
        "#a21caf",  # C7: Plum Magenta
    ),
}

RESIDUAL_CODEWORD_COLORS: dict[str, str] = {
    "cosmic": "#64748b",
    "textbook": "#71717a",
}

HERO_RING_COLORS: dict[str, str] = {
    "cosmic": "#78dcff",    # Homage to render.py SELECTED_RING = (120, 220, 255)
    "textbook": "#0284c7",
}

_COSMIC_HEAT_STOPS = np.array(
    [
        [8.0, 12.0, 40.0],
        [40.0, 30.0, 110.0],
        [150.0, 40.0, 110.0],
        [230.0, 100.0, 50.0],
        [255.0, 230.0, 140.0],
    ],
    dtype=float,
) / 255.0

_TEXTBOOK_HEAT_STOPS = np.array(
    [
        [250.0, 248.0, 241.0],
        [254.0, 243.0, 199.0],
        [252.0, 211.0, 77.0],
        [245.0, 158.0, 11.0],
        [180.0, 83.0, 9.0],
    ],
    dtype=float,
) / 255.0


def codeword_colors(theme: Theme, n: int = 8) -> list[str]:
    """Return ``n`` high-contrast Shannon joint-codeword palette hues for ``theme``."""
    pal = CODEWORD_PALETTES.get(theme.name, CODEWORD_PALETTES["cosmic"])
    return [pal[i % len(pal)] for i in range(max(1, n))]


def codeword_color(theme: Theme, codeword_id: int, max_codewords: int = 8) -> str:
    """Map an integer codeword index ``c`` to its canonical palette hue (or residual slate)."""
    pal = CODEWORD_PALETTES.get(theme.name, CODEWORD_PALETTES["cosmic"])
    cid = int(codeword_id)
    if cid < 0 or cid >= max_codewords:
        return RESIDUAL_CODEWORD_COLORS.get(theme.name, "#64748b")
    return pal[cid % len(pal)]


def hero_ring_color(theme: Theme) -> str:
    """Signature cyan selection ring color marking ``Hero #0`` across all panels."""
    return HERO_RING_COLORS.get(theme.name, theme.cyan)


def entropy_cmap(theme: Theme) -> LinearSegmentedColormap:
    """Colormap for binary/marginal Shannon entropy ``H in [0, 1]`` bit."""
    if theme.is_dark:
        stops = [
            (0.00, "#0a0f29"),
            (0.20, "#1e293b"),
            (0.50, "#7c3aed"),
            (0.78, "#f43f5e"),
            (1.00, "#fde047"),
        ]
    else:
        stops = [
            (0.00, "#faf8f1"),
            (0.25, "#e2e8f0"),
            (0.55, "#7c3aed"),
            (0.82, "#be123c"),
            (1.00, "#b45309"),
        ]
    return LinearSegmentedColormap.from_list(f"braitenberg_entropy_{theme.name}", stops, N=256)


def barcode_cmap(theme: Theme, *, is_learning: bool = False) -> LinearSegmentedColormap:
    """Colormap for 1D Empirical Density Barcode strips over ``[0, 1]``."""
    if theme.is_dark:
        if is_learning:
            stops = [
                (0.00, "#080e24"),
                (0.12, "#0f2e2e"),
                (0.45, "#0d9488"),
                (0.78, "#34d399"),
                (1.00, "#d1fae5"),
            ]
        else:
            stops = [
                (0.00, "#080e24"),
                (0.12, "#132347"),
                (0.42, "#0284c7"),
                (0.74, "#38bdf8"),
                (0.92, "#fbbf24"),
                (1.00, "#fef9c3"),
            ]
    else:
        if is_learning:
            stops = [
                (0.00, "#f1ede2"),
                (0.18, "#d1fae5"),
                (0.55, "#10b981"),
                (0.85, "#047857"),
                (1.00, "#064e3b"),
            ]
        else:
            stops = [
                (0.00, "#f1ede2"),
                (0.18, "#dbeafe"),
                (0.52, "#2563eb"),
                (0.82, "#1d4ed8"),
                (1.00, "#1e1b4b"),
            ]
    tag = "learn" if is_learning else "sig"
    return LinearSegmentedColormap.from_list(f"braitenberg_barcode_{tag}_{theme.name}", stops, N=256)


def activity_fill_rgba(levels: np.ndarray | float, theme: Theme) -> np.ndarray:
    """Vectorized Braitenberg signal fill mapping ``[0, 1]`` -> RGBA (homage to ``render.py``)."""
    stops = _COSMIC_HEAT_STOPS if theme.is_dark else _TEXTBOOK_HEAT_STOPS
    arr = np.clip(np.asarray(levels, dtype=float), 0.0, 1.0)
    pos = arr * (len(stops) - 1)
    idx = np.minimum(pos.astype(int), len(stops) - 2)
    frac = (pos - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    alpha = np.ones((*arr.shape, 1), dtype=float)
    return np.concatenate([rgb, alpha], axis=-1)


def activity_fill_color(level: float, theme: Theme) -> tuple[float, float, float, float]:
    """Scalar convenience wrapper around :func:`activity_fill_rgba`."""
    rgba = activity_fill_rgba(float(level), theme)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


# --- Axes, Typography & Halo Helpers ---


def text_halo(theme: Theme, width: float = 2.8) -> list[pe.AbstractPathEffect]:
    """Subtle background stroke outline so labels remain crisp over streamlines/contours."""
    return [pe.withStroke(linewidth=width, foreground=theme.ax_bg, alpha=0.92)]


def badge_bbox(theme: Theme, alpha: float = 0.88) -> dict[str, object]:
    """Translucent pill container for in-axes telemetry & callout annotations."""
    return {
        "boxstyle": "round,pad=0.32",
        "facecolor": theme.panel_bg,
        "edgecolor": theme.spine,
        "linewidth": 0.8,
        "alpha": alpha if theme.is_dark else 0.93,
    }


def style_axes(
    ax: mpl.axes.Axes,
    theme: Theme,
    *,
    title: str | None = None,
    tag: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    top_right_spines: bool = False,
) -> None:
    """Apply consistent Strogatz/Cosmic styling, hairlines, and title hierarchy to ``ax``."""
    fig_w = ax.figure.get_figwidth() if ax.figure is not None else 12.8
    scale = min(1.0, max(0.76, fig_w / 11.6))

    ax.set_facecolor(theme.ax_bg)
    for side, spine in ax.spines.items():
        if not top_right_spines and side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(theme.spine)
            spine.set_linewidth(0.9 if theme.is_dark else 1.1)

    ax.tick_params(colors=theme.fg_dim, labelsize=8.3 * scale, length=3.2, pad=3)
    if grid:
        ax.grid(True, color=theme.grid, alpha=theme.grid_alpha, linewidth=0.6, linestyle="-", zorder=0)
    else:
        ax.grid(False)

    if xlabel is not None:
        ax.set_xlabel(xlabel, color=theme.fg_muted, fontsize=9.1 * scale, labelpad=3.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=theme.fg_muted, fontsize=9.1 * scale, labelpad=3.5)

    if title is not None or tag is not None:
        header = f"{tag}  {title}" if (tag and title) else (title or tag or "")
        ax.set_title(
            header,
            loc="left",
            color=theme.fg,
            fontsize=10.1 * scale,
            fontweight="bold",
            pad=7.5,
        )


def add_poster_header(
    fig: mpl.figure.Figure,
    theme: Theme,
    title: str,
    subtitle: str,
    *,
    kicker: str = "SYNTHETIC PSYCHOLOGY × NONLINEAR DYNAMICS",
    y_top: float = 0.974,
    compact: bool = False,
) -> None:
    """Render an editorial poster masthead scaled cleanly to any canvas width."""
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    w_scale = min(1.0, max(0.56, fig_w / 12.2))
    is_tall = (fig_h / max(fig_w, 1e-6)) > 1.35

    title_fs = (15.5 if compact else 17.2) * w_scale
    sub_fs = (9.2 if compact else 10.0) * max(w_scale, 0.68)
    kicker_fs = 8.0 * max(w_scale, 0.72)

    x_left = 0.058 if not is_tall else 0.065
    dy1 = 0.017 if is_tall else (0.020 if compact else 0.022)
    dy2 = 0.043 if is_tall else (0.049 if compact else 0.055)

    fig.text(
        x_left,
        y_top,
        kicker,
        color=theme.cyan if theme.is_dark else theme.coral,
        fontsize=kicker_fs,
        fontweight="bold",
        va="top",
        ha="left",
    )
    fig.text(
        x_left,
        y_top - dy1,
        title,
        color=theme.fg,
        fontsize=title_fs,
        fontweight="bold",
        va="top",
        ha="left",
    )
    fig.text(
        x_left,
        y_top - dy2,
        subtitle,
        color=theme.fg_muted,
        fontsize=sub_fs,
        va="top",
        ha="left",
    )


def style_legend(legend: mpl.legend.Legend | None, theme: Theme) -> None:
    """Style a Matplotlib legend card with clean contrast and no visual clutter."""
    if legend is None:
        return
    legend.set_zorder(12)
    frame = legend.get_frame()
    frame.set_facecolor(theme.panel_bg)
    frame.set_edgecolor(theme.spine)
    frame.set_alpha(0.90 if theme.is_dark else 0.94)
    frame.set_linewidth(0.8)
    for txt in legend.get_texts():
        txt.set_color(theme.fg_muted)


# --- Multi-Pass Neon Glow & Textbook Vector Primitives ---


def plot_glow_line(
    ax: mpl.axes.Axes,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    color: str,
    theme: Theme,
    *,
    lw: float = 2.0,
    ls: str = "-",
    alpha: float = 0.96,
    label: str | None = None,
    zorder: float = 5,
) -> list[mpl.lines.Line2D]:
    """Draw a 3-pass bioluminescent neon glow curve in ``cosmic`` or crisp ink line in ``textbook``."""
    artists: list[mpl.lines.Line2D] = []
    if theme.glow_enabled:
        for mult, halo_alpha in ((4.4, 0.10 * alpha), (2.2, 0.26 * alpha)):
            artists.append(
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=lw * mult,
                    linestyle=ls,
                    alpha=halo_alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=zorder - 0.2,
                )[0]
            )
    core = ax.plot(
        x,
        y,
        color=color,
        linewidth=lw if theme.glow_enabled else lw * 1.08,
        linestyle=ls,
        alpha=alpha,
        label=label,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )[0]
    artists.append(core)
    return artists


def plot_glow_scatter(
    ax: mpl.axes.Axes,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    color: str | Sequence[str],
    theme: Theme,
    *,
    size: float = 38.0,
    marker: str = "o",
    alpha: float = 0.96,
    edgecolor: str | None = None,
    linewidths: float = 0.8,
    label: str | None = None,
    zorder: float = 6,
) -> list[mpl.collections.PathCollection]:
    """Draw a 3-pass glowing scatter in ``cosmic`` or crisp print marker in ``textbook``."""
    cols: list[mpl.collections.PathCollection] = []
    unfilled = marker in ("x", "+", "|", "_", "1", "2", "3", "4")
    if theme.glow_enabled:
        for mult, halo_alpha in ((5.2, 0.11 * alpha), (2.4, 0.28 * alpha)):
            halo_kw = {} if unfilled else {"edgecolors": "none"}
            cols.append(
                ax.scatter(
                    x,
                    y,
                    s=size * mult,
                    c=color,
                    marker=marker,
                    alpha=halo_alpha,
                    zorder=zorder - 0.2,
                    **halo_kw,
                )
            )
    if unfilled:
        core = ax.scatter(
            x,
            y,
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            linewidths=max(linewidths, 1.5),
            label=label,
            zorder=zorder,
        )
    else:
        edge = edgecolor if edgecolor is not None else ("#ffffff" if theme.is_dark else theme.fg)
        core = ax.scatter(
            x,
            y,
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            edgecolors=edge,
            linewidths=linewidths,
            label=label,
            zorder=zorder,
        )
    cols.append(core)
    return cols


def plot_comet_trails(
    ax: mpl.axes.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: Sequence[str] | str,
    theme: Theme,
    *,
    max_lw: float = 2.1,
    alpha: float = 0.90,
    arena_size: tuple[float, float] | None = None,
    head_size: float = 22.0,
    zorder: float = 5,
) -> list[mpl.artist.Artist]:
    """Vectorized rendering of comet-tailed particle ensembles ``xs, ys`` of shape ``(T, N)``.

    Automatically masks wrap-around jumps across periodic boundaries when ``arena_size`` is set.
    """
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if xs_arr.ndim == 1:
        xs_arr = xs_arr[:, None]
        ys_arr = ys_arr[:, None]

    t_len, n_particles = xs_arr.shape
    artists: list[mpl.artist.Artist] = []
    if n_particles == 0 or t_len == 0:
        return artists

    if isinstance(colors, str):
        base_rgba = np.tile(to_rgba(colors), (n_particles, 1))
    else:
        base_rgba = np.array([to_rgba(c) for c in colors], dtype=float)

    if t_len >= 2:
        p0 = np.stack([xs_arr[:-1], ys_arr[:-1]], axis=-1)  # (T-1, N, 2)
        p1 = np.stack([xs_arr[1:], ys_arr[1:]], axis=-1)
        segs = np.stack([p0, p1], axis=2)                   # (T-1, N, 2, 2)

        valid = np.isfinite(segs).all(axis=(2, 3))
        if arena_size is not None:
            w, h = arena_size
            dx = np.abs(xs_arr[1:] - xs_arr[:-1])
            dy = np.abs(ys_arr[1:] - ys_arr[:-1])
            valid &= (dx < 0.45 * w) & (dy < 0.45 * h)

        ramp = np.linspace(0.06, 1.0, t_len - 1)[:, None]   # (T-1, 1)
        rgba = np.broadcast_to(base_rgba[None, :, :], (t_len - 1, n_particles, 4)).copy()
        rgba[:, :, 3] = alpha * (ramp ** 1.65)
        lws = np.broadcast_to(0.35 + (max_lw - 0.35) * ramp, (t_len - 1, n_particles))

        flat_segs = segs[valid]
        flat_rgba = rgba[valid]
        flat_lws = lws[valid]

        if len(flat_segs) > 0:
            if theme.glow_enabled:
                halo_rgba = flat_rgba.copy()
                halo_rgba[:, 3] *= 0.24
                halo_lc = LineCollection(
                    flat_segs,
                    colors=halo_rgba,
                    linewidths=flat_lws * 2.8,
                    capstyle="round",
                    zorder=zorder - 0.2,
                )
                ax.add_collection(halo_lc)
                artists.append(halo_lc)

            core_lc = LineCollection(
                flat_segs,
                colors=flat_rgba,
                linewidths=flat_lws,
                capstyle="round",
                zorder=zorder,
            )
            ax.add_collection(core_lc)
            artists.append(core_lc)

    hx, hy = xs_arr[-1], ys_arr[-1]
    finite_heads = np.isfinite(hx) & np.isfinite(hy)
    if np.any(finite_heads):
        head_colors = base_rgba[finite_heads].copy()
        head_colors[:, 3] = min(1.0, alpha * 1.05)
        if theme.glow_enabled:
            halo_heads = head_colors.copy()
            halo_heads[:, 3] = 0.24 * alpha
            artists.append(
                ax.scatter(
                    hx[finite_heads],
                    hy[finite_heads],
                    s=head_size * 3.6,
                    c=halo_heads,
                    edgecolors="none",
                    zorder=zorder + 0.1,
                )
            )
        artists.append(
            ax.scatter(
                hx[finite_heads],
                hy[finite_heads],
                s=head_size,
                c=head_colors,
                edgecolors="#ffffff" if theme.is_dark else theme.fg,
                linewidths=0.45,
                zorder=zorder + 0.2,
            )
        )
    return artists


def plot_strogatz_equilibrium(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    eq_type: str,
    theme: Theme,
    *,
    size: float = 64.0,
    zorder: float = 9,
) -> list[mpl.artist.Artist]:
    """Plot a fixed point following Steven Strogatz's textbook convention:

    - Stable equilibrium (``stable_node``, ``stable_spiral``, ``stable``): solid filled disc.
    - Unstable equilibrium (``unstable_node``, ``unstable_spiral``, ``unstable``, ``center``):
      open white/background-filled circle with crisp rim.
    - Saddle point (``saddle``): open circle with orthogonal saddle cross-hair ticks.
    """
    kind = eq_type.lower()
    artists: list[mpl.artist.Artist] = []

    if "saddle" in kind:
        rim = theme.amber if theme.is_dark else theme.fg
        fill = theme.ax_bg if theme.is_dark else "#ffffff"
        if theme.glow_enabled:
            artists.append(
                ax.scatter([x], [y], s=size * 3.8, c=[rim], alpha=0.20, edgecolors="none", zorder=zorder - 0.2)
            )
        artists.append(
            ax.scatter(
                [x],
                [y],
                s=size * 1.08,
                facecolors=fill,
                edgecolors=rim,
                linewidths=1.8,
                marker="o",
                zorder=zorder,
            )
        )
        artists.append(
            ax.scatter(
                [x],
                [y],
                s=size * 1.45,
                c=[rim],
                marker="+",
                linewidths=1.5,
                zorder=zorder + 0.1,
            )
        )
    elif kind.startswith("stable"):
        col = theme.emerald if theme.is_dark else theme.fg
        edge = "#ffffff" if theme.is_dark else theme.fg
        if theme.glow_enabled:
            artists.append(
                ax.scatter([x], [y], s=size * 3.8, c=[col], alpha=0.22, edgecolors="none", zorder=zorder - 0.2)
            )
        artists.append(
            ax.scatter([x], [y], s=size, facecolors=col, edgecolors=edge, linewidths=1.2, zorder=zorder)
        )
    else:
        rim = theme.coral if theme.is_dark else theme.fg
        fill = theme.ax_bg if theme.is_dark else "#ffffff"
        if theme.glow_enabled:
            artists.append(
                ax.scatter([x], [y], s=size * 3.6, c=[rim], alpha=0.18, edgecolors="none", zorder=zorder - 0.2)
            )
        artists.append(
            ax.scatter(
                [x],
                [y],
                s=size,
                facecolors=fill,
                edgecolors=rim,
                linewidths=1.8,
                zorder=zorder,
            )
        )
    return artists


def plot_lamp_beacon(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    theme: Theme,
    *,
    color: str | None = None,
    size: float = 95.0,
    label: str | None = None,
    zorder: float = 8,
) -> list[mpl.artist.Artist]:
    """Render an isotropic stimulus lamp beacon with warm concentric radiance."""
    c = color or (theme.gold if theme.is_dark else theme.amber)
    artists: list[mpl.artist.Artist] = []
    if theme.glow_enabled:
        for mult, a in ((8.5, 0.08), (4.0, 0.18), (2.0, 0.38)):
            artists.append(
                ax.scatter([x], [y], s=size * mult, c=[c], alpha=a, edgecolors="none", zorder=zorder - 0.2)
            )
    else:
        artists.append(
            ax.scatter(
                [x],
                [y],
                s=size * 2.3,
                facecolors="none",
                edgecolors=c,
                linewidths=1.1,
                alpha=0.65,
                zorder=zorder - 0.1,
            )
        )
    core = ax.scatter(
        [x],
        [y],
        s=size,
        c=["#fffbeb" if theme.is_dark else c],
        edgecolors=c if theme.is_dark else theme.fg,
        linewidths=1.5,
        label=label,
        zorder=zorder,
    )
    artists.append(core)
    return artists


def plot_hero_beacon(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    theme: Theme,
    *,
    fill_color: str | None = None,
    size: float = 92.0,
    label: str | None = None,
    zorder: float = 10,
) -> list[mpl.artist.Artist]:
    """Render the signature Cyan Selection Target Ring around ``Hero #0``."""
    ring_col = hero_ring_color(theme)
    core_col = fill_color or ring_col
    artists: list[mpl.artist.Artist] = []
    if theme.glow_enabled:
        artists.append(
            ax.scatter(
                [x],
                [y],
                s=size * 4.2,
                c=[ring_col],
                alpha=0.22,
                edgecolors="none",
                zorder=zorder - 0.3,
            )
        )
    artists.append(
        ax.scatter(
            [x],
            [y],
            s=size * 2.15,
            facecolors="none",
            edgecolors=ring_col,
            linewidths=1.8,
            zorder=zorder - 0.1,
        )
    )
    artists.append(
        ax.scatter(
            [x],
            [y],
            s=size * 0.72,
            facecolors=core_col,
            edgecolors="#ffffff" if theme.is_dark else theme.fg,
            linewidths=1.2,
            label=label,
            zorder=zorder,
        )
    )
    return artists
