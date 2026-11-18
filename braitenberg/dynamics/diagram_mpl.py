"""Three-Tier Hero + Claude Shannon Ensemble Nervous System renderer in Matplotlib.

Upgrades the classic Braitenberg live schematic (``specs/vehicle-diagram.md`` and
``specs/images/vehicle9-shy.gif``) into a lossless, information-theoretic 3-tier panel:

- **Tier A (Top ~52%): Braitenberg Body Schematic**
  - Inner core of every part and active wire pulse reflects **Hero #0**'s exact
    instantaneous state (preserving single-circuit causal legibility).
  - Outer segmented ring (**Joint Codebook Arc-Ring**, angular span ``360 * p_c``)
    colors each part's active probability mass by exact **Shannon Codeword Hue ``c``**,
    preserving mutual exclusivity and synergy without mean-collapse distortion.
  - Subtle outer halo scales strictly with marginal **Shannon Entropy ``H(A_k)``** in
    ``[0, 1]`` bit (so constant bias units such as ``awake = 1`` emit zero halo).
  - Bundled wires scale stroke width & glow by **Channel Mutual Information /
    Transfer Entropy** (``I(S; D)`` / ``T_{S -> D}`` in bits).
- **Tier B (Middle ~26%): 1D Empirical Density Barcodes + Shannon Telemetry Strip**
  - Displays live ensemble ``H(W)``, Watanabe ``Synergy TC``, and ``I(Phase; W)`` in bits,
    above 1D empirical density spectrograph strips over ``[0, 1]`` (with vertical
    ``w_c = 0.7`` learning threshold ticks and bright ``Hero #0`` cursor ticks).
- **Tier C (Bottom ~22%): Lossless Sorted N x M Swarm Microstate Bit-Raster**
  - Displays every vehicle's exact binary codeword vector sorted by codeword & phase,
    colored by Codeword Hue ``c``, with ``Hero #0``'s row marked in cyan.

Designed with zero ``ax.clear()`` per-frame artist mutation so 60 FPS animations and
4K print/cosmic posters share identical geometry and render in milliseconds per frame.
Never imports ``pygame`` (per AGENTS.md).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge
import matplotlib.patheffects as pe
import numpy as np

from braitenberg.definition import ACTIVE_STRENGTH
from braitenberg.diagram import LEARNING_LABELS, bundles
from braitenberg.dynamics.themes import (
    Theme,
    activity_fill_color,
    activity_fill_rgba,
    barcode_cmap,
    codeword_color,
    entropy_cmap,
    hero_ring_color,
)
from braitenberg.vehicle import Diagram, Part, Wire

NEURAL_VIEWS: tuple[str, ...] = ("composite", "codebook", "entropy", "raster")


# ---------------------------------------------------------------------------
# Layout & Snapshot Adapter Helpers
# ---------------------------------------------------------------------------


def make_neural_tier_axes(
    fig: mpl.figure.Figure,
    slot: SubplotSpec,
    *,
    neural_view: str = "composite",
    aspect: str = "16:9",
) -> tuple[mpl.axes.Axes, mpl.axes.Axes, mpl.axes.Axes]:
    """Subdivide ``slot`` into ``(ax_schem, ax_barcodes, ax_raster)``."""
    view = neural_view.strip().lower()
    if view == "raster":
        ratios = [0.46, 0.21, 0.33]
    elif aspect == "9:16":
        ratios = [0.54, 0.24, 0.22]
    else:
        ratios = [0.57, 0.23, 0.20]

    sub = slot.subgridspec(3, 1, height_ratios=ratios, hspace=0.13 if aspect != "9:16" else 0.15)
    ax_schem = fig.add_subplot(sub[0, 0])
    ax_bar = fig.add_subplot(sub[1, 0])
    ax_ras = fig.add_subplot(sub[2, 0])
    return ax_schem, ax_bar, ax_ras


def _extract_cell_grids(
    snapshot: dict[str, Any], part_name: str, shape: tuple[int, ...]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Look up ``(prob_grid, entropy_grid)`` for camera or array ``part_name``."""
    rows, cols = shape if len(shape) == 2 else (1, shape[0])

    prob_grid = None
    for key in ("grid_prob", "cell_prob", "part_grid_prob", "array_prob"):
        d = snapshot.get(key)
        if isinstance(d, dict) and part_name in d:
            prob_grid = np.asarray(d[part_name], dtype=float).reshape(rows, cols)
            break

    ent_grid = None
    for key in ("grid_entropy", "cell_entropy", "part_grid_entropy", "array_entropy"):
        d = snapshot.get(key)
        if isinstance(d, dict) and part_name in d:
            ent_grid = np.asarray(d[part_name], dtype=float).reshape(rows, cols)
            break

    return prob_grid, ent_grid


def _extract_part_arcs(snapshot: dict[str, Any], part_name: str, part_index: int) -> list[tuple[int, float]]:
    """Return ``[(codeword_id, prob_mass), ...]`` for ``part_name`` (including V4a fallback)."""
    arcs_map = snapshot.get("part_codeword_arcs") or {}
    raw = arcs_map.get(part_name)
    if raw is None:
        # Memoryless V4a sensorimotor aliases in shannon.py:
        alias_map = {
            "left_eye": "left_eye_high",
            "left_sensor": "left_eye_high",
            "right_eye": "right_eye_high",
            "right_sensor": "right_eye_high",
            "left_motor": "turn_right",
            "right_motor": "turn_left",
        }
        alt = alias_map.get(part_name)
        if alt and alt in arcs_map:
            raw = arcs_map[alt]
    if not raw:
        return []
    out: list[tuple[int, float]] = []
    for item in raw:
        if isinstance(item, dict):
            cid = int(item.get("codeword_id", item.get("id", 0)))
            pm = float(item.get("prob", item.get("mass", 0.0)))
        else:
            cid, pm = int(item[0]), float(item[1])
        if pm > 1e-4:
            out.append((cid, pm))
    return out


def _extract_part_entropy(snapshot: dict[str, Any], part_name: str) -> float:
    ent_map = snapshot.get("part_entropy") or {}
    if part_name in ent_map:
        return float(np.clip(ent_map[part_name], 0.0, 1.0))
    alias_map = {
        "left_eye": "left_eye_high",
        "right_eye": "right_eye_high",
        "left_motor": "turn_right",
        "right_motor": "turn_left",
    }
    alt = alias_map.get(part_name)
    if alt and alt in ent_map:
        return float(np.clip(ent_map[alt], 0.0, 1.0))
    return 0.0


def _lookup_wire_bits(snapshot: dict[str, Any], w: Wire, idx: int) -> float:
    """Extract channel MI / Transfer Entropy ``max(I(S;D), T_{S->D})`` in bits for wire ``w``."""
    candidates = (
        snapshot.get("wire_info_bits"),
        snapshot.get("wire_mi_bits"),
        snapshot.get("wire_te_bits"),
    )
    keys_to_try = (
        (w.source, w.target, w.sign),
        (w.source, w.target, w.source_port, w.target_port, w.sign),
        (w.source, w.target),
        f"{w.source}->{w.target}",
        f"{w.source}->{w.target}:{'+' if w.sign > 0 else '-'}",
    )
    best = 0.0
    for store in candidates:
        if store is None:
            continue
        if isinstance(store, dict):
            for k in keys_to_try:
                if k in store:
                    val = store[k]
                    if isinstance(val, dict):
                        v_num = float(
                            val.get("wire_info_bits", max(val.get("mi_bits", 0.0), val.get("te_bits", 0.0)))
                        )
                    else:
                        v_num = float(val)
                    best = max(best, v_num)
                    break
        elif isinstance(store, (list, tuple, np.ndarray)) and idx < len(store):
            item = store[idx]
            if isinstance(item, dict):
                v_num = float(item.get("wire_info_bits", max(item.get("mi_bits", 0.0), item.get("te_bits", 0.0))))
            else:
                v_num = float(item)
            best = max(best, v_num)
    return float(max(0.0, best))


def _select_featured_barcodes(
    diagram: Diagram, barcodes: dict[str, dict[str, Any]], max_rows: int = 6
) -> list[str]:
    """Select the ``max_rows`` most informative continuous/plastic signals deterministically."""
    if not barcodes:
        return []
    keys = list(barcodes.keys())
    if len(keys) <= max_rows:
        return keys

    part_kind = {p.name: p.kind for p in diagram.parts}
    priority_names = {
        "burned",
        "pain",
        "full",
        "hungry",
        "seeking",
        "symmetric",
        "front_ahead",
        "fleeing",
        "turn_away",
        "run",
        "danger",
        "avoid",
        "caudate",
    }

    def score(k: str) -> tuple[int, float, int]:
        entry = barcodes[k]
        is_learn = bool(entry.get("is_learning", False)) or ("->" in k)
        kind = part_kind.get(k, "")
        ent = float(entry.get("entropy_bits", 0.0))
        if is_learn:
            tier = 0
        elif kind in ("camera", "sensor", "control"):
            tier = 1
        elif kind == "array":
            tier = 2
        elif kind == "motor":
            tier = 3
        elif k in priority_names:
            tier = 4
        else:
            tier = 5
        return (tier, -ent, keys.index(k))

    ranked = sorted(keys, key=score)
    chosen = set(ranked[:max_rows])
    # Preserve canonical top-to-bottom circuit reading order (sensors -> units -> motors -> learning)
    return [k for k in keys if k in chosen]


# ---------------------------------------------------------------------------
# Smart Deterministic Non-Overlapping Part Label Solver
# ---------------------------------------------------------------------------


def _part_radii(p: Part, bw: float, bl: float, crowded: bool, very_crowded: bool = False) -> tuple[float, float]:
    """Outer visual half-extents ``(rx, ry)`` in schematic data coordinates."""
    if p.kind == "camera" and p.shape is not None:
        rows, cols = p.shape if len(p.shape) == 2 else (1, p.shape[0])
        w = 1.12 * bw
        cell = min(0.062, w / max(cols, 1))
        return (0.5 * cols * cell, 0.5 * rows * cell)
    if p.shape is not None:
        rows, cols = p.shape if len(p.shape) == 2 else (1, p.shape[0])
        w = 0.25 * bw
        cell = min(0.064, w / max(cols, 1))
        return (0.5 * cols * cell + 0.015, 0.5 * rows * cell + 0.034)
    if p.kind == "module":
        n_ports = max(len(p.inputs), len(p.outputs), 2)
        return (0.12 * n_ports, 0.11)
    if p.kind == "motor":
        return (0.120, 0.078)
    base = 0.058 if very_crowded else (0.066 if crowded else 0.076)
    ring = 0.024 if very_crowded else (0.028 if crowded else 0.034)
    return (base + ring, base + ring)


def _rect_overlap_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    if dx <= 0.0:
        return 0.0
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dy <= 0.0:
        return 0.0
    return dx * dy


def solve_part_labels(
    parts: Sequence[Part],
    wires: Sequence[Wire],
    body: tuple[float, float],
    *,
    xlim: tuple[float, float] = (-1.28, 1.28),
    ylim: tuple[float, float] = (-1.18, 1.20),
    char_w: float | None = None,
    char_h: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute collision-free label placements ``{part_name: {x, y, ha, va, bbox, leader}}``.

    Guarantees that crowded circuits such as Vehicle 9 (*Shy*, 16 parts with a vertical
    ``front_ahead -> symmetric -> fleeing`` spine) and Vehicle 5/7/12 never suffer text-on-node
    or text-on-text collisions across any aspect ratio.
    """
    bw, bl = body
    crowded = len(parts) >= 12 or any(p.shape is not None for p in parts)
    very_crowded = len(parts) >= 15
    cw = char_w if char_w is not None else (0.036 if crowded else 0.041)
    ch = char_h if char_h is not None else (0.076 if crowded else 0.084)

    pos_map: dict[str, tuple[float, float]] = {}
    rad_map: dict[str, tuple[float, float]] = {}
    node_boxes: dict[str, tuple[float, float, float, float]] = {}

    for p in parts:
        px, py = p.position[0] * bw, p.position[1] * bl
        rx, ry = _part_radii(p, bw, bl, crowded, very_crowded)
        pos_map[p.name] = (px, py)
        rad_map[p.name] = (rx, ry)
        pad = 0.014
        node_boxes[p.name] = (px - rx - pad, py - ry - pad, px + rx + pad, py + ry + pad)
        if p.kind == "sensor":
            # Include forward sensor antennae in obstacle footprint
            node_boxes[p.name] = (px - rx - pad, py - ry - pad, px + rx + pad, py + ry + 0.082)

    # Self-loop arcs also count as obstacles so labels avoid landing on self-feedback loops
    loop_boxes: list[tuple[float, float, float, float]] = []
    for w in wires:
        if w.source == w.target and w.source in pos_map:
            px, py = pos_map[w.source]
            rx, ry = rad_map[w.source]
            side = -1.0 if px < 0.0 else 1.0
            lx = px + side * (rx + 0.055)
            ly = py + ry * 0.55
            loop_boxes.append((lx - 0.055, ly - 0.055, lx + 0.055, ly + 0.055))

    # Order parts by local crowding (most constrained central units first!)
    def crowding_score(p: Part) -> float:
        px, py = pos_map[p.name]
        s = 0.0
        for q in parts:
            if q.name == p.name:
                continue
            qx, qy = pos_map[q.name]
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            s += 1.0 / max(d2, 0.02)
        return s

    ordered = [p for p in sorted(parts, key=crowding_score, reverse=True) if p.kind != "module"]
    placed_boxes: list[tuple[float, float, float, float]] = []
    result: dict[str, dict[str, Any]] = {}

    x_lo, x_hi = xlim[0] + 0.03, xlim[1] - 0.03
    y_lo, y_hi = ylim[0] + 0.03, ylim[1] - 0.03

    for p in ordered:
        px, py = pos_map[p.name]
        rx, ry = rad_map[p.name]
        tw = max(0.10, len(p.name) * cw + 0.018)
        th = ch

        gap_y = 0.020
        gap_x = 0.024

        outward_side = "left" if px < -0.08 else ("right" if px > 0.08 else "right")
        inward_side = "right" if outward_side == "left" else "left"

        raw_candidates: list[tuple[str, float, float, str, str, float]] = [
            ("below", px, py - ry - gap_y, "center", "top", 0.0),
            ("above", px, py + ry + gap_y, "center", "bottom", 0.14),
            ("right", px + rx + gap_x, py, "left", "center", 0.19 if outward_side == "right" else 0.28),
            ("left", px - rx - gap_x, py, "right", "center", 0.19 if outward_side == "left" else 0.28),
            ("above_right", px + rx * 0.72, py + ry + gap_y, "left", "bottom", 0.32),
            ("above_left", px - rx * 0.72, py + ry + gap_y, "right", "bottom", 0.32),
            ("below_right", px + rx * 0.72, py - ry - gap_y, "left", "top", 0.32),
            ("below_left", px - rx * 0.72, py - ry - gap_y, "right", "top", 0.32),
            ("right_far", px + rx + 0.14, py + 0.04, "left", "center", 0.52 if outward_side == "right" else 0.65),
            ("left_far", px - rx - 0.14, py + 0.04, "right", "center", 0.52 if outward_side == "left" else 0.65),
            ("above_far", px, py + ry + 0.11, "center", "bottom", 0.58),
            ("below_far", px, py - ry - 0.11, "center", "top", 0.58),
        ]

        # Domain-aware aesthetic biases:
        adjusted: list[tuple[str, float, float, str, str, float]] = []
        for tag, cx, cy, ha, va, pref in raw_candidates:
            cost = pref
            if p.kind == "camera" and py > 0.65 and tag == "above":
                cost = -0.08  # Keep retina title cleanly above the pixel bar, freeing the wire fan-out below!
            elif p.kind == "sensor" and py > 0.70 and tag == "below":
                cost = -0.06  # Keep left_eye / right_eye labels symmetrically below while antennae point up!
            elif p.kind == "sensor" and py > 0.70 and tag == "above":
                # Account for antennae height when placing above a top sensor (e.g. red_eye)
                cy = py + ry + 0.085
                cost = 0.08
            elif p.kind == "motor" and tag == "below":
                cost = -0.08
            elif inward_side == tag and abs(px) > 0.55:
                cost += 0.12
            adjusted.append((tag, cx, cy, ha, va, cost))

        best_choice = None
        best_score = math.inf

        for tag, cx, cy, ha, va, pref_cost in adjusted:
            if ha == "center":
                bx0, bx1 = cx - 0.5 * tw, cx + 0.5 * tw
            elif ha == "left":
                bx0, bx1 = cx, cx + tw
            else:
                bx0, bx1 = cx - tw, cx

            if va == "center":
                by0, by1 = cy - 0.5 * th, cy + 0.5 * th
            elif va == "bottom":
                by0, by1 = cy, cy + th
            else:
                by0, by1 = cy - th, cy

            bbox = (bx0, by0, bx1, by1)

            node_pen = sum(_rect_overlap_area(bbox, nb) for nb in node_boxes.values())
            loop_pen = sum(_rect_overlap_area(bbox, lb) for lb in loop_boxes)
            lbl_pen = sum(_rect_overlap_area(bbox, pb) for pb in placed_boxes)

            # Ambiguity penalty: a label's center must stay closer to its own part than to any neighbor!
            bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
            own_dist = math.hypot(bcx - px, bcy - py)
            ambig_pen = 0.0
            for q_name, (qx, qy) in pos_map.items():
                if q_name == p.name:
                    continue
                qrx, qry = rad_map[q_name]
                # Distance from label center to neighbor q's outer boundary
                q_edge_d = max(0.0, math.hypot(bcx - qx, bcy - qy) - 0.5 * (qrx + qry))
                own_edge_d = max(0.0, own_dist - 0.5 * (rx + ry))
                if q_edge_d < own_edge_d + 0.03:
                    ambig_pen += (own_edge_d + 0.03 - q_edge_d)

            oob_x = max(0.0, x_lo - bx0) + max(0.0, bx1 - x_hi)
            oob_y = max(0.0, y_lo - by0) + max(0.0, by1 - y_hi)
            oob_pen = oob_x + oob_y

            total = (
                2500.0 * node_pen
                + 2200.0 * lbl_pen
                + 900.0 * loop_pen
                + 12.0 * ambig_pen
                + 45.0 * oob_pen
                + pref_cost
            )
            if total < best_score:
                best_score = total
                best_choice = (tag, cx, cy, ha, va, bbox)

        assert best_choice is not None
        tag, cx, cy, ha, va, bbox = best_choice
        # Clamp gently inside xlim/ylim if a wide outer label slightly grazes the margin
        shift_x = max(0.0, x_lo - bbox[0]) - max(0.0, bbox[2] - x_hi)
        shift_y = max(0.0, y_lo - bbox[1]) - max(0.0, bbox[3] - y_hi)
        if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
            trial_bbox = (bbox[0] + shift_x, bbox[1] + shift_y, bbox[2] + shift_x, bbox[3] + shift_y)
            if sum(_rect_overlap_area(trial_bbox, nb) for nb in node_boxes.values()) <= 1e-6:
                cx += shift_x
                cy += shift_y
                bbox = trial_bbox

        placed_boxes.append(bbox)
        result[p.name] = {
            "tag": tag,
            "x": cx,
            "y": cy,
            "ha": ha,
            "va": va,
            "bbox": bbox,
            "leader": tag not in ("below", "above") or abs(cx - px) > 0.08,
            "anchor": (px, py),
            "radii": (rx, ry),
        }

    return result


# ---------------------------------------------------------------------------
# Fast Reusable 3-Tier Nervous System Artist (Posters & 60 FPS Video)
# ---------------------------------------------------------------------------


def _wire_base_color(w: Wire, active_hero: bool, theme: Theme) -> str:
    if w.control:
        return theme.violet
    if w.learning:
        if w.learning == "ergotrix":
            return "#2dd4bf" if theme.is_dark else "#0d9488"
        return theme.emerald if active_hero else ("#1e5a41" if theme.is_dark else "#86efac")
    if not w.active:
        return "#475569" if theme.is_dark else "#9ca3af"
    if active_hero:
        return theme.gold if theme.is_dark else (theme.cyan if w.sign > 0 else theme.coral)
    return ("#334872" if w.sign > 0 else "#5c2b41") if theme.is_dark else ("#93c5fd" if w.sign > 0 else "#fca5a5")


class ShannonNeuralPanelArtist:
    """Three-Tier Hero + Shannon Ensemble Connectome artist with zero-clear frame updates."""

    def __init__(
        self,
        ax_schem: mpl.axes.Axes,
        ax_bar: mpl.axes.Axes,
        ax_ras: mpl.axes.Axes,
        initial_snapshot: dict[str, Any],
        theme: Theme,
        *,
        neural_view: str = "composite",
        aspect: str = "16:9",
        max_codewords: int = 6,
    ) -> None:
        self.ax_schem = ax_schem
        self.ax_bar = ax_bar
        self.ax_ras = ax_ras
        self.theme = theme
        self.neural_view = neural_view.strip().lower()
        self.aspect = aspect
        self.max_codewords = max_codewords

        fig = ax_schem.figure
        fig_w = fig.get_figwidth() if fig is not None else 12.8
        self.scale = min(1.05, max(0.74, fig_w / 11.8))
        if aspect == "9:16":
            self.scale = 0.86

        diagram: Diagram = initial_snapshot["hero_diagram"]
        self.bw, self.bl = diagram.body
        self.parts_order = [p.name for p in diagram.parts]
        self.crowded = len(diagram.parts) >= 12 or any(p.shape is not None for p in diagram.parts)
        self.very_crowded = len(diagram.parts) >= 15

        self._setup_tier_a(diagram, initial_snapshot)
        self._setup_tier_b(diagram, initial_snapshot)
        self._setup_tier_c(diagram, initial_snapshot)
        self.update(initial_snapshot)

    # --- Tier A: Braitenberg Body Schematic + Joint Codebook Arcs + MI Wires ---

    def _setup_tier_a(self, diagram: Diagram, snapshot: dict[str, Any]) -> None:
        ax = self.ax_schem
        theme = self.theme
        ax.set_facecolor(theme.ax_bg)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(theme.spine)
            sp.set_linewidth(0.9 if theme.is_dark else 1.05)
        ax.set_xticks([])
        ax.set_yticks([])

        ylim = (-1.18, 1.22)
        fig = ax.figure
        if fig is not None:
            bbox = ax.get_position()
            box_ratio = (bbox.width * fig.get_figwidth()) / max(bbox.height * fig.get_figheight(), 1e-6)
            half_x = max(1.28, 0.5 * (ylim[1] - ylim[0]) * box_ratio)
        else:
            half_x = 1.32
        xlim = (-half_x, half_x)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        hero_col = hero_ring_color(theme)
        view_badge = {
            "composite": "Shannon Codebook & MI",
            "codebook": "Joint Codebook Arcs",
            "entropy": "Entropy & Transfer MI",
            "raster": "Microstate Bit-Raster",
        }.get(self.neural_view, "Shannon Connectome")

        ax.set_title(
            f"C  {diagram.title} (Cyan) · {view_badge}",
            loc="left",
            color=theme.fg,
            fontsize=9.4 * self.scale,
            fontweight="bold",
            pad=5.5,
        )

        # Rounded top-down vehicle body outline (+y forward = up)
        pad_b = 0.04
        body_fill = "#0d1536" if theme.is_dark else "#f5f2e8"
        body_rim = "#94a3b8" if theme.is_dark else "#3f3f46"
        if theme.glow_enabled:
            ax.add_patch(
                FancyBboxPatch(
                    (-self.bw + pad_b, -self.bl + pad_b),
                    2.0 * (self.bw - pad_b),
                    2.0 * (self.bl - pad_b),
                    boxstyle=f"round,pad={pad_b},rounding_size={0.28 * min(self.bw, self.bl):.3f}",
                    facecolor="none",
                    edgecolor=hero_col,
                    linewidth=3.4,
                    alpha=0.14,
                    zorder=1.2,
                )
            )
        ax.add_patch(
            FancyBboxPatch(
                (-self.bw + pad_b, -self.bl + pad_b),
                2.0 * (self.bw - pad_b),
                2.0 * (self.bl - pad_b),
                boxstyle=f"round,pad={pad_b},rounding_size={0.28 * min(self.bw, self.bl):.3f}",
                facecolor=body_fill,
                edgecolor=body_rim,
                linewidth=1.55,
                zorder=1.5,
            )
        )
        # Subtle anterior heading chevron at front nose (0, +bl)
        ny = self.bl + 0.015
        ax.plot([-0.06, 0.0, 0.06], [ny - 0.035, ny + 0.015, ny - 0.035], color=body_rim, lw=1.1, alpha=0.65, zorder=2)

        parts_by_name = {p.name: p for p in diagram.parts}
        self.part_positions: dict[str, tuple[float, float]] = {
            p.name: (p.position[0] * self.bw, p.position[1] * self.bl) for p in diagram.parts
        }
        self.part_radii: dict[str, tuple[float, float]] = {
            p.name: _part_radii(p, self.bw, self.bl, self.crowded, self.very_crowded) for p in diagram.parts
        }

        # Module boxes underneath wires
        for p in diagram.parts:
            if p.kind == "module":
                px, py = self.part_positions[p.name]
                rx, ry = self.part_radii[p.name]
                ax.add_patch(
                    FancyBboxPatch(
                        (px - rx, py - ry),
                        2 * rx,
                        2 * ry,
                        boxstyle="round,pad=0.01,rounding_size=0.03",
                        facecolor=theme.panel_bg,
                        edgecolor=body_rim,
                        linewidth=1.2,
                        zorder=2.0,
                    )
                )
                ax.text(
                    px,
                    py,
                    p.name,
                    ha="center",
                    va="center",
                    fontsize=7.2 * self.scale,
                    fontweight="bold",
                    color=theme.fg,
                    zorder=9.0,
                )

        # Precompute bundled wire geometry once
        self.bundled_wires = bundles(diagram.wires)
        pairs_count: dict[frozenset[str], int] = {}
        directions: dict[frozenset[str], set[tuple[str, str]]] = {}
        for w in self.bundled_wires:
            pair = frozenset((w.source, w.target))
            pairs_count[pair] = pairs_count.get(pair, 0) + 1
            directions.setdefault(pair, set()).add((w.source, w.target))

        drawn_pair: dict[frozenset[str], int] = {}
        self.wire_polylines: list[np.ndarray] = []
        self.wire_sign_texts: list[tuple[int, mpl.text.Text]] = []

        for idx, w in enumerate(self.bundled_wires):
            sx, sy = self.part_positions[w.source]
            tx, ty = self.part_positions[w.target]
            srx, sry = self.part_radii[w.source]
            trx, try_ = self.part_radii[w.target]

            if w.source == w.target:
                side = -1.0 if sx < 0.0 else 1.0
                t_curve = np.linspace(0.0, 1.0, 22)
                p0 = np.array([sx + side * srx * 0.72, sy + sry * 0.62])
                p1 = np.array([sx + side * (srx + 0.14), sy + sry + 0.16])
                p2 = np.array([sx + side * (srx + 0.16), sy - sry * 0.15])
                p3 = np.array([sx + side * srx * 0.85, sy - sry * 0.05])
                omt = 1.0 - t_curve
                pts = (
                    (omt**3)[:, None] * p0
                    + (3 * omt**2 * t_curve)[:, None] * p1
                    + (3 * omt * t_curve**2)[:, None] * p2
                    + (t_curve**3)[:, None] * p3
                )
                self.wire_polylines.append(pts)
                lx, ly = sx + side * (srx + 0.125), sy + sry * 0.78
            else:
                dx, dy = tx - sx, ty - sy
                dist = math.hypot(dx, dy) or 1.0
                ux, uy = dx / dist, dy / dist
                pair = frozenset((w.source, w.target))
                k_idx = drawn_pair[pair] = drawn_pair.get(pair, -1) + 1
                shift = (k_idx - (pairs_count[pair] - 1) / 2.0) * 0.036
                if len(directions[pair]) > 1 and w.source > w.target:
                    shift = -shift
                nx_p, ny_p = -uy, ux
                x1, y1 = sx + nx_p * shift, sy + ny_p * shift
                x2, y2 = tx + nx_p * shift, ty + ny_p * shift

                start_off = min(srx / max(abs(ux), 1e-3), sry / max(abs(uy), 1e-3))
                end_off = min(trx / max(abs(ux), 1e-3), try_ / max(abs(uy), 1e-3))
                start_off = min(start_off, 0.35 * dist)
                end_off = min(end_off, 0.35 * dist)
                p_start = np.array([x1 + ux * start_off, y1 + uy * start_off])
                p_end = np.array([x2 - ux * end_off, y2 - uy * end_off])
                pts = np.vstack([p_start, p_end])
                self.wire_polylines.append(pts)
                lx = p_start[0] + 0.80 * (p_end[0] - p_start[0]) + nx_p * 0.035
                ly = p_start[1] + 0.80 * (p_end[1] - p_start[1]) + ny_p * 0.035

            if not w.control and (not w.label or w.learning):
                sign_str = ("+" if w.sign > 0 else "−") + LEARNING_LABELS.get(w.learning, "")
                stxt = ax.text(
                    lx,
                    ly,
                    sign_str,
                    ha="center",
                    va="center",
                    fontsize=(7.4 if self.crowded else 8.2) * self.scale,
                    fontweight="bold",
                    color=_wire_base_color(w, w.active, theme),
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=body_fill, alpha=0.92)],
                    zorder=4.8,
                )
                self.wire_sign_texts.append((idx, stxt))

        # Wire LineCollections: MI/TE outer glow + crisp core + dashed overlay for inactive/control wires
        self.wire_mi_lc = LineCollection(self.wire_polylines, capstyle="round", zorder=2.6)
        self.wire_solid_lc = LineCollection([], capstyle="round", zorder=3.2)
        self.wire_dash_lc = LineCollection([], linestyles=(0, (3.2, 2.6)), capstyle="round", zorder=3.1)
        ax.add_collection(self.wire_mi_lc)
        ax.add_collection(self.wire_solid_lc)
        ax.add_collection(self.wire_dash_lc)

        # Parts: Entropy Halo + Quiescent Ring + Codebook Wedge Pool + Hero Core + Threshold Text
        self.entropy_halos: dict[str, mpl.patches.Patch] = {}
        self.wedge_pools: dict[str, list[Wedge]] = {}
        self.array_bar_pools: dict[str, tuple[float, float, float, list[Rectangle]]] = {}
        self.core_patches: dict[str, mpl.patches.Patch] = {}
        self.thresh_texts: dict[str, mpl.text.Text] = {}
        self.grid_images: dict[str, mpl.image.AxesImage] = {}
        self.grid_borders: dict[str, Rectangle] = {}

        base_r = 0.058 if self.very_crowded else (0.066 if self.crowded else 0.076)
        ring_w = (
            (0.029 if self.neural_view == "codebook" else 0.023)
            if self.very_crowded
            else ((0.034 if self.neural_view == "codebook" else 0.027) if self.crowded else (0.040 if self.neural_view == "codebook" else 0.031))
        )
        r_ring_out = base_r + 0.005 + ring_w
        quiescent_col = "#162044" if theme.is_dark else "#e5e0d3"
        outline_col = "#cbd5e1" if theme.is_dark else "#27272a"

        for p in diagram.parts:
            if p.kind == "module":
                continue
            px, py = self.part_positions[p.name]
            rx, ry = self.part_radii[p.name]

            if p.shape is not None:
                # Camera (retina 4x24) or Array (odd_left / odd_right 4x4)
                rows, cols = p.shape if len(p.shape) == 2 else (1, p.shape[0])
                w_box = 1.12 * self.bw if p.kind == "camera" else 0.25 * self.bw
                cell = min(0.064, w_box / max(cols, 1))
                gw, gh = cols * cell, rows * cell
                ext = [px - 0.5 * gw, px + 0.5 * gw, py - 0.5 * gh, py + 0.5 * gh]

                # Outer entropy/codebook aura rectangle for arrays/cameras
                aura = Rectangle(
                    (ext[0] - 0.016, ext[2] - 0.016),
                    gw + 0.032,
                    gh + 0.032,
                    facecolor="none",
                    edgecolor=theme.cyan,
                    linewidth=2.2,
                    alpha=0.0,
                    zorder=5.2,
                )
                ax.add_patch(aura)
                self.entropy_halos[p.name] = aura

                init_rgba = np.zeros((rows, cols, 4), dtype=float)
                init_rgba[..., 3] = 1.0
                im = ax.imshow(
                    init_rgba,
                    extent=ext,
                    origin="upper",
                    interpolation="nearest",
                    zorder=6.0,
                )
                self.grid_images[p.name] = im

                # Crisp internal pixel/cell hairlines
                grid_segs = []
                for r_i in range(1, rows):
                    yy = ext[3] - r_i * cell
                    grid_segs.append([(ext[0], yy), (ext[1], yy)])
                for c_i in range(1, cols):
                    xx = ext[0] + c_i * cell
                    grid_segs.append([(xx, ext[2]), (xx, ext[3])])
                if grid_segs:
                    glc = LineCollection(
                        grid_segs,
                        colors="#263459" if theme.is_dark else "#a1a1aa",
                        linewidths=0.42 if cols > 10 else 0.65,
                        alpha=0.75,
                        zorder=6.4,
                    )
                    ax.add_collection(glc)

                bdr = Rectangle(
                    (ext[0], ext[2]),
                    gw,
                    gh,
                    facecolor="none",
                    edgecolor=outline_col,
                    linewidth=1.15,
                    zorder=6.6,
                )
                ax.add_patch(bdr)
                self.grid_borders[p.name] = bdr

                # Crisp horizontal segmented Codebook Bar along top edge of array grid
                if p.kind == "array":
                    bar_y = ext[3] + 0.008
                    bar_h = 0.022
                    ax.add_patch(
                        Rectangle(
                            (ext[0], bar_y),
                            gw,
                            bar_h,
                            facecolor=quiescent_col,
                            edgecolor=theme.spine,
                            linewidth=0.45,
                            zorder=5.5,
                        )
                    )
                    rpool: list[Rectangle] = []
                    for _ in range(self.max_codewords + 1):
                        rc = Rectangle(
                            (ext[0], bar_y),
                            0.0,
                            bar_h,
                            facecolor=theme.cyan,
                            edgecolor="none",
                            visible=False,
                            zorder=5.8,
                        )
                        ax.add_patch(rc)
                        rpool.append(rc)
                    self.array_bar_pools[p.name] = (ext[0], bar_y, gw, rpool)
                continue

            # Circular/diamond/motor parts:
            halo_r = (0.145 if p.kind == "motor" else r_ring_out) + 0.022
            halo = Circle(
                (px, py),
                radius=halo_r,
                facecolor="none",
                edgecolor=theme.amber,
                linewidth=3.2,
                alpha=0.0,
                zorder=5.1,
            )
            ax.add_patch(halo)
            self.entropy_halos[p.name] = halo

            # Background quiescent track + Segmented Shannon Joint Codebook Wedge Ring
            r_w_out = (0.132 if p.kind == "motor" else r_ring_out)
            r_w_width = 0.022 if p.kind == "motor" else ring_w
            ax.add_patch(
                Wedge(
                    (px, py),
                    r_w_out,
                    0.0,
                    360.0,
                    width=r_w_width,
                    facecolor=quiescent_col,
                    edgecolor=theme.spine,
                    linewidth=0.45,
                    zorder=5.4,
                )
            )
            pool = []
            for _ in range(self.max_codewords + 1):
                wg = Wedge(
                    (px, py),
                    r_w_out,
                    90.0,
                    90.0,
                    width=r_w_width,
                    facecolor=theme.cyan,
                    edgecolor=body_fill,
                    linewidth=0.35,
                    visible=False,
                    zorder=5.8,
                )
                ax.add_patch(wg)
                pool.append(wg)
            self.wedge_pools[p.name] = pool

            # Sensor antennae (+y forward = up!)
            if p.kind == "sensor":
                for dx_a in (-0.042, 0.042):
                    ax.plot(
                        [px + dx_a * 0.45, px + dx_a],
                        [py + base_r * 0.85, py + r_w_out + 0.052],
                        color=outline_col,
                        lw=1.35,
                        solid_capstyle="round",
                        zorder=5.9,
                    )

            # Inner Hero #0 core shape
            if p.kind == "motor":
                mw, mh = 0.19, 0.115
                core = FancyBboxPatch(
                    (px - 0.5 * mw, py - 0.5 * mh),
                    mw,
                    mh,
                    boxstyle="round,pad=0.0,rounding_size=0.025",
                    facecolor=activity_fill_color(0.0, theme),
                    edgecolor=outline_col,
                    linewidth=1.25,
                    zorder=6.5,
                )
                ax.add_patch(core)
                self.core_patches[p.name] = core
            elif p.kind == "control":
                dr = base_r * 1.05
                verts = [(px, py + dr), (px + dr, py), (px, py - dr), (px - dr, py)]
                core = Polygon(
                    verts,
                    closed=True,
                    facecolor=activity_fill_color(0.0, theme),
                    edgecolor=theme.violet,
                    linewidth=1.4,
                    zorder=6.5,
                )
                ax.add_patch(core)
                self.core_patches[p.name] = core
                self.thresh_texts[p.name] = ax.text(
                    px,
                    py,
                    "θ",
                    ha="center",
                    va="center",
                    fontsize=7.6 * self.scale,
                    fontweight="bold",
                    color=theme.violet,
                    zorder=7.2,
                )
            else:
                core = Circle(
                    (px, py),
                    radius=base_r,
                    facecolor=activity_fill_color(0.0, theme),
                    edgecolor=outline_col,
                    linewidth=1.2,
                    zorder=6.5,
                )
                ax.add_patch(core)
                self.core_patches[p.name] = core
                if p.threshold is not None:
                    t_str = f"{p.effective:.1f}" if p.effective is not None else f"{p.threshold:g}"
                    self.thresh_texts[p.name] = ax.text(
                        px,
                        py,
                        t_str,
                        ha="center",
                        va="center",
                        fontsize=(5.5 if self.very_crowded else (6.2 if len(t_str) >= 4 else 6.8)) * self.scale,
                        fontweight="bold",
                        color=theme.fg,
                        zorder=7.2,
                    )

        # Smart non-overlapping part labels using exact physical-to-data font metrics
        if fig is not None:
            bbox_a = ax.get_position()
            ax_w_in = max(2.0, bbox_a.width * fig.get_figwidth())
            ax_h_in = max(1.5, bbox_a.height * fig.get_figheight())
        else:
            ax_w_in, ax_h_in = 4.2, 3.4

        label_fs = min(7.2, max(5.2, ax_h_in * (1.88 if self.very_crowded else (2.05 if self.crowded else 2.35))))
        cw_data = (label_fs * 0.64 / 72.0) * ((xlim[1] - xlim[0]) / ax_w_in)
        ch_data = (label_fs * 1.32 / 72.0) * ((ylim[1] - ylim[0]) / ax_h_in)

        self.label_placements = solve_part_labels(
            diagram.parts,
            self.bundled_wires,
            diagram.body,
            xlim=xlim,
            ylim=ylim,
            char_w=cw_data,
            char_h=ch_data,
        )
        label_halo = [pe.withStroke(linewidth=2.6, foreground=body_fill, alpha=0.95)]
        lbl_col = "#e2e8f0" if theme.is_dark else "#27272a"

        for p_name, spec in self.label_placements.items():
            if spec["leader"]:
                ax_x, ax_y = spec["anchor"]
                rx, ry = spec["radii"]
                dx_l = spec["x"] - ax_x
                dy_l = spec["y"] - ax_y
                ln = math.hypot(dx_l, dy_l) or 1.0
                sx_l = ax_x + (dx_l / ln) * rx * 0.95
                sy_l = ax_y + (dy_l / ln) * ry * 0.95
                ax.plot(
                    [sx_l, spec["x"]],
                    [sy_l, spec["y"]],
                    color=theme.fg_dim,
                    lw=0.65,
                    alpha=0.58,
                    zorder=7.8,
                )
            ax.text(
                spec["x"],
                spec["y"],
                p_name,
                ha=spec["ha"],
                va=spec["va"],
                fontsize=label_fs,
                fontweight="bold",
                color=lbl_col,
                path_effects=label_halo,
                zorder=8.5,
            )

    # --- Tier B: Shannon Telemetry Pill Bar + 1D Empirical Density Barcodes ---

    def _setup_tier_b(self, diagram: Diagram, snapshot: dict[str, Any]) -> None:
        ax = self.ax_bar
        theme = self.theme
        ax.set_facecolor(theme.ax_bg)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(theme.spine)
            sp.set_linewidth(0.85 if theme.is_dark else 1.0)
        ax.set_xticks([])
        ax.set_yticks([])

        barcodes = snapshot.get("barcodes") or {}
        max_rows = 6 if self.aspect != "1:1" else 7
        self.barcode_keys = _select_featured_barcodes(diagram, barcodes, max_rows=max_rows)
        n_rows = max(1, len(self.barcode_keys))

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, float(n_rows) + 1.25)

        # Live Shannon Telemetry Header Strip at top of Tier B
        tel_bg = "#131f4c" if theme.is_dark else "#ebe5d5"
        ax.add_patch(
            Rectangle(
                (0.0, float(n_rows) + 0.12),
                1.0,
                1.13,
                facecolor=tel_bg,
                edgecolor=theme.spine,
                linewidth=0.75,
                zorder=2,
            )
        )
        fig_b = ax.figure
        bar_w_in = (ax.get_position().width * fig_b.get_figwidth()) if fig_b is not None else 4.2
        tel_fs = min(7.4, max(5.6, bar_w_in * 1.62))
        self.telemetry_text = ax.text(
            0.50,
            float(n_rows) + 0.68,
            "",
            ha="center",
            va="center",
            fontsize=tel_fs,
            fontweight="bold",
            color=theme.cyan if theme.is_dark else theme.fg,
            zorder=5,
        )

        self.barcode_ims: list[mpl.image.AxesImage] = []
        self.barcode_cursors: list[Line2D] = []
        self.barcode_val_texts: list[mpl.text.Text] = []
        self.cmap_sig = barcode_cmap(theme, is_learning=False)
        self.cmap_lrn = barcode_cmap(theme, is_learning=True)

        x_bar0, x_bar1 = 0.36, 0.84
        hero_col = hero_ring_color(theme)

        for i, key in enumerate(self.barcode_keys):
            entry = barcodes.get(key, {})
            is_learn = bool(entry.get("is_learning", False)) or ("->" in key)
            gate = entry.get("gate", ACTIVE_STRENGTH if is_learn else None)
            row_y = float(n_rows - 1 - i) + 0.50

            short_lbl = key.replace("sees_", "").replace("_motor", "_mot")
            if len(short_lbl) > 15:
                short_lbl = short_lbl[:14] + "…"
            lbl_str = f"{short_lbl} [M]" if is_learn else short_lbl

            ax.text(
                0.022,
                row_y,
                lbl_str,
                ha="left",
                va="center",
                fontsize=6.9 * self.scale,
                fontweight="bold" if is_learn else "normal",
                color=theme.emerald if is_learn else theme.fg_muted,
                zorder=4,
            )

            init_strip = np.zeros((1, 20, 4), dtype=float)
            init_strip[..., 3] = 1.0
            im = ax.imshow(
                init_strip,
                extent=[x_bar0, x_bar1, row_y - 0.28, row_y + 0.28],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                zorder=3,
            )
            ax.add_patch(
                Rectangle(
                    (x_bar0, row_y - 0.28),
                    x_bar1 - x_bar0,
                    0.56,
                    facecolor="none",
                    edgecolor=theme.spine,
                    linewidth=0.65,
                    zorder=4,
                )
            )
            self.barcode_ims.append(im)

            if gate is not None:
                gx = x_bar0 + float(gate) * (x_bar1 - x_bar0)
                ax.plot(
                    [gx, gx],
                    [row_y - 0.36, row_y + 0.36],
                    color=theme.amber if theme.is_dark else theme.coral,
                    lw=1.35,
                    ls="--",
                    zorder=5.5,
                )

            cur = ax.plot(
                [x_bar0, x_bar0],
                [row_y - 0.35, row_y + 0.35],
                color=hero_col,
                lw=1.9,
                solid_capstyle="round",
                zorder=6.5,
            )[0]
            self.barcode_cursors.append(cur)

            vtxt = ax.text(
                0.982,
                row_y,
                "",
                ha="right",
                va="center",
                fontsize=6.8 * self.scale,
                fontweight="bold",
                color=theme.fg,
                zorder=5,
            )
            self.barcode_val_texts.append(vtxt)

        self._bar_bounds = (x_bar0, x_bar1)

    # --- Tier C: Lossless N x M Sorted Swarm Microstate Bit-Raster ---

    def _setup_tier_c(self, diagram: Diagram, snapshot: dict[str, Any]) -> None:
        ax = self.ax_ras
        theme = self.theme
        ax.set_facecolor(theme.ax_bg)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(theme.spine)
            sp.set_linewidth(0.85 if theme.is_dark else 1.0)

        rmat = np.asarray(snapshot.get("raster_matrix", np.zeros((1, 1))), dtype=float)
        if rmat.ndim != 2 or rmat.size == 0:
            rmat = np.zeros((1, 1), dtype=float)
        n_veh, m_units = rmat.shape
        unit_names: list[str] = list(snapshot.get("raster_unit_names") or [f"u{k}" for k in range(m_units)])

        self.ras_n = n_veh
        self.ras_m = m_units

        ax.set_xlim(0.0, float(m_units))
        ax.set_ylim(float(n_veh), 0.0)
        ax.set_yticks([])

        # Abbreviated non-colliding column ticks
        if m_units <= 20:
            ax.set_xticks(np.arange(m_units) + 0.5)
            short_cols = [
                u.replace("idea_", "I")
                .replace("left_", "L_")
                .replace("right_", "R_")
                .replace("_brighter", "_br")
                .replace("_heavier", "_hv")
                .replace("front_ahead", "front")
                .replace("symmetric", "sym")
                .replace("turn_away", "t_awy")
                .replace("turn_back", "t_bck")
                .replace("run_over", "r_ovr")[:6]
                for u in unit_names
            ]
            ax.set_xticklabels(
                short_cols,
                rotation=32 if m_units > 6 else 0,
                ha="right" if m_units > 6 else "center",
                fontsize=(5.9 if m_units > 11 else 6.5) * self.scale,
                color=theme.fg_dim,
            )
            ax.tick_params(axis="x", length=2.0, pad=1.5)
        else:
            ax.set_xticks([])

        self.ras_title = ax.set_title(
            f"Sorted Swarm Microstate Raster ({n_veh}×{m_units} bits · Zero Loss)",
            loc="left",
            color=theme.fg_muted,
            fontsize=7.8 * self.scale,
            fontweight="bold",
            pad=3.2,
        )

        init_rgba = np.zeros((n_veh, m_units, 4), dtype=float)
        init_rgba[..., 3] = 1.0
        self.ras_im = ax.imshow(
            init_rgba,
            extent=[0.0, float(m_units), float(n_veh), 0.0],
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            zorder=2,
        )

        # Subtle vertical column dividers
        col_segs = [[(float(c), 0.0), (float(c), float(n_veh))] for c in range(1, m_units)]
        if col_segs:
            ax.add_collection(
                LineCollection(
                    col_segs,
                    colors=theme.grid,
                    linewidths=0.45,
                    alpha=0.55,
                    zorder=3,
                )
            )

        hero_col = hero_ring_color(theme)
        self.hero_row_rect = Rectangle(
            (0.0, 0.0),
            float(m_units),
            1.0,
            facecolor="none",
            edgecolor=hero_col,
            linewidth=1.5,
            zorder=5,
        )
        ax.add_patch(self.hero_row_rect)
        self.hero_row_tag = ax.text(
            -0.015,
            0.5,
            "▶#0",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.3 * self.scale,
            fontweight="bold",
            color=hero_col,
            zorder=6,
        )

    # --- Fast Per-Frame / Static Update ---

    def update(self, snapshot: dict[str, Any]) -> None:
        """Mutate all 3 neural tiers in-place from ``snapshot``."""
        theme = self.theme
        diagram: Diagram = snapshot["hero_diagram"]
        ent_cmap = entropy_cmap(theme)

        # 1. Update Wires (MI/TE outer glow + Hero #0 active core)
        mi_colors = []
        mi_widths = []
        solid_segs, solid_cols, solid_lws = [], [], []
        dash_segs, dash_cols, dash_lws = [], [], []

        live_bundles = bundles(diagram.wires)
        wire_lookup = {(w.source, w.target, w.sign, w.learning, w.control): w for w in live_bundles}

        for idx, base_w in enumerate(self.bundled_wires):
            w = wire_lookup.get(
                (base_w.source, base_w.target, base_w.sign, base_w.learning, base_w.control),
                base_w,
            )
            pts = self.wire_polylines[idx]
            info_bits = _lookup_wire_bits(snapshot, w, idx)
            info_norm = min(1.0, info_bits / 0.85)

            if self.neural_view == "codebook":
                halo_a = 0.14 * info_norm
            elif self.neural_view == "entropy":
                halo_a = 0.55 * info_norm
            else:
                halo_a = 0.36 * info_norm

            mi_hue = theme.emerald if w.learning else (theme.cyan if w.sign > 0 else theme.coral)
            mi_rgba = list(to_rgba(mi_hue))
            mi_rgba[3] = halo_a if info_bits > 0.015 else 0.0
            mi_colors.append(tuple(mi_rgba))
            mi_widths.append(2.0 + 5.2 * info_norm)

            hero_firing = bool(w.active and float(w.level) > 0.05 and not w.control)
            core_col = _wire_base_color(w, hero_firing, theme)
            core_rgba = list(to_rgba(core_col))
            core_rgba[3] = 0.96 if hero_firing else (0.52 if w.active else 0.38)
            lw = (1.1 + 2.2 * float(w.strength)) if w.learning else (2.0 if hero_firing else 0.95)

            if w.control or not w.active:
                dash_segs.append(pts)
                dash_cols.append(tuple(core_rgba))
                dash_lws.append(max(0.85, lw * 0.85))
            else:
                solid_segs.append(pts)
                solid_cols.append(tuple(core_rgba))
                solid_lws.append(lw)

        self.wire_mi_lc.set_color(mi_colors)
        self.wire_mi_lc.set_linewidths(mi_widths)
        self.wire_solid_lc.set_segments(solid_segs)
        self.wire_solid_lc.set_color(solid_cols)
        self.wire_solid_lc.set_linewidths(solid_lws)
        self.wire_dash_lc.set_segments(dash_segs)
        self.wire_dash_lc.set_color(dash_cols)
        self.wire_dash_lc.set_linewidths(dash_lws)

        for idx, stxt in self.wire_sign_texts:
            base_w = self.bundled_wires[idx]
            w = wire_lookup.get(
                (base_w.source, base_w.target, base_w.sign, base_w.learning, base_w.control),
                base_w,
            )
            hero_firing = bool(w.active and float(w.level) > 0.05)
            stxt.set_color(_wire_base_color(w, hero_firing, theme))

        # 2. Update Parts (Camera/Array Cell Grids, Hero Core, Codebook Arcs, Entropy Halo)
        for p_idx, p in enumerate(diagram.parts):
            if p.kind == "module":
                continue

            h_bits = _extract_part_entropy(snapshot, p.name)
            halo_patch = self.entropy_halos.get(p.name)
            if halo_patch is not None:
                if h_bits > 0.02 and self.neural_view != "codebook":
                    ecol = list(ent_cmap(h_bits))
                    ecol[3] = (0.72 if self.neural_view == "entropy" else 0.44) * h_bits
                    halo_patch.set_edgecolor(tuple(ecol))
                else:
                    halo_patch.set_edgecolor((0.0, 0.0, 0.0, 0.0))

            if p.shape is not None and p.name in self.grid_images:
                rows, cols = p.shape if len(p.shape) == 2 else (1, p.shape[0])
                hero_lvls = np.asarray(p.level, dtype=float).reshape(rows, cols)
                prob_g, ent_g = _extract_cell_grids(snapshot, p.name, (rows, cols))
                hero_rgba = activity_fill_rgba(hero_lvls, theme)

                if self.neural_view == "entropy" and ent_g is not None:
                    swarm_rgba = ent_cmap(np.clip(ent_g, 0.0, 1.0))
                    blend = np.where(hero_lvls[..., None] > 0.25, 0.65 * hero_rgba + 0.35 * swarm_rgba, swarm_rgba)
                    blend[..., 3] = 1.0
                    self.grid_images[p.name].set_data(np.clip(blend, 0.0, 1.0))
                elif prob_g is not None:
                    swarm_tint = to_rgba(theme.cyan)
                    bg_tint = activity_fill_rgba(np.zeros_like(hero_lvls), theme)
                    ens_rgba = bg_tint * (1.0 - 0.62 * prob_g[..., None]) + np.array(swarm_tint) * (
                        0.62 * prob_g[..., None]
                    )
                    out_rgba = np.where(hero_lvls[..., None] > 0.08, hero_rgba, ens_rgba)
                    out_rgba[..., 3] = 1.0
                    self.grid_images[p.name].set_data(np.clip(out_rgba, 0.0, 1.0))
                else:
                    self.grid_images[p.name].set_data(hero_rgba)

            if p.name in self.core_patches:
                lvl = float(p.level) if not isinstance(p.level, list) else 0.0
                self.core_patches[p.name].set_facecolor(activity_fill_color(lvl, theme))
                if p.name in self.thresh_texts and p.kind != "control":
                    t_str = f"{p.effective:.1f}" if p.effective is not None else f"{p.threshold:g}"
                    txt = self.thresh_texts[p.name]
                    txt.set_text(t_str)
                    if theme.is_dark:
                        txt.set_color("#0f172a" if lvl > 0.52 else "#f8fafc")
                    else:
                        txt.set_color("#ffffff" if lvl > 0.68 else "#18181b")

            # Update Joint Codebook Arc-Ring Wedges (or segmented bar on rectangular arrays)
            if p.name in self.array_bar_pools:
                x0_b, y0_b, gw_b, rpool = self.array_bar_pools[p.name]
                arcs = _extract_part_arcs(snapshot, p.name, p_idx)
                show_arcs = self.neural_view in ("composite", "codebook", "raster")
                cur_x = x0_b
                used = 0
                if show_arcs:
                    for cid, pm in arcs[: len(rpool)]:
                        seg_w = max(0.0, min(gw_b - (cur_x - x0_b), gw_b * pm))
                        if seg_w < 1e-3:
                            continue
                        rc = rpool[used]
                        rc.set_x(cur_x)
                        rc.set_y(y0_b)
                        rc.set_width(seg_w)
                        rc.set_facecolor(codeword_color(theme, cid, self.max_codewords))
                        rc.set_visible(True)
                        cur_x += seg_w
                        used += 1
                for rem_rc in rpool[used:]:
                    rem_rc.set_visible(False)

            if p.name in self.wedge_pools:
                pool = self.wedge_pools[p.name]
                arcs = _extract_part_arcs(snapshot, p.name, p_idx)
                show_arcs = self.neural_view in ("composite", "codebook", "raster")
                cursor_deg = 90.0
                used = 0
                if show_arcs:
                    for cid, pm in arcs[: len(pool)]:
                        span = max(0.0, min(360.0, 360.0 * pm))
                        if span < 1.2:
                            continue
                        wg = pool[used]
                        t1 = cursor_deg - span
                        t2 = cursor_deg
                        wg.set_theta1(t1)
                        wg.set_theta2(t2)
                        wg.set_facecolor(codeword_color(theme, cid, self.max_codewords))
                        wg.set_visible(True)
                        cursor_deg -= span
                        used += 1
                for rem_wg in pool[used:]:
                    rem_wg.set_visible(False)

        # 3. Update Tier B (Shannon Telemetry + 1D Empirical Density Barcodes)
        h_joint = float(snapshot.get("joint_entropy_bits", 0.0))
        tc_bits = float(snapshot.get("total_correlation_bits", 0.0))
        sp_mi = float(snapshot.get("spatial_mi_bits", 0.0))
        self.telemetry_text.set_text(
            f"H(W) = {h_joint:.2f}b  ·  Synergy TC = {tc_bits:.2f}b  ·  I(Phase; W) = {sp_mi:.2f}b"
        )

        barcodes = snapshot.get("barcodes") or {}
        x_b0, x_b1 = self._bar_bounds
        for i, key in enumerate(self.barcode_keys):
            entry = barcodes.get(key)
            if not entry:
                continue
            hist = np.asarray(entry.get("hist", np.zeros(20)), dtype=float).ravel()
            if hist.size == 0:
                hist = np.zeros(20, dtype=float)
            peak = float(np.max(hist))
            norm_h = (hist / peak) ** 0.62 if peak > 1e-9 else np.zeros_like(hist)
            is_learn = bool(entry.get("is_learning", False)) or ("->" in key)
            cmap = self.cmap_lrn if is_learn else self.cmap_sig
            rgba_strip = cmap(norm_h)[None, :, :]
            self.barcode_ims[i].set_data(rgba_strip)

            lo = float(entry.get("min", 0.0))
            hi = float(entry.get("max", 1.0))
            h_val = float(entry.get("hero_val", entry.get("hero", 0.0)))
            frac = np.clip((h_val - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            cx = x_b0 + float(frac) * (x_b1 - x_b0)
            self.barcode_cursors[i].set_xdata([cx, cx])

            if self.neural_view == "entropy":
                eb = float(entry.get("entropy_bits", 0.0))
                self.barcode_val_texts[i].set_text(f"{eb:.2f}b")
            else:
                self.barcode_val_texts[i].set_text(f"{h_val:.2f}")

        # 4. Update Tier C (Sorted N x M Microstate Bit-Raster)
        rmat = np.asarray(snapshot.get("raster_matrix", np.zeros((self.ras_n, self.ras_m))), dtype=float)
        if rmat.ndim == 2 and rmat.shape == (self.ras_n, self.ras_m):
            cids = np.asarray(
                snapshot.get("raster_codeword_ids", np.zeros(self.ras_n, dtype=int)),
                dtype=int,
            )
            off_rgba = np.array(to_rgba("#0d1533" if theme.is_dark else "#f3efe4"), dtype=float)
            ras_rgba = np.broadcast_to(off_rgba, (self.ras_n, self.ras_m, 4)).copy()
            for r_i in range(self.ras_n):
                row_col = np.array(to_rgba(codeword_color(theme, int(cids[r_i]), self.max_codewords)), dtype=float)
                on_mask = rmat[r_i] > 0.5
                if np.any(on_mask):
                    ras_rgba[r_i, on_mask] = row_col
                # Faint row tint on off-cells so sorted codeword blocks are visible even when sparse
                off_mask = ~on_mask
                if np.any(off_mask):
                    ras_rgba[r_i, off_mask] = 0.88 * off_rgba + 0.12 * row_col
                    ras_rgba[r_i, off_mask, 3] = 1.0

            self.ras_im.set_data(ras_rgba)
            hero_row = int(np.clip(snapshot.get("hero_sorted_row", 0), 0, max(0, self.ras_n - 1)))
            self.hero_row_rect.set_y(float(hero_row))
            self.hero_row_tag.set_y(float(hero_row) + 0.5)


def render_shannon_neural_panel(
    fig: mpl.figure.Figure,
    slot: SubplotSpec,
    snapshot: dict[str, Any],
    theme: Theme,
    *,
    neural_view: str = "composite",
    aspect: str = "16:9",
    max_codewords: int = 6,
) -> ShannonNeuralPanelArtist:
    """Create the 3-tier axes inside ``slot`` and render ``snapshot``."""
    ax_schem, ax_bar, ax_ras = make_neural_tier_axes(
        fig, slot, neural_view=neural_view, aspect=aspect
    )
    return ShannonNeuralPanelArtist(
        ax_schem,
        ax_bar,
        ax_ras,
        snapshot,
        theme,
        neural_view=neural_view,
        aspect=aspect,
        max_codewords=max_codewords,
    )
