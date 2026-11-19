"""Unified 3-Series Command-Line Interface for Braitenberg Strogatz × Shannon Dynamics Studio.

Supports all three visual series:
1. ``phase-portraits``: Pure Axiom-1 4-Panel Strogatz Deep Phase Portraits (8 vehicles)
2. ``tripanels``: Pure Axiom-1 Shannon × Strogatz Ensemble Connectome Tri-Panels (6 swarms)
3. ``atlas``: Restored Round-1 Comparative Synthesis Phase-Space Atlas & Classic Animations

Examples:
    python -m braitenberg.dynamics poster --series phase-portraits --experiment v2a_fear --out-dir /tmp/out
    python -m braitenberg.dynamics poster --series tripanels --experiment v9_shy --out-dir /tmp/out
    python -m braitenberg.dynamics poster --series atlas --act atlas --theme textbook --out-dir /tmp/out
    python -m braitenberg.dynamics animate --series tripanels --experiment v5_moth --out-dir /tmp/out
    python -m braitenberg.dynamics animate --series atlas --scene ftle-swarm --out-dir /tmp/out
    python -m braitenberg.dynamics showcase --out-dir experiments/strogatz-dynamics/media --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from braitenberg.dynamics.diagram_mpl import NEURAL_VIEWS
from braitenberg.dynamics.phase_atlas import ATLAS_ACTS, ATLAS_SCENES, SCENE_STEMS
from braitenberg.dynamics.phase_portraits import PHASE_PORTRAIT_EXPERIMENTS
from braitenberg.dynamics.studio import (
    ANIM_SERIES_CHOICES,
    CANONICAL_EXPERIMENTS,
    SERIES_CHOICES,
    render_animations,
    render_posters,
    render_showcase,
)
from braitenberg.dynamics.themes import ASPECT_SPECS, THEMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m braitenberg.dynamics",
        description="Unified 3-Series Strogatz × Shannon complex dynamics studio for Braitenberg Vehicles.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    poster_exp_choices = tuple(dict.fromkeys((*CANONICAL_EXPERIMENTS, *PHASE_PORTRAIT_EXPERIMENTS, "all")))
    anim_exp_choices = (*CANONICAL_EXPERIMENTS, "all")
    act_choices = (*ATLAS_ACTS, "all")
    scene_choices = (*tuple(dict.fromkeys((*ATLAS_SCENES, *SCENE_STEMS.keys()))), "all")

    # --- poster ---
    p_poster = sub.add_parser(
        "poster",
        help="Render static 4K plates across tripanels, phase-portraits, atlas, or all.",
    )
    p_poster.add_argument(
        "--series",
        choices=SERIES_CHOICES,
        default=None,
        help="Target poster series: tripanels (default with --experiment), phase-portraits, atlas, or all.",
    )
    p_poster.add_argument(
        "--experiment",
        choices=poster_exp_choices,
        default=None,
        help="Pure single-species experiment ID or 'all' (for tripanels / phase-portraits).",
    )
    p_poster.add_argument(
        "--act",
        choices=act_choices,
        default=None,
        help="Phase-Atlas act ('1', '2', '3', '4', 'atlas', or 'all') for --series atlas.",
    )
    p_poster.add_argument(
        "--neural-view",
        choices=NEURAL_VIEWS,
        default="composite",
        help="Shannon nervous-system rendering mode for tripanels (default: composite).",
    )
    p_poster.add_argument(
        "--theme",
        choices=tuple(THEMES),
        default="cosmic",
        help="Visual aesthetic theme (default: cosmic).",
    )
    p_poster.add_argument(
        "--aspect",
        choices=tuple(ASPECT_SPECS),
        default="16:9",
        help="Canvas aspect ratio (default: 16:9).",
    )
    p_poster.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for generated posters.",
    )
    p_poster.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Override output DPI (defaults to 4K-grade DPI per aspect ratio).",
    )
    p_poster.add_argument(
        "--format",
        choices=("png", "svg"),
        default="png",
        help="Vector or raster image format (default: png).",
    )
    p_poster.add_argument(
        "--suffix",
        default="",
        help="Optional filename stem suffix before extension (e.g. '_textbook').",
    )
    p_poster.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-rendering poster files that already exist on disk.",
    )

    # --- animate ---
    p_anim = sub.add_parser(
        "animate",
        help="Render synchronized animations for tripanels or phase-atlas scenes.",
    )
    p_anim.add_argument(
        "--series",
        choices=ANIM_SERIES_CHOICES,
        default=None,
        help="Animation series: tripanels (default with --experiment), atlas (default with --scene), or all.",
    )
    p_anim.add_argument(
        "--experiment",
        choices=anim_exp_choices,
        default=None,
        help="Canonical isolated Tri-Panel experiment ID or 'all'.",
    )
    p_anim.add_argument(
        "--scene",
        choices=scene_choices,
        default=None,
        help="Classic Phase-Atlas scene ('orbiter', 'ftle-swarm', 'plasticity', 'daydreamer', or 'all').",
    )
    p_anim.add_argument(
        "--neural-view",
        choices=NEURAL_VIEWS,
        default="composite",
        help="Shannon nervous-system rendering mode for tripanels (default: composite).",
    )
    p_anim.add_argument(
        "--theme",
        choices=tuple(THEMES),
        default="cosmic",
        help="Visual aesthetic theme (default: cosmic).",
    )
    p_anim.add_argument(
        "--aspect",
        choices=tuple(ASPECT_SPECS),
        default="16:9",
        help="Video/GIF aspect ratio (default: 16:9).",
    )
    p_anim.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frame rate in frames per second (default: 60).",
    )
    p_anim.add_argument(
        "--seconds",
        type=float,
        default=7.0,
        help="Animation duration in seconds (default: 7.0).",
    )
    p_anim.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for generated animations.",
    )
    p_anim.add_argument(
        "--formats",
        default="gif,mp4",
        help="Comma-separated output formats from {gif,mp4} (default: gif,mp4).",
    )
    p_anim.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Optional DPI override for animation rendering.",
    )
    p_anim.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-rendering animation files that already exist on disk.",
    )

    # --- showcase ---
    p_show = sub.add_parser(
        "showcase",
        help="Generate the standardized 3-Series gallery tree (tripanels, phase_portraits, phase_atlas).",
    )
    p_show.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/strogatz-dynamics/media"),
        help="Gallery root output directory (default: experiments/strogatz-dynamics/media).",
    )
    p_show.add_argument(
        "--series",
        choices=SERIES_CHOICES,
        default="all",
        help="Which showcase series to build: all (default), tripanels, phase-portraits, or atlas.",
    )
    p_show.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-rendering existing showcase files for fast incremental builds.",
    )
    p_show.add_argument(
        "--fps",
        type=int,
        default=60,
        help="MP4 frame rate for showcase videos (default: 60).",
    )
    p_show.add_argument(
        "--seconds",
        type=float,
        default=7.0,
        help="Duration in seconds for showcase animations (default: 7.0).",
    )
    p_show.add_argument(
        "--neural-view",
        choices=NEURAL_VIEWS,
        default="composite",
        help="Shannon nervous-system rendering mode for tripanels (default: composite).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> list[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.subcommand == "poster":
            written = render_posters(
                series=args.series,
                experiment=args.experiment,
                act=args.act,
                neural_view=args.neural_view,
                theme=args.theme,
                aspect=args.aspect,
                out_dir=args.out_dir,
                dpi=args.dpi,
                fmt=args.format,
                suffix=args.suffix,
                skip_existing=args.skip_existing,
            )
        elif args.subcommand == "animate":
            formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
            written = render_animations(
                series=args.series,
                experiment=args.experiment,
                scene=args.scene,
                neural_view=args.neural_view,
                theme=args.theme,
                aspect=args.aspect,
                fps=args.fps,
                seconds=args.seconds,
                out_dir=args.out_dir,
                formats=formats,
                dpi=args.dpi,
                skip_existing=args.skip_existing,
            )
        elif args.subcommand == "showcase":
            written = render_showcase(
                out_dir=args.out_dir,
                series=args.series,
                skip_existing=args.skip_existing,
                fps=args.fps,
                seconds=args.seconds,
                neural_view=args.neural_view,
            )
        else:  # pragma: no cover
            parser.error(f"Unknown subcommand: {args.subcommand}")
            return []
    except ValueError as exc:
        parser.error(str(exc))
        return []

    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    main()
