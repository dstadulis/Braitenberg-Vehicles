"""Command-line interface for Braitenberg Strogatz × Shannon Tri-Panel Studio.

Examples:
    python -m braitenberg.dynamics poster --experiment v9_shy --theme cosmic --aspect 16:9 --out-dir /tmp/out
    python -m braitenberg.dynamics animate --experiment v5_moth --fps 60 --seconds 7 --out-dir /tmp/out
    python -m braitenberg.dynamics showcase --out-dir experiments/strogatz-dynamics/media
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from braitenberg.dynamics.diagram_mpl import NEURAL_VIEWS
from braitenberg.dynamics.studio import (
    CANONICAL_EXPERIMENTS,
    render_animations,
    render_posters,
    render_showcase,
)
from braitenberg.dynamics.themes import ASPECT_SPECS, THEMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m braitenberg.dynamics",
        description="Strogatz × Shannon complex dynamics & Tri-Panel visual studio for Braitenberg Vehicles.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    exp_choices = (*CANONICAL_EXPERIMENTS, "all")

    # --- poster ---
    p_poster = sub.add_parser(
        "poster",
        help="Render Axiom-1 Tri-Panel static posters (1 isolated experiment per figure).",
    )
    p_poster.add_argument(
        "--experiment",
        choices=exp_choices,
        default="all",
        help="Canonical isolated experiment ID or 'all' (default: all).",
    )
    p_poster.add_argument(
        "--neural-view",
        choices=NEURAL_VIEWS,
        default="composite",
        help="Shannon nervous-system rendering mode (default: composite).",
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

    # --- animate ---
    p_anim = sub.add_parser(
        "animate",
        help="Render synchronized Tri-Panel animations (Arena + Phase Space + Shannon Brain).",
    )
    p_anim.add_argument(
        "--experiment",
        choices=exp_choices,
        default="all",
        help="Canonical isolated experiment ID or 'all' (default: all).",
    )
    p_anim.add_argument(
        "--neural-view",
        choices=NEURAL_VIEWS,
        default="composite",
        help="Shannon nervous-system rendering mode (default: composite).",
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

    # --- showcase ---
    p_show = sub.add_parser(
        "showcase",
        help="Generate the standardized 1-Experiment-Per-File gallery (cosmic + textbook 4K posters, GIFs, MP4s).",
    )
    p_show.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/strogatz-dynamics/media"),
        help="Gallery output directory (default: experiments/strogatz-dynamics/media).",
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
        help="Shannon nervous-system rendering mode (default: composite).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> list[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "poster":
        written = render_posters(
            experiment=args.experiment,
            neural_view=args.neural_view,
            theme=args.theme,
            aspect=args.aspect,
            out_dir=args.out_dir,
            dpi=args.dpi,
            fmt=args.format,
        )
    elif args.subcommand == "animate":
        formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
        written = render_animations(
            experiment=args.experiment,
            neural_view=args.neural_view,
            theme=args.theme,
            aspect=args.aspect,
            fps=args.fps,
            seconds=args.seconds,
            out_dir=args.out_dir,
            formats=formats,
            dpi=args.dpi,
        )
    elif args.subcommand == "showcase":
        written = render_showcase(
            out_dir=args.out_dir,
            fps=args.fps,
            seconds=args.seconds,
            neural_view=args.neural_view,
        )
    else:  # pragma: no cover
        parser.error(f"Unknown subcommand: {args.subcommand}")
        return []

    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    main()
