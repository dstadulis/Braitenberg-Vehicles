"""Tests for braitenberg.dynamics.themes, studio, and CLI entry point."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from braitenberg.dynamics import __main__ as dynamics_cli
from braitenberg.dynamics.themes import (
    COSMIC_THEME,
    TEXTBOOK_THEME,
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
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_includes_dynamics_subpackage() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = data["tool"]["setuptools"]["packages"]
    assert "braitenberg" in packages
    assert "braitenberg.dynamics" in packages


def test_themes_registry_and_primitives() -> None:
    assert get_theme("cosmic") == COSMIC_THEME
    assert get_theme("TEXTBOOK") == TEXTBOOK_THEME
    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("neon-vapor")

    for aspect, expected_ratio in (("16:9", 16 / 9), ("1:1", 1.0), ("9:16", 9 / 16)):
        w, h, dpi = aspect_figsize(aspect, "poster")
        assert dpi >= 200
        assert (w / h) == pytest.approx(expected_ratio, rel=1e-2)

    for theme in (COSMIC_THEME, TEXTBOOK_THEME):
        assert ftle_cmap(theme)(0.5) is not None
        assert field_cmap(theme)(0.5) is not None
        assert phase_cmap(theme)(0.25) is not None
        assert len(basin_colors(theme, 5)) == 5

        fig = Figure(figsize=(4, 3), dpi=80, facecolor=theme.fig_bg)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        style_axes(ax, theme, tag="A", title="Test", xlabel="x", ylabel="y")

        lines = plot_glow_line(ax, [0, 1, 2], [0, 1, 0], theme.cyan, theme, label="curve")
        assert len(lines) == (3 if theme.glow_enabled else 1)

        sc = plot_glow_scatter(ax, [0.5], [0.5], theme.coral, theme)
        assert len(sc) == (3 if theme.glow_enabled else 1)

        # Periodic wrap jump between t=1 (x=98) and t=2 (x=2) should be masked cleanly
        xs = np.array([[95.0, 10.0], [98.0, 12.0], [2.0, 14.0]])
        ys = np.array([[50.0, 20.0], [50.0, 22.0], [50.0, 24.0]])
        artists = plot_comet_trails(ax, xs, ys, [theme.cyan, theme.amber], theme, arena_size=(100.0, 100.0))
        assert len(artists) >= 2

        from matplotlib.colors import to_rgba
        from braitenberg.dynamics.studio import CometArtist

        ca = CometArtist(ax, theme, arena_size=(100.0, 100.0))
        ca.update(xs, ys, np.array([to_rgba(theme.cyan), to_rgba(theme.amber)]))
        FigureCanvasAgg(fig).draw()

        for eq in ("stable_node", "unstable_spiral", "saddle"):
            assert len(plot_strogatz_equilibrium(ax, 1.0, 0.2, eq, theme)) >= 1
        assert len(plot_lamp_beacon(ax, 1.5, 0.5, theme)) >= 2


def test_never_imports_pygame() -> None:
    code = (
        "import sys; "
        "import braitenberg.dynamics; "
        "import braitenberg.dynamics.themes; "
        "import braitenberg.dynamics.studio; "
        "import braitenberg.dynamics.__main__; "
        "assert 'pygame' not in sys.modules, 'pygame was imported!'"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert proc.returncode == 0


@pytest.mark.parametrize(
    ("aspect", "ratio"),
    [("16:9", 16 / 9), ("1:1", 1.0), ("9:16", 9 / 16)],
)
def test_cli_poster_aspects_and_themes(tmp_path: Path, aspect: str, ratio: float) -> None:
    theme = "textbook" if aspect == "1:1" else "cosmic"
    written = dynamics_cli.main(
        [
            "poster",
            "--act",
            "1",
            "--theme",
            theme,
            "--aspect",
            aspect,
            "--dpi",
            "72",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert len(written) == 1
    out = written[0]
    assert out.exists() and out.stat().st_size > 5_000
    with Image.open(out) as im:
        w, h = im.size
        assert (w / h) == pytest.approx(ratio, rel=0.02)


def test_cli_all_posters_and_svg(tmp_path: Path) -> None:
    written = dynamics_cli.main(
        [
            "poster",
            "--act",
            "all",
            "--theme",
            "cosmic",
            "--aspect",
            "16:9",
            "--dpi",
            "64",
            "--out-dir",
            str(tmp_path),
        ]
    )
    names = {p.name for p in written}
    assert names == {
        "act1_polar_phase_portraits.png",
        "act2_lagrangian_chaos_ftle.png",
        "act3_hysteresis_and_plasticity.png",
        "act4_neural_criticality.png",
        "strogatz_atlas.png",
    }
    for p in written:
        assert p.exists() and p.stat().st_size > 5_000

    svg_out = dynamics_cli.main(
        [
            "poster",
            "--act",
            "1",
            "--theme",
            "textbook",
            "--format",
            "svg",
            "--dpi",
            "72",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert svg_out[0].suffix == ".svg"
    assert "<svg" in svg_out[0].read_text(encoding="utf-8")[:500]


@pytest.mark.parametrize("scene", ["orbiter", "ftle-swarm", "plasticity", "daydreamer"])
def test_cli_animate_gif_and_mp4(tmp_path: Path, scene: str) -> None:
    has_ffmpeg = shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists()
    fmts = "gif,mp4" if has_ffmpeg else "gif"
    written = dynamics_cli.main(
        [
            "animate",
            "--scene",
            scene,
            "--theme",
            "cosmic",
            "--aspect",
            "16:9",
            "--fps",
            "8",
            "--seconds",
            "0.5",
            "--dpi",
            "54",
            "--formats",
            fmts,
            "--out-dir",
            str(tmp_path),
        ]
    )
    gifs = [p for p in written if p.suffix == ".gif"]
    assert len(gifs) == 1
    with Image.open(gifs[0]) as im:
        assert getattr(im, "is_animated", False)
        assert im.n_frames >= 3

    if has_ffmpeg:
        mp4s = [p for p in written if p.suffix == ".mp4"]
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 1_000


def test_showcase_gallery_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify showcase populates the exact canonical filenames expected by the monograph."""
    from braitenberg.dynamics import studio

    # Use compact DPI in unit test so full gallery contract runs in seconds
    orig_figsize = studio.aspect_figsize
    monkeypatch.setattr(studio, "aspect_figsize", lambda aspect, target="poster": (orig_figsize(aspect, target)[0], orig_figsize(aspect, target)[1], 48))

    written = dynamics_cli.main(
        [
            "showcase",
            "--fps",
            "6",
            "--seconds",
            "0.4",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert len(written) >= 14
    expected_top = {
        "strogatz_atlas.png",
        "act1_polar_phase_portraits.png",
        "act2_lagrangian_chaos_ftle.png",
        "act3_hysteresis_and_plasticity.png",
        "act4_neural_criticality.png",
        "strogatz_atlas_textbook.png",
        "orbiter.gif",
        "ftle_swarm.gif",
        "plasticity.gif",
        "daydreamer.gif",
    }
    actual_top = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert expected_top.issubset(actual_top)

    if shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists():
        mp4_names = {p.name for p in (tmp_path / "mp4").glob("*.mp4")}
        for stem in ("orbiter", "ftle_swarm", "plasticity", "daydreamer"):
            for tag in ("16x9", "1x1", "9x16"):
                assert f"{stem}_{tag}.mp4" in mp4_names
