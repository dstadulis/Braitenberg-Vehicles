"""Tests for braitenberg.dynamics.themes, diagram_mpl, studio, Axiom 1 enforcement, and CLI."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import braitenberg.dynamics  # noqa: F401 - configure MPLCONFIGDIR before matplotlib imports
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from braitenberg.definition import load_definition
from braitenberg.diagram import bundles
from braitenberg.dynamics import __main__ as dynamics_cli
from braitenberg.dynamics.diagram_mpl import (
    NEURAL_VIEWS,
    _rect_overlap_area,
    render_shannon_neural_panel,
    solve_part_labels,
)
from braitenberg.dynamics.studio import CANONICAL_EXPERIMENTS, CometArtist, HeroMarkerArtist
from braitenberg.dynamics.themes import (
    COSMIC_THEME,
    TEXTBOOK_THEME,
    activity_fill_color,
    activity_fill_rgba,
    aspect_figsize,
    barcode_cmap,
    basin_colors,
    codeword_color,
    codeword_colors,
    entropy_cmap,
    field_cmap,
    ftle_cmap,
    get_theme,
    phase_cmap,
    plot_comet_trails,
    plot_glow_line,
    plot_glow_scatter,
    plot_hero_beacon,
    plot_lamp_beacon,
    plot_strogatz_equilibrium,
    style_axes,
)
from braitenberg.vehicle import Vehicle

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_includes_dynamics_subpackage() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = data["tool"]["setuptools"]["packages"]
    assert "braitenberg" in packages
    assert "braitenberg.dynamics" in packages


def test_never_imports_pygame() -> None:
    code = (
        "import sys; "
        "import braitenberg.dynamics; "
        "import braitenberg.dynamics.themes; "
        "import braitenberg.dynamics.diagram_mpl; "
        "import braitenberg.dynamics.studio; "
        "import braitenberg.dynamics.__main__; "
        "assert 'pygame' not in sys.modules, 'pygame was imported!'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_themes_codeword_palettes_and_primitives() -> None:
    assert get_theme("cosmic") == COSMIC_THEME
    assert get_theme("TEXTBOOK") == TEXTBOOK_THEME
    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("neon-vapor")

    for aspect, expected_ratio in (("16:9", 16 / 9), ("1:1", 1.0), ("9:16", 9 / 16)):
        w, h, dpi = aspect_figsize(aspect, "poster")
        assert dpi >= 200
        assert (w / h) == pytest.approx(expected_ratio, rel=1e-2)

    for theme in (COSMIC_THEME, TEXTBOOK_THEME):
        cols = codeword_colors(theme, 6)
        assert len(cols) == 6
        assert len(set(cols)) == 6
        assert codeword_color(theme, 0, 6) == cols[0]
        assert codeword_color(theme, -1, 6) != cols[0]

        assert entropy_cmap(theme)(0.5) is not None
        assert barcode_cmap(theme, is_learning=False)(0.5) is not None
        assert barcode_cmap(theme, is_learning=True)(0.5) is not None
        assert ftle_cmap(theme)(0.5) is not None
        assert field_cmap(theme)(0.5) is not None
        assert phase_cmap(theme)(0.25) is not None
        assert len(basin_colors(theme, 5)) == 5

        rgba_grid = activity_fill_rgba(np.array([[0.0, 0.5], [1.0, 0.25]]), theme)
        assert rgba_grid.shape == (2, 2, 4)
        assert len(activity_fill_color(0.8, theme)) == 4

        fig = Figure(figsize=(4, 3), dpi=80, facecolor=theme.fig_bg)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        style_axes(ax, theme, tag="A", title="Test", xlabel="x", ylabel="y")

        assert len(plot_glow_line(ax, [0, 1, 2], [0, 1, 0], theme.cyan, theme)) == (
            3 if theme.glow_enabled else 1
        )
        assert len(plot_glow_scatter(ax, [0.5], [0.5], theme.coral, theme)) == (
            3 if theme.glow_enabled else 1
        )
        assert len(plot_hero_beacon(ax, 1.0, 0.5, theme)) >= 2
        assert len(plot_lamp_beacon(ax, 1.5, 0.5, theme)) >= 2

        xs = np.array([[95.0, 10.0], [98.0, 12.0], [2.0, 14.0]])
        ys = np.array([[50.0, 20.0], [50.0, 22.0], [50.0, 24.0]])
        assert len(plot_comet_trails(ax, xs, ys, [theme.cyan, theme.amber], theme, arena_size=(100.0, 100.0))) >= 2

        from matplotlib.colors import to_rgba

        ca = CometArtist(ax, theme, arena_size=(100.0, 100.0))
        ca.update(xs, ys, np.array([to_rgba(theme.cyan), to_rgba(theme.amber)]))
        ha = HeroMarkerArtist(ax, theme, arena_size=(100.0, 100.0), fov_cone=(60.0, 25.0))
        ha.update(xs[:, 0], ys[:, 0], to_rgba(theme.cyan), heading_rad=0.5)
        FigureCanvasAgg(fig).draw()

        for eq in ("stable_node", "unstable_spiral", "saddle"):
            assert len(plot_strogatz_equilibrium(ax, 1.0, 0.2, eq, theme)) >= 1


@pytest.mark.parametrize(
    "vehicle_file",
    [
        "vehicle4a-orbiter.yaml",
        "vehicle5-moth.yaml",
        "vehicle5-forager.yaml",
        "vehicle7-wary-moth.yaml",
        "vehicle9-shy.yaml",
        "vehicle12-daydreamer.yaml",
    ],
)
def test_diagram_mpl_collision_free_labels_and_views(vehicle_file: str) -> None:
    vdef = load_definition(REPO_ROOT / "vehicles" / vehicle_file)
    diag = Vehicle(vdef, 0, 0, 0).diagram()
    placements = solve_part_labels(diag.parts, bundles(diag.wires), diag.body)
    non_module_parts = [p for p in diag.parts if p.kind != "module"]
    assert len(placements) == len(non_module_parts)

    boxes = list(placements.items())
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            area = _rect_overlap_area(boxes[i][1]["bbox"], boxes[j][1]["bbox"])
            assert area <= 1e-4, f"Labels '{boxes[i][0]}' and '{boxes[j][0]}' overlap in {vehicle_file}"

    # Verify all 4 neural views render cleanly via ShannonNeuralPanelArtist
    n_veh, m_units = 12, max(1, len(vdef.units))
    synth_snap = {
        "hero_diagram": diag,
        "part_codeword_arcs": {p.name: [(0, 0.45), (1, 0.25)] for p in diag.parts},
        "part_entropy": {p.name: (0.0 if p.name == "awake" else 0.72) for p in diag.parts},
        "joint_entropy_bits": 1.85,
        "total_correlation_bits": 1.12,
        "spatial_mi_bits": 0.64,
        "wire_info_bits": {(w.source, w.target, w.sign): 0.42 for w in bundles(diag.wires)},
        "barcodes": {
            p.name: {
                "hist": np.linspace(0.02, 0.12, 20) / np.sum(np.linspace(0.02, 0.12, 20)),
                "hero_val": 0.4,
                "min": 0.0,
                "max": 1.0,
                "entropy_bits": 0.6,
                "is_learning": False,
                "gate": None,
            }
            for p in diag.parts[:5]
        },
        "raster_matrix": (np.arange(n_veh * m_units).reshape(n_veh, m_units) % 2).astype(float),
        "raster_codeword_ids": np.arange(n_veh) % 4,
        "raster_unit_names": [u.name for u in vdef.units] or ["s0"],
        "hero_sorted_row": 2,
        "vehicle_codeword_ids": np.arange(n_veh) % 4,
    }
    for view in NEURAL_VIEWS:
        fig = Figure(figsize=(4.5, 6.5), dpi=72, facecolor=COSMIC_THEME.fig_bg)
        canvas = FigureCanvasAgg(fig)
        gs = fig.add_gridspec(1, 1)
        artist = render_shannon_neural_panel(fig, gs[0, 0], synth_snap, COSMIC_THEME, neural_view=view)
        artist.update(synth_snap)
        canvas.draw()


def test_axiom_1_strict_single_experiment_binding() -> None:
    """Every canonical studio experiment binds strictly to 1 ExperimentSpec & rejects mashups."""
    from braitenberg.dynamics.shannon import (
        EXPERIMENTS,
        AxiomViolationError,
        validate_single_experiment_axiom,
    )
    from braitenberg.dynamics.studio import get_experiment_rollout

    assert set(CANONICAL_EXPERIMENTS) == set(EXPERIMENTS.keys())

    # Passing multiple experiments must immediately raise AxiomViolationError
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom([EXPERIMENTS["v5_moth"], EXPERIMENTS["v7_wary_moth"]])

    # Mixing heterogeneous vehicles must raise AxiomViolationError
    v_moth = Vehicle(load_definition(REPO_ROOT / "vehicles/vehicle5-moth.yaml"), 0, 0, 0)
    v_shy = Vehicle(load_definition(REPO_ROOT / "vehicles/vehicle9-shy.yaml"), 0, 0, 0)
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom(EXPERIMENTS["v5_moth"], vehicles=[v_moth, v_shy])

    # v9_shy rollout must strictly isolate Shy vehicles (excluding Chaser/Walker stimulus agents)
    shy_rollout = get_experiment_rollout("v9_shy", steps=40, stride=10)
    assert shy_rollout["focal_def"].general.name == "Shy"
    assert shy_rollout["focal_def"].general.type == "9"
    snap0 = shy_rollout["snapshots"][0]
    assert snap0["hero_diagram"].subtitle == "Vehicle 9: Shy"
    assert snap0["raster_matrix"].shape[0] == shy_rollout["n_focal"]


@pytest.mark.parametrize(
    ("aspect", "ratio"),
    [("16:9", 16 / 9), ("1:1", 1.0), ("9:16", 9 / 16)],
)
def test_cli_poster_aspects_and_themes(tmp_path: Path, aspect: str, ratio: float) -> None:
    theme = "textbook" if aspect == "1:1" else "cosmic"
    written = dynamics_cli.main(
        [
            "poster",
            "--experiment",
            "v9_shy",
            "--neural-view",
            "composite",
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
    assert out.name == "v9_shy_tripanel.png"
    assert out.exists() and out.stat().st_size > 5_000
    with Image.open(out) as im:
        w, h = im.size
        assert (w / h) == pytest.approx(ratio, rel=0.02)


def test_cli_all_posters_and_svg(tmp_path: Path) -> None:
    written = dynamics_cli.main(
        [
            "poster",
            "--experiment",
            "all",
            "--theme",
            "cosmic",
            "--aspect",
            "16:9",
            "--dpi",
            "60",
            "--out-dir",
            str(tmp_path),
        ]
    )
    names = {p.name for p in written}
    assert names == {f"{exp}_tripanel.png" for exp in CANONICAL_EXPERIMENTS}
    for p in written:
        assert p.exists() and p.stat().st_size > 5_000

    svg_out = dynamics_cli.main(
        [
            "poster",
            "--experiment",
            "v5_moth",
            "--neural-view",
            "entropy",
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


@pytest.mark.parametrize("exp_id", ["v4a_orbiter", "v5_moth", "v7_wary_moth", "v9_shy", "v12_daydreamer"])
def test_cli_animate_gif_and_mp4(tmp_path: Path, exp_id: str) -> None:
    has_ffmpeg = shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists()
    fmts = "gif,mp4" if has_ffmpeg else "gif"
    written = dynamics_cli.main(
        [
            "animate",
            "--experiment",
            exp_id,
            "--theme",
            "cosmic",
            "--aspect",
            "16:9",
            "--fps",
            "8",
            "--seconds",
            "0.5",
            "--dpi",
            "52",
            "--formats",
            fmts,
            "--out-dir",
            str(tmp_path),
        ]
    )
    gifs = [p for p in written if p.suffix == ".gif"]
    assert len(gifs) == 1
    assert gifs[0].name == f"{exp_id}.gif"
    with Image.open(gifs[0]) as im:
        assert getattr(im, "is_animated", False)
        assert im.n_frames >= 3

    if has_ffmpeg:
        mp4s = [p for p in written if p.suffix == ".mp4"]
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 1_000


def test_showcase_gallery_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify showcase populates all 36 standardized 1-Experiment-Per-File gallery assets."""
    from braitenberg.dynamics import studio

    orig_figsize = studio.aspect_figsize
    monkeypatch.setattr(
        studio,
        "aspect_figsize",
        lambda aspect, target="poster": (
            orig_figsize(aspect, target)[0],
            orig_figsize(aspect, target)[1],
            44,
        ),
    )

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
    assert len(written) >= 18
    expected_top = (
        {f"{exp}_tripanel.png" for exp in CANONICAL_EXPERIMENTS}
        | {f"{exp}_tripanel_textbook.png" for exp in CANONICAL_EXPERIMENTS}
        | {f"{exp}.gif" for exp in CANONICAL_EXPERIMENTS}
    )
    actual_top = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert expected_top.issubset(actual_top)

    if shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists():
        mp4_names = {p.name for p in (tmp_path / "mp4").glob("*.mp4")}
        for exp in CANONICAL_EXPERIMENTS:
            for tag in ("16x9", "1x1", "9x16"):
                assert f"{exp}_{tag}.mp4" in mp4_names
