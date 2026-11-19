"""Tests for braitenberg.dynamics 3-series studio, phase_portraits, phase_atlas, Axiom 1, and CLI."""

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
from braitenberg.dynamics.phase_atlas import (
    ATLAS_ACTS,
    ATLAS_POSTER_METADATA,
    ATLAS_SCENE_METADATA,
    ATLAS_SCENES,
    POSTER_FILENAMES,
    SCENE_STEMS,
    build_atlas_poster,
)
from braitenberg.dynamics.phase_portraits import (
    PHASE_PORTRAIT_EXPERIMENTS,
    build_phase_portrait_poster,
)
from braitenberg.dynamics.shannon import (
    EXPERIMENTS,
    PHASE_EXPERIMENTS,
    AxiomViolationError,
    validate_single_experiment_axiom,
)
from braitenberg.dynamics.studio import (
    CANONICAL_EXPERIMENTS,
    CometArtist,
    HeroMarkerArtist,
    build_experiment_poster,
    get_experiment_rollout,
)
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
        "import braitenberg.dynamics.phase_portraits; "
        "import braitenberg.dynamics.phase_atlas; "
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


def test_axiom_1_strict_binding_and_comparative_atlas_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every tripanels & phase-portraits render enforces Axiom 1; phase_atlas is tagged separately."""
    assert set(CANONICAL_EXPERIMENTS) == set(EXPERIMENTS.keys())
    assert tuple(PHASE_EXPERIMENTS.keys()) == PHASE_PORTRAIT_EXPERIMENTS
    assert len(PHASE_PORTRAIT_EXPERIMENTS) == 8

    # Comparative phase_atlas metadata explicitly marks multi-vehicle synthesis plates
    for key in ("atlas", "1", "3"):
        meta = ATLAS_POSTER_METADATA[key]
        assert meta.series == "phase_atlas"
        assert meta.comparative_synthesis is True
        with pytest.raises(AxiomViolationError):
            validate_single_experiment_axiom(meta)

    assert ATLAS_SCENE_METADATA["plasticity"].series == "phase_atlas"
    assert ATLAS_SCENE_METADATA["plasticity"].comparative_synthesis is True
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom(ATLAS_SCENE_METADATA["plasticity"])

    atlas_fig = build_atlas_poster(theme_name="cosmic", aspect="16:9", dpi=48)
    assert getattr(atlas_fig, "series", None) == "phase_atlas"
    assert getattr(atlas_fig, "comparative_synthesis", None) is True

    # Passing multiple experiments or heterogeneous vehicles must raise AxiomViolationError
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom([PHASE_EXPERIMENTS["v2a_fear"], PHASE_EXPERIMENTS["v3a_love"]])

    v_fear = Vehicle(load_definition(REPO_ROOT / "vehicles/vehicle2a-fear.yaml"), 0, 0, 0)
    v_love = Vehicle(load_definition(REPO_ROOT / "vehicles/vehicle3a-love.yaml"), 0, 0, 0)
    with pytest.raises(AxiomViolationError):
        validate_single_experiment_axiom(PHASE_EXPERIMENTS["v2a_fear"], vehicles=[v_fear, v_love])

    # Spy on validate_single_experiment_axiom across EVERY single tripanel & phase-portrait build!
    from braitenberg.dynamics import phase_portraits, shannon

    validated_ids: list[str] = []
    orig_validate = shannon.validate_single_experiment_axiom

    def _spy_validate(spec_or_specs, vehicles=None, scenario=None):
        res = orig_validate(spec_or_specs, vehicles=vehicles, scenario=scenario)
        validated_ids.append(res.experiment_id)
        return res

    monkeypatch.setattr(shannon, "validate_single_experiment_axiom", _spy_validate)
    monkeypatch.setattr(phase_portraits, "validate_single_experiment_axiom", _spy_validate)

    for pp_id in PHASE_PORTRAIT_EXPERIMENTS:
        validated_ids.clear()
        fig_pp = build_phase_portrait_poster(pp_id, theme_name="cosmic", aspect="16:9", dpi=48)
        assert pp_id in validated_ids
        assert set(validated_ids) == {pp_id}, f"Cross-experiment leak in phase_portrait {pp_id}: {validated_ids}"
        assert fig_pp.axiom1_metadata["series"] == "phase_portraits"
        assert fig_pp.axiom1_metadata["experiment_id"] == pp_id
        assert fig_pp.comparative_synthesis is False

    for tri_id in CANONICAL_EXPERIMENTS:
        validated_ids.clear()
        fig_tri = build_experiment_poster(tri_id, theme_name="cosmic", aspect="16:9", dpi=48)
        assert tri_id in validated_ids
        assert set(validated_ids) == {tri_id}, f"Cross-experiment leak in tripanel {tri_id}: {validated_ids}"
        assert fig_tri.axiom1_metadata["series"] == "tripanels"
        assert fig_tri.axiom1_metadata["experiment_id"] == tri_id
        assert fig_tri.comparative_synthesis is False

    shy_rollout = get_experiment_rollout("v9_shy", steps=40, stride=10)
    assert shy_rollout["focal_def"].general.name == "Shy"
    assert shy_rollout["snapshots"][0]["hero_diagram"].subtitle == "Vehicle 9: Shy"


@pytest.mark.parametrize(
    ("aspect", "ratio"),
    [("16:9", 16 / 9), ("1:1", 1.0), ("9:16", 9 / 16)],
)
def test_cli_poster_aspects_themes_and_series(tmp_path: Path, aspect: str, ratio: float) -> None:
    theme = "textbook" if aspect == "1:1" else "cosmic"

    # Series II (default tripanels)
    written_tri = dynamics_cli.main(
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
            "64",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert len(written_tri) == 1
    assert written_tri[0].name == "v9_shy_tripanel.png"
    with Image.open(written_tri[0]) as im:
        assert (im.size[0] / im.size[1]) == pytest.approx(ratio, rel=0.02)

    # Series I (phase-portraits)
    written_pp = dynamics_cli.main(
        [
            "poster",
            "--series",
            "phase-portraits",
            "--experiment",
            "v11_cautious_driver",
            "--theme",
            theme,
            "--aspect",
            aspect,
            "--dpi",
            "64",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert len(written_pp) == 1
    assert written_pp[0].name == "v11_cautious_driver_phase.png"
    with Image.open(written_pp[0]) as im:
        assert (im.size[0] / im.size[1]) == pytest.approx(ratio, rel=0.02)


def test_cli_all_posters_atlas_and_svg(tmp_path: Path) -> None:
    tri_dir = tmp_path / "tri"
    pp_dir = tmp_path / "pp"
    atlas_dir = tmp_path / "atlas"

    written_tri = dynamics_cli.main(
        ["poster", "--series", "tripanels", "--experiment", "all", "--dpi", "52", "--out-dir", str(tri_dir)]
    )
    assert {p.name for p in written_tri} == {f"{exp}_tripanel.png" for exp in CANONICAL_EXPERIMENTS}

    written_pp = dynamics_cli.main(
        ["poster", "--series", "phase-portraits", "--experiment", "all", "--dpi", "52", "--out-dir", str(pp_dir)]
    )
    assert {p.name for p in written_pp} == {f"{exp}_phase.png" for exp in PHASE_PORTRAIT_EXPERIMENTS}

    written_atlas = dynamics_cli.main(
        ["poster", "--series", "atlas", "--act", "all", "--dpi", "52", "--out-dir", str(atlas_dir)]
    )
    assert {p.name for p in written_atlas} == {f"{POSTER_FILENAMES[a]}.png" for a in ATLAS_ACTS}

    svg_out = dynamics_cli.main(
        [
            "poster",
            "--series",
            "phase-portraits",
            "--experiment",
            "v2a_fear",
            "--theme",
            "textbook",
            "--format",
            "svg",
            "--dpi",
            "64",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert svg_out[0].name == "v2a_fear_phase.svg"
    assert "<svg" in svg_out[0].read_text(encoding="utf-8")[:500]


def test_cli_animate_tripanels_and_atlas_scenes(tmp_path: Path) -> None:
    has_ffmpeg = shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists()
    fmts = "gif,mp4" if has_ffmpeg else "gif"

    # Tri-Panel animation
    written_tri = dynamics_cli.main(
        [
            "animate",
            "--series",
            "tripanels",
            "--experiment",
            "v5_moth",
            "--fps",
            "8",
            "--seconds",
            "0.5",
            "--dpi",
            "48",
            "--formats",
            fmts,
            "--out-dir",
            str(tmp_path / "tri_anim"),
        ]
    )
    tri_gifs = [p for p in written_tri if p.suffix == ".gif"]
    assert len(tri_gifs) == 1 and tri_gifs[0].name == "v5_moth.gif"
    with Image.open(tri_gifs[0]) as im:
        assert getattr(im, "is_animated", False) and im.n_frames >= 3

    # Restored Phase-Atlas animation (--scene implies --series atlas)
    written_atlas = dynamics_cli.main(
        [
            "animate",
            "--scene",
            "plasticity",
            "--fps",
            "8",
            "--seconds",
            "0.5",
            "--dpi",
            "48",
            "--formats",
            fmts,
            "--out-dir",
            str(tmp_path / "atlas_anim"),
        ]
    )
    atlas_gifs = [p for p in written_atlas if p.suffix == ".gif"]
    assert len(atlas_gifs) == 1 and atlas_gifs[0].name == "plasticity.gif"
    with Image.open(atlas_gifs[0]) as im:
        assert getattr(im, "is_animated", False) and im.n_frames >= 3
    if has_ffmpeg:
        assert any(p.suffix == ".mp4" and p.stat().st_size > 1_000 for p in written_atlas)


def test_showcase_three_series_tree_and_skip_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify showcase populates all 3 series directories cleanly and honors --skip-existing."""
    from braitenberg.dynamics import studio

    orig_figsize = studio.aspect_figsize
    monkeypatch.setattr(
        studio,
        "aspect_figsize",
        lambda aspect, target="poster": (
            orig_figsize(aspect, target)[0],
            orig_figsize(aspect, target)[1],
            40,
        ),
    )

    written = dynamics_cli.main(
        [
            "showcase",
            "--series",
            "all",
            "--fps",
            "6",
            "--seconds",
            "0.4",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert len(written) >= 48

    # 1. Series II (top-level <out-dir>/ + <out-dir>/mp4/)
    expected_top = (
        {f"{exp}_tripanel.png" for exp in CANONICAL_EXPERIMENTS}
        | {f"{exp}_tripanel_textbook.png" for exp in CANONICAL_EXPERIMENTS}
        | {f"{exp}.gif" for exp in CANONICAL_EXPERIMENTS}
    )
    actual_top = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert expected_top == actual_top

    # 2. Series I (<out-dir>/phase_portraits/)
    expected_pp = (
        {f"{exp}_phase.png" for exp in PHASE_PORTRAIT_EXPERIMENTS}
        | {f"{exp}_phase_textbook.png" for exp in PHASE_PORTRAIT_EXPERIMENTS}
    )
    actual_pp = {p.name for p in (tmp_path / "phase_portraits").iterdir() if p.is_file()}
    assert expected_pp == actual_pp

    # 3. Series III (<out-dir>/phase_atlas/ + <out-dir>/mp4/phase_atlas/)
    expected_atlas = (
        {f"{POSTER_FILENAMES[a]}.png" for a in ATLAS_ACTS}
        | {f"{POSTER_FILENAMES[a]}_textbook.png" for a in ATLAS_ACTS}
        | {f"{SCENE_STEMS[s]}.gif" for s in ATLAS_SCENES}
    )
    actual_atlas = {p.name for p in (tmp_path / "phase_atlas").iterdir() if p.is_file()}
    assert expected_atlas == actual_atlas

    if shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists():
        mp4_tri = {p.name for p in (tmp_path / "mp4").glob("*.mp4")}
        for exp in CANONICAL_EXPERIMENTS:
            for tag in ("16x9", "1x1", "9x16"):
                assert f"{exp}_{tag}.mp4" in mp4_tri

        mp4_atlas = {p.name for p in (tmp_path / "mp4" / "phase_atlas").glob("*.mp4")}
        for s in ATLAS_SCENES:
            stem = SCENE_STEMS[s]
            for tag in ("16x9", "1x1", "9x16"):
                assert f"{stem}_{tag}.mp4" in mp4_atlas

    # 4. Verify --skip-existing preserves mtimes without re-rendering
    probe_file = tmp_path / "v5_moth_tripanel.png"
    mtime_before = probe_file.stat().st_mtime_ns
    dynamics_cli.main(
        [
            "showcase",
            "--series",
            "all",
            "--skip-existing",
            "--fps",
            "6",
            "--seconds",
            "0.4",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert probe_file.stat().st_mtime_ns == mtime_before


def test_v7_wary_moth_pass_classification_matches_avoidance_bifurcation() -> None:
    """Every pass in v7_wary_moth Panel D must classify pre-burn approach weight w > 0.7 vs <= 0.7."""
    from braitenberg.definition import ACTIVE_STRENGTH
    from braitenberg.dynamics.phase_portraits import _get_v7_wary_moth_data

    data = _get_v7_wary_moth_data()
    pass_rmin = np.asarray(data["pass_rmin"], dtype=float)
    pass_w = np.asarray(data["pass_w"], dtype=float)
    assert len(pass_rmin) >= 15

    repelled = pass_w > ACTIVE_STRENGTH
    assert np.any(repelled) and np.any(~repelled)
    # Every approach entering with w > 0.7 must be repelled near the red-sight halo (r_min > 15),
    # and every approach entering with w <= 0.7 must plunge into the lamp core (r_min < 5)!
    assert np.all(pass_rmin[repelled] > 15.0), (
        f"False 'repelled' pass plunged into core: r_min={pass_rmin[repelled]}, w={pass_w[repelled]}"
    )
    assert np.all(pass_rmin[~repelled] < 5.0), (
        f"False 'plunged' pass was actually repelled: r_min={pass_rmin[~repelled]}, w={pass_w[~repelled]}"
    )


def test_cli_auto_series_routing_and_filtered_all_series(tmp_path: Path) -> None:
    """Verify CLI auto-routes phase-only experiments and filters --series all strictly to requested items."""
    # 1. Omitting --series with a phase-portraits-only ID (v2a_fear) auto-routes to phase-portraits
    auto_out = dynamics_cli.main(
        ["poster", "--experiment", "v2a_fear", "--dpi", "44", "--out-dir", str(tmp_path / "auto")]
    )
    assert [p.name for p in auto_out] == ["v2a_fear_phase.png"]

    # 2. --series all --experiment v2a_fear renders ONLY v2a_fear_phase.png (no unrelated tripanels/atlas)
    only_pp = dynamics_cli.main(
        ["poster", "--series", "all", "--experiment", "v2a_fear", "--dpi", "44", "--out-dir", str(tmp_path / "all_pp")]
    )
    assert [p.name for p in only_pp] == ["v2a_fear_phase.png"]

    # 3. --series all --experiment v9_shy renders ONLY v9_shy_tripanel.png
    only_tri = dynamics_cli.main(
        ["poster", "--series", "all", "--experiment", "v9_shy", "--dpi", "44", "--out-dir", str(tmp_path / "all_tri")]
    )
    assert [p.name for p in only_tri] == ["v9_shy_tripanel.png"]

    # 4. --series all --experiment v4a_orbiter renders BOTH series containing v4a_orbiter and zero atlas plates
    both_v4 = dynamics_cli.main(
        ["poster", "--series", "all", "--experiment", "v4a_orbiter", "--dpi", "44", "--out-dir", str(tmp_path / "all_v4")]
    )
    assert {p.name for p in both_v4} == {"v4a_orbiter_tripanel.png", "v4a_orbiter_phase.png"}

    # 5. --series all --act 1 renders ONLY act1_polar_phase_portraits.png
    only_act = dynamics_cli.main(
        ["poster", "--series", "all", "--act", "1", "--dpi", "44", "--out-dir", str(tmp_path / "all_act")]
    )
    assert [p.name for p in only_act] == ["act1_polar_phase_portraits.png"]

    # 6. animate --series all --scene daydreamer renders ONLY daydreamer.gif
    only_sc = dynamics_cli.main(
        [
            "animate",
            "--series",
            "all",
            "--scene",
            "daydreamer",
            "--fps",
            "6",
            "--seconds",
            "0.3",
            "--dpi",
            "40",
            "--formats",
            "gif",
            "--out-dir",
            str(tmp_path / "all_sc"),
        ]
    )
    assert [p.name for p in only_sc] == ["daydreamer.gif"]

    # 7. Invalid --formats is rejected via parser.error (SystemExit)
    with pytest.raises(SystemExit):
        dynamics_cli.main(["animate", "--experiment", "v5_moth", "--formats", "invalid_fmt", "--out-dir", str(tmp_path)])


def test_partial_skip_existing_preserves_gif_when_mp4_missing(tmp_path: Path) -> None:
    """When .gif already exists and .mp4 is missing, --skip-existing must NOT overwrite .gif."""
    has_ffmpeg = shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists()
    fmts = "gif,mp4" if has_ffmpeg else "gif"

    # Tri-Panel animation: pre-populate sentinel .gif while .mp4 is absent
    tri_dir = tmp_path / "tri_partial"
    tri_dir.mkdir(parents=True)
    sentinel_tri_gif = tri_dir / "v5_moth.gif"
    sentinel_bytes = b"GIF89a-SENTINEL-TRIPANEL"
    sentinel_tri_gif.write_bytes(sentinel_bytes)

    written_tri = dynamics_cli.main(
        [
            "animate",
            "--series",
            "tripanels",
            "--experiment",
            "v5_moth",
            "--fps",
            "6",
            "--seconds",
            "0.3",
            "--dpi",
            "40",
            "--formats",
            fmts,
            "--skip-existing",
            "--out-dir",
            str(tri_dir),
        ]
    )
    assert sentinel_tri_gif.read_bytes() == sentinel_bytes, "Existing tripanel .gif was overwritten!"
    assert sentinel_tri_gif in written_tri
    if has_ffmpeg:
        assert (tri_dir / "v5_moth.mp4").is_file() and (tri_dir / "v5_moth.mp4").stat().st_size > 1_000

    # Phase-Atlas animation: pre-populate sentinel .gif while .mp4 is absent
    atlas_dir = tmp_path / "atlas_partial"
    atlas_dir.mkdir(parents=True)
    sentinel_atlas_gif = atlas_dir / "daydreamer.gif"
    sentinel_atlas_bytes = b"GIF89a-SENTINEL-ATLAS"
    sentinel_atlas_gif.write_bytes(sentinel_atlas_bytes)

    written_atlas = dynamics_cli.main(
        [
            "animate",
            "--series",
            "atlas",
            "--scene",
            "daydreamer",
            "--fps",
            "6",
            "--seconds",
            "0.3",
            "--dpi",
            "40",
            "--formats",
            fmts,
            "--skip-existing",
            "--out-dir",
            str(atlas_dir),
        ]
    )
    assert sentinel_atlas_gif.read_bytes() == sentinel_atlas_bytes, "Existing atlas .gif was overwritten!"
    assert sentinel_atlas_gif in written_atlas
    if has_ffmpeg:
        assert (atlas_dir / "daydreamer.mp4").is_file() and (atlas_dir / "daydreamer.mp4").stat().st_size > 1_000
