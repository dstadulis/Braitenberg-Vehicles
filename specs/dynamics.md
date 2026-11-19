# Complex Dynamics, Deep Phase Portraits & Shannon Connectome Studio

## Summary

Every Braitenberg vehicle is a continuous- or hybrid-time dynamical system coupling a physical pose $(x, y, \theta) \in \mathbb{R}^2 \times S^1$ to an internal nervous state. The `braitenberg.dynamics` subpackage characterizes Vehicles 2a through 12 using the geometric methods of nonlinear dynamics (after Steven Strogatz's *Nonlinear Dynamics and Chaos* and Lagrangian coherent structure theory) united with Claude Shannon's lossless ensemble information theory, rendering three coordinated visual series without importing `pygame`.

| Module | Role |
|---|---|
| `braitenberg/dynamics/shannon.py` | **Axiom 1 & Shannon Engine:** `ExperimentSpec` (`EXPERIMENTS` & `PHASE_EXPERIMENTS`), `validate_single_experiment_axiom`, homogeneous species filtering, exact Joint Codebook distributions, Watanabe Total Correlation $TC$, wire Mutual Information & Transfer Entropy, 1D empirical density barcodes, and sorted $N \times M$ bit-rasters. |
| `braitenberg/dynamics/phase_plane.py` | **Cylindrical Phase Space:** $SO(2)$ symmetry-reduced polar phase plane $(r, \psi)$, nullclines, fixed-point classification, limit cycles, and parameter bifurcation sweeps. |
| `braitenberg/dynamics/lagrangian.py` | **Lagrangian Chaos:** Vectorized multi-source swarm integration, forward Finite-Time Lyapunov Exponent ($\sigma_T$) ridges, capture basins, and Poincaré sections. |
| `braitenberg/dynamics/neuro.py` | **Hybrid Memory & Criticality:** Hysteresis folds (Vehicle 5 Moth), slow–fast relaxation oscillations (Vehicle 5 Forager), synaptic bifurcation across $w_c = 0.7$ (Vehicles 7 & 11), and caudate criticality / Kuramoto ring order parameter $Z(t)$ (Vehicle 12). |
| `braitenberg/dynamics/themes.py` | **Dual Aesthetic & Codeword Engine:** `cosmic` bioluminescent dark vs. `textbook` Strogatz print plate, high-contrast Joint Codeword palettes, entropy/barcode colormaps, and multi-aspect geometry (`16:9`, `1:1`, `9:16`). |
| `braitenberg/dynamics/diagram_mpl.py` | **3-Tier Nervous System Panel:** Top-down Braitenberg schematic (`Hero #0` core + Joint Codebook Arc-Rings + marginal entropy halos + MI/TE wire glow + collision-free labels), 1D empirical density barcodes, and sorted $N \times M$ microstate bit-raster. |
| `braitenberg/dynamics/phase_portraits.py` | **Series I — Pure Axiom-1 Deep Phase Portraits:** 8 strictly isolated 1-experiment-per-file 4-panel 4K Strogatz phase-portrait plates (`v2a_fear`, `v3a_love`, `v4a_orbiter`, `v5_moth`, `v5_forager`, `v7_wary_moth`, `v11_cautious_driver`, `v12_daydreamer`). |
| `braitenberg/dynamics/studio.py` | **Series II — Pure Axiom-1 Shannon Tri-Panel Studio & Unified Dispatcher:** Synchronized `[Arena | Phase Portrait | 3-Tier Nervous System]` 4K posters, 60 FPS animations, and unified 3-series `showcase` gallery builder. |
| `braitenberg/dynamics/phase_atlas.py` | **Series III — Comparative Synthesis Phase-Space Atlas:** Classic 4-Act Strogatz synthesis plates (`strogatz_atlas`, `act1..4`) and 2-panel phase animations (`orbiter`, `ftle_swarm`, `plasticity`, `daydreamer`), explicitly tagged `series="phase_atlas"`. |
| `braitenberg/dynamics/__main__.py` | Unified command-line entry point `python -m braitenberg.dynamics`. |

---

## Axiom 1 & Three-Series Architecture

### Series I — Pure Axiom-1 Deep Phase-Portrait Plates (`phase_portraits.py`)

Every plate in `media/phase_portraits/` calls `validate_single_experiment_axiom` at entry and on every simulated `Scenario`, isolating **strictly one homogeneous vehicle species** across 4 analytical panels:

1. **`v2a_fear`** (`vehicles/vehicle2a-fear.yaml`): (A) Cylindrical $(r, \psi)$ repulsive deflection, (B) physical arena scattering fan, (C) radial drift $\dot{r}(r,\psi)$ & repulsive torque $-4\omega(r,\psi)$, (D) Rutherford deflection curve $b \mapsto \Delta\theta(b)$.
2. **`v3a_love`** (`vehicles/vehicle3a-love.yaml`): (A) Cylindrical $(r, \psi)$ arrest attractor $(\psi=0, v\to 0)$ & off-axis saddles, (B) arena capture ring vs. saddle escape, (C) head-on speed deceleration $v(r)$, (D) equilibrium arrest distance $r^*(I_0)$ sweep.
3. **`v4a_orbiter`** (`vehicles/vehicle4a-orbiter.yaml`): (A) Single-lamp $(r, \psi)$ supercritical limit cycle & spiral sinks, (B) 1D `peak.at` orbital bifurcation & escape catastrophe, (C) 3-lamp forward FTLE ridges $\sigma_T$ & trefoil basins, (D) 3-lamp Poincaré section $\Sigma: x = L_x/2$.
4. **`v5_moth`** (`vehicles/vehicle5-moth.yaml`): (A) Irreversible burn-latch fold in $(r, \dot{r})$, (B) pre-burn (`burned=0`) attraction $(r, \psi)$ portrait, (C) post-burn (`burned=1`) repulsion $(r, \psi)$ portrait (topological flip of $\psi=0$), (D) two-lamp arena trajectory.
5. **`v5_forager`** (`vehicles/vehicle5-forager.yaml`): (A) Slow–fast $(r, E)$ relaxation limit cycle across $[30, 90]$, (B) arena rosette foraging track, (C) $E(t), r(t)$ & `full` hysteresis bouts, (D) foraging duty-cycle breakdown vs. $I_0$.
6. **`v7_wary_moth`** (`vehicles/vehicle7-wary-moth.yaml`): (A) Hybrid $(r_{\text{red}}, w_{\text{red}})$ trajectory across $w_c = 0.7$, (B) Mnemotrix acquisition & cooled-lamp extinction $w(t)$, (C) selective avoidance arena trajectories, (D) minimum approach distance $r_{\min}$ per pass.
7. **`v11_cautious_driver`** (`vehicles/vehicle11-cautious-driver.yaml`): (A) Ergotrix sequence staircases crossing $w_c = 0.7$ vs. $0$ reverse wires, (B) longitudinal braking portrait $(d_{\text{line}}, v)$ on yellow, (C) G/Y/R cycle alignment & anticipatory `expects_red`, (D) $3 \times 3$ ($6$ Ergotrix + $3$ innate) asymptotic weight matrix.
8. **`v12_daydreamer`** (`vehicles/vehicle12-daydreamer.yaml`): (A) Caudate homeostatic criticality sweep across $\tau_{\text{target}}$, (B) cortical mandala & Kuramoto order parameter $Z(t) = Re^{i\Phi}$, (C) 8-idea cascade & $\theta_{\text{eff}}(t)$ breathing, (D) dark-arena thought loops across seeds.

### Series II — Pure Axiom-1 Shannon × Strogatz Tri-Panels (`studio.py`)

Unites `[Arena | Ensemble Phase Space | 3-Tier Hero #0 + Lossless Shannon Nervous System]` for the 6 canonical swarm experiments (`v4a_orbiter`, `v5_moth`, `v5_forager`, `v7_wary_moth`, `v9_shy`, `v12_daydreamer`), preserving Joint Codebooks, Watanabe Synergy $TC$, wire MI/TE, 1D empirical density barcodes, and sorted $N \times M$ bit-rasters.

### Series III — Comparative Synthesis Phase-Space Atlas (`phase_atlas.py`)

Preserves the classic comparative Strogatz 4-Act overview (`strogatz_atlas`, `act1..4`) and 2-panel animations (`orbiter`, `ftle_swarm`, `plasticity`, `daydreamer`), explicitly tagged `series="phase_atlas"` (`comparative_synthesis=True` on multi-vehicle synthesis plates, which `validate_single_experiment_axiom` strictly rejects if passed as a single-experiment spec).

---

## Command-Line Interface

```bash
# Series I: Pure Axiom-1 4-Panel Deep Phase Portraits (8 isolated experiments)
python -m braitenberg.dynamics poster --series phase-portraits \
    --experiment v2a_fear|v3a_love|v4a_orbiter|v5_moth|v5_forager|v7_wary_moth|v11_cautious_driver|v12_daydreamer|all \
    [--theme cosmic|textbook] [--aspect 16:9|1:1|9:16] --out-dir <dir>

# Series II: Pure Axiom-1 Shannon × Strogatz Tri-Panel Posters & Animations (6 swarms)
python -m braitenberg.dynamics poster [--series tripanels] \
    --experiment v4a_orbiter|v5_moth|v5_forager|v7_wary_moth|v9_shy|v12_daydreamer|all \
    [--neural-view composite|codebook|entropy|raster] [--theme cosmic|textbook] --out-dir <dir>

python -m braitenberg.dynamics animate [--series tripanels] \
    --experiment v4a_orbiter|v5_moth|v5_forager|v7_wary_moth|v9_shy|v12_daydreamer|all \
    [--fps 60] [--seconds 7] --out-dir <dir> [--formats gif,mp4]

# Series III: Comparative Synthesis Phase-Atlas Posters & Classic Animations
python -m braitenberg.dynamics poster --series atlas --act 1|2|3|4|atlas|all \
    [--theme cosmic|textbook] [--aspect 16:9|1:1|9:16] --out-dir <dir>

python -m braitenberg.dynamics animate --series atlas \
    --scene orbiter|ftle-swarm|plasticity|daydreamer|all --out-dir <dir>

# Full Standardized 3-Series Gallery Tree (supports --skip-existing for incremental builds)
python -m braitenberg.dynamics showcase \
    --out-dir experiments/strogatz-dynamics/media \
    [--series all|tripanels|phase-portraits|atlas] [--skip-existing]
```

### Standardized 3-Series Showcase Tree (`<out-dir>`)

1. **`<out-dir>/` (Series II — Shannon Tri-Panels):** 6 `*_tripanel.png` + 6 `*_tripanel_textbook.png` + 6 `*.gif` + `<out-dir>/mp4/*_{16x9,1x1,9x16}.mp4`.
2. **`<out-dir>/phase_portraits/` (Series I — Pure Deep Phase Portraits):** 8 `*_phase.png` + 8 `*_phase_textbook.png`.
3. **`<out-dir>/phase_atlas/` (Series III — Comparative Phase Atlas):** 5 Cosmic PNGs + 5 `*_textbook.png` (`strogatz_atlas`, `act1..4`) + 4 GIFs (`orbiter`, `ftle_swarm`, `plasticity`, `daydreamer`) + `<out-dir>/mp4/phase_atlas/*_{16x9,1x1,9x16}.mp4`.

---

## Acceptance Criteria

1. **Architecture boundary:** Importing and running `braitenberg.dynamics` (`shannon`, `phase_plane`, `lagrangian`, `neuro`, `themes`, `diagram_mpl`, `phase_portraits`, `phase_atlas`, `studio`, `__main__`) never imports `pygame` and never opens a GUI window.
2. **Axiom 1 single-experiment purity & Atlas separation:** Every single `tripanels` (6 experiments) and `phase-portraits` (8 experiments) render calls `validate_single_experiment_axiom` against strictly one `ExperimentSpec` and homogeneous focal species, while `phase_atlas` items carry `series="phase_atlas"` (`comparative_synthesis=True` on multi-vehicle plates) and are rejected by `validate_single_experiment_axiom`.
3. **Collision-free 3-tier neural schematic:** `diagram_mpl.solve_part_labels` and `ShannonNeuralPanelArtist` render all 6 canonical vehicles with zero overlapping part labels across all 4 `--neural-view` modes.
4. **Dual-theme & multi-aspect geometry across all 3 series:** `poster` renders valid PNG and SVG plates across both `cosmic` and `textbook` themes and all three aspect ratios (`16:9`, `1:1`, `9:16`) for `tripanels`, `phase-portraits`, and `atlas`.
5. **Headless animation & 3-series incremental showcase:** `animate` exports valid multi-frame GIFs and H.264 `yuv420p` MP4s for both `tripanels` and `atlas` scenes, and `showcase` populates the exact 3-series tree while respecting `--skip-existing`.
