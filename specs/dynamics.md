# Complex Dynamics & Shannon Connectome Studio

## Summary

Every Braitenberg vehicle is a continuous- or hybrid-time dynamical system coupling a physical pose $(x, y, \theta) \in \mathbb{R}^2 \times S^1$ to an internal nervous state. The `braitenberg.dynamics` subpackage characterizes Vehicles 4a through 12 using the geometric methods of nonlinear dynamics (after Steven Strogatz's *Nonlinear Dynamics and Chaos* and Lagrangian coherent structure theory) united with Claude Shannon's lossless ensemble information theory, rendering 4K Tri-Panel static plates, multi-aspect 60 FPS animations, and showcase galleries without importing `pygame`.

| Module | Role |
|---|---|
| `braitenberg/dynamics/shannon.py` | **Axiom 1 & Shannon Engine:** `ExperimentSpec` registry, `validate_single_experiment_axiom`, homogeneous species filtering, exact Joint Codebook distributions, Watanabe Total Correlation $TC$, wire Mutual Information & Transfer Entropy, 1D empirical density barcodes, and sorted $N \times M$ bit-rasters. |
| `braitenberg/dynamics/phase_plane.py` | **Cylindrical Phase Space:** $SO(2)$ symmetry-reduced polar phase plane $(r, \psi)$, nullclines, fixed-point classification, limit cycles, and parameter bifurcation sweeps. |
| `braitenberg/dynamics/lagrangian.py` | **Lagrangian Chaos:** Vectorized multi-source swarm integration, forward Finite-Time Lyapunov Exponent ($\sigma_T$) ridges, capture basins, and Poincaré sections. |
| `braitenberg/dynamics/neuro.py` | **Hybrid Memory & Criticality:** Hysteresis folds (Vehicle 5 Moth), slow–fast relaxation oscillations (Vehicle 5 Forager), synaptic bifurcation across $w_c = 0.7$ (Vehicles 7 & 11), and caudate criticality / Kuramoto ring order parameter $Z(t)$ (Vehicle 12). |
| `braitenberg/dynamics/themes.py` | **Dual Aesthetic & Codeword Engine:** `cosmic` bioluminescent dark vs. `textbook` Strogatz print plate, high-contrast Joint Codeword palettes, entropy/barcode colormaps, and multi-aspect geometry (`16:9`, `1:1`, `9:16`). |
| `braitenberg/dynamics/diagram_mpl.py` | **3-Tier Nervous System Panel:** Top-down Braitenberg schematic (`Hero #0` core + Joint Codebook Arc-Rings + marginal entropy halos + MI/TE wire glow + collision-free labels), 1D empirical density barcodes, and sorted $N \times M$ microstate bit-raster. |
| `braitenberg/dynamics/studio.py` | **Tri-Panel Studio:** Synchronized `[Arena | Phase Portrait | 3-Tier Nervous System]` 4K posters, 60 FPS animations, and standardized `showcase` gallery builder. |
| `braitenberg/dynamics/__main__.py` | Command-line entry point `python -m braitenberg.dynamics`. |

---

## Axiom 1 — Strict Single-Experiment Isolation

Every poster and animation characterizes **strictly one isolated experiment** (`ExperimentSpec`) and **one homogeneous focal vehicle species** (`validate_single_experiment_axiom`), never mixing heterogeneous vehicles across experiments:

1. **`v4a_orbiter`** (`scenarios/dynamics-orbiter-swarm.yaml`): Vehicle 4a *Orbiter* swarm in 3-lamp forward FTLE ($\sigma_T$) chaos + cylindrical $(r, \psi)$ orbital attractors + sensorimotor channel mutual information.
2. **`v5_moth`** (`scenarios/dynamics-moth-swarm.yaml`): Vehicle 5 *Moth* swarm (9 threshold units) — irreversible self-excitatory `pain -> burned` fold catastrophe in $(r, \dot{r})$ + seeking/burned codebook split.
3. **`v5_forager`** (`scenarios/dynamics-forager-swarm.yaml`): Vehicle 5 *Forager* swarm — slow–fast $(r, E)$ relaxation limit cycle across Schmitt-trigger thresholds $E_{\text{low}}=30, E_{\text{high}}=90$.
4. **`v7_wary_moth`** (`scenarios/dynamics-wary-moth-swarm.yaml`): Vehicle 7 *Wary Moth* swarm — Hebbian Mnemotrix wire `sees_red -> danger` splitting bimodally across the $w_c = 0.7$ active gate in $(r_{\text{red}}, w_{\text{red}})$.
5. **`v9_shy`** (`scenarios/dynamics-shy-swarm.yaml`): Vehicle 9 *Shy* swarm — $4 \times 24$ `retina` camera + $4 \times 4$ `odd_left`/`odd_right` mirror-symmetry arrays + 11 reflex units (`front_ahead`, `symmetric`, `turn_away`, `run`, `turn_back`, `watch`, `fleeing`, …) in target bearing space $(r_{\text{target}}, \psi)$, filtering environmental `Chaser`/`Walker` stimulus vehicles strictly out of the focal ensemble.
6. **`v12_daydreamer`** (`scenarios/dynamics-daydreamer-swarm.yaml`): Vehicle 12 *Daydreamer* swarm — 8-idea cortical ring + homeostatic `caudate` threshold controller + complex Kuramoto order parameter $Z(t) = Re^{i\Phi}$ in dark unwrapped space.

---

## Mathematical & Information-Theoretic Model

### 1. Why Sample Means Fail on Neural Ensembles (After Claude Shannon)

Averaging binary or continuous unit activations $a_k^{(i)} \in [0, 1]$ across an $N$-vehicle swarm into a scalar mean $\bar{a}_k = \frac{1}{N}\sum_{i=1}^N a_k^{(i)}$ destroys ensemble structure in three ways:

1. **Marginal bimodal collapse:** A swarm split 50/50 between `burned = 0` and `burned = 1` ($H = 1.0\text{ bit}$) collapses to $\bar{a} = 0.5$, indistinguishable from uniform half-activation.
2. **Synergy & mutual-exclusivity erasure:** Anti-correlated steering states `[1, 0]` / `[0, 1]` (`seek_left` vs. `seek_right`) and co-firing states `[1, 1]` / `[0, 0]` share identical marginal means $(0.5, 0.5)$ while differing by $1.0\text{ bit}$ of Watanabe **Total Correlation**:
   $$TC(\mathbf{A}) = \sum_{k=1}^M H(A_k) - H(\mathbf{W}) \ge 0, \qquad H(\mathbf{W}) = -\sum_{c} p_c \log_2 p_c.$$
3. **Zero-bit bias illusion:** A constant tonic unit (`awake = 1` across all $N$ vehicles) exhibits maximum mean $1.0$ despite carrying strictly $H(\text{awake}) = 0.0\text{ bits}$ of information.

### 2. Three-Tier Hero + Lossless Shannon Nervous System (`diagram_mpl.py`)

Each Tri-Panel figure unites single-vehicle causal legibility (**Cyan-Ringed `Hero #0`**) with lossless ensemble coding across `--neural-view composite|codebook|entropy|raster`:

- **Tier A — Braitenberg Body Schematic:**
  - **Inner core & wire pulse:** `Hero #0`'s instantaneous part level and active signal transmission, with unit thresholds printed inside.
  - **Joint Codebook Arc-Ring:** Around each part $k$, segmented `Wedge` arcs of angular span $360^\circ \cdot p_c$ for every joint codeword $c$ in which $w_{c,k} = 1$, colored by **Codeword Palette Hue $c$** ($\sum_{c: w_{c,k}=1} p_c = P(A_k = 1)$).
  - **Marginal Entropy Halo:** Outer aura scaled by $H(A_k) \in [0, 1]\text{ bit}$ (`0` halo on constant units).
  - **Information-Weighted Wires:** Bundled wires (`braitenberg.diagram.bundles`) with glow width scaled by $\max\bigl(I(S_t; D_t),\, T_{S\to D}\bigr)$ in bits.
- **Tier B — Shannon Telemetry & 1D Empirical Density Barcodes:**
  - Displays live `H(W)`, `Synergy TC`, and `I(Phase; W)` in bits above 1D empirical histograms over `[0, 1]` (marking the `w_c = 0.7` learning gate and `Hero #0`'s live cursor).
- **Tier C — Sorted $N \times M$ Swarm Microstate Bit-Raster:**
  - Exact binary matrix sorted by `(codeword_id, phase)`, colored by Codeword Hue $c$ with `Hero #0`'s row marked in cyan.

### 3. Geometric Phase Planes & Lagrangian Structures

- **$SO(2)$ Polar Reduction:** Around one isotropic lamp, $\dot{r} = -v(r,\psi)\cos\psi$ and $\dot{\psi} = \frac{v(r,\psi)}{r}\sin\psi - \omega(r,\psi)$.
- **Forward FTLE Ridges:** From the unwrapped spatial flow map $\Phi_T(x_0, y_0)$ and Cauchy–Green tensor $\Delta = (D\Phi_T)^\top(D\Phi_T)$, $\sigma_T = \frac{1}{|T|}\ln\sqrt{\lambda_{\max}(\Delta)}$ reveals repelling separatrices.
- **Cortical Ring Order Parameter:** $Z(t) = R(t)e^{i\Phi(t)} = \frac{1}{8}\sum_{k=0}^7 a_k(t)e^{i2\pi k / 8}$.

---

## Command-Line Interface

```bash
# Static 4K Tri-Panel posters (1 isolated experiment per file)
python -m braitenberg.dynamics poster \
    --experiment v4a_orbiter|v5_moth|v5_forager|v7_wary_moth|v9_shy|v12_daydreamer|all \
    [--neural-view composite|codebook|entropy|raster] \
    [--theme cosmic|textbook] \
    [--aspect 16:9|1:1|9:16] \
    --out-dir <dir> [--dpi 240] [--format png|svg]

# Synchronized Tri-Panel animations (GIF and/or 60 FPS H.264 MP4)
python -m braitenberg.dynamics animate \
    --experiment v4a_orbiter|v5_moth|v5_forager|v7_wary_moth|v9_shy|v12_daydreamer|all \
    [--neural-view composite|codebook|entropy|raster] \
    [--theme cosmic|textbook] \
    [--aspect 16:9|1:1|9:16] \
    [--fps 60] [--seconds 7] \
    --out-dir <dir> [--formats gif,mp4]

# Full standardized 1-Experiment-Per-File gallery
python -m braitenberg.dynamics showcase --out-dir experiments/strogatz-dynamics/media
```

### Standardized Showcase Outputs

- **6 Cosmic 4K (`3840×2160`) Tri-Panel Posters:** `{v4a_orbiter,v5_moth,v5_forager,v7_wary_moth,v9_shy,v12_daydreamer}_tripanel.png`
- **6 Strogatz Textbook 4K Tri-Panel Posters:** `{v4a_orbiter,v5_moth,v5_forager,v7_wary_moth,v9_shy,v12_daydreamer}_tripanel_textbook.png`
- **6 GitHub README Animated GIFs (`16:9`):** `{v4a_orbiter,v5_moth,v5_forager,v7_wary_moth,v9_shy,v12_daydreamer}.gif`
- **18 Local 60 FPS H.264 MP4 Social Cuts:** `mp4/{v4a_orbiter,v5_moth,v5_forager,v7_wary_moth,v9_shy,v12_daydreamer}_{16x9,1x1,9x16}.mp4`

---

## Acceptance Criteria

1. **Architecture boundary:** Importing and running `braitenberg.dynamics` (`shannon`, `phase_plane`, `lagrangian`, `neuro`, `themes`, `diagram_mpl`, `studio`, `__main__`) never imports `pygame` and never opens a GUI window.
2. **Axiom 1 single-experiment purity:** Every poster and animation validates `validate_single_experiment_axiom` against strictly one `ExperimentSpec` and homogeneous focal species (excluding environmental `Chaser`/`Walker` targets on `v9_shy`).
3. **Collision-free 3-tier neural schematic:** `diagram_mpl.solve_part_labels` and `ShannonNeuralPanelArtist` render all 6 canonical vehicles (including 16-part `v9_shy` and 18-part `v5_forager` / `v7_wary_moth`) with zero overlapping part labels and support all 4 `--neural-view` modes (`composite`, `codebook`, `entropy`, `raster`).
4. **Dual-theme & multi-aspect Tri-Panel geometry:** `poster` renders valid PNG and SVG plates across both `cosmic` and `textbook` themes and all three aspect ratios (`16:9` 3-column, `1:1` 2×2 grid, `9:16` 3-row stack) at exact expected aspect proportions.
5. **Headless animation & standardized showcase:** `animate` exports valid multi-frame GIFs and H.264 `yuv420p` MP4s, and `showcase` populates all 36 canonical 1-experiment-per-file gallery assets.
