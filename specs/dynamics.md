# Complex Dynamics & Visual Studio

## Summary

Every Braitenberg vehicle is a continuous- or hybrid-time dynamical system coupling a physical pose $(x, y, \theta) \in \mathbb{R}^2 \times S^1$ to an internal nervous state. The `braitenberg.dynamics` subpackage characterizes Vehicles 1 through 12 using the geometric methods of nonlinear dynamics (after Steven Strogatz's *Nonlinear Dynamics and Chaos* and Lagrangian coherent structure theory) and renders static plates, multi-aspect animations, and showcase galleries without importing `pygame`.

| Module | Role |
|---|---|
| `braitenberg/dynamics/phase_plane.py` | **Act I:** $SO(2)$ symmetry-reduced cylindrical phase plane $(r, \psi)$, nullclines, fixed-point classification, limit cycles, and parameter bifurcation sweeps. |
| `braitenberg/dynamics/lagrangian.py` | **Act II:** Vectorized multi-source swarm integration, forward Finite-Time Lyapunov Exponent ($\sigma_T$) ridges, capture basins, and Poincaré sections. |
| `braitenberg/dynamics/neuro.py` | **Acts III & IV:** Hybrid hysteresis folds (Vehicle 5 Moth), slow-fast relaxation oscillations (Vehicle 5 Forager), synaptic bifurcation across $w_c = 0.7$ (Vehicles 7 & 11), and caudate criticality / Kuramoto ring order parameter $Z(t)$ (Vehicle 12). |
| `braitenberg/dynamics/themes.py` | Dual aesthetic engine (`cosmic` bioluminescent dark vs. `textbook` Strogatz print plate) and multi-aspect layout geometry (`16:9`, `1:1`, `9:16`). |
| `braitenberg/dynamics/studio.py` | 4-Act posters, Master Synthesis Atlas, 60 FPS animations (`orbiter`, `ftle-swarm`, `plasticity`, `daydreamer`), and `showcase` gallery builder. |
| `braitenberg/dynamics/__main__.py` | Command-line entry point `python -m braitenberg.dynamics`. |

---

## Mathematical Model

### Act I — $SO(2)$ Symmetry Reduction Around One Source

Around a single isotropic stimulus source at $(s_x, s_y)$, rotational and translational invariance reduce a memoryless vehicle's 3D state $(x, y, \theta)$ to a 2D autonomous flow on the cylinder $(r, \psi) \in (0, \infty) \times [-\pi, \pi]$:

$$r = \hypot(x - s_x,\, y - s_y), \qquad \psi = \operatorname{wrap\_pi}\!\bigl(\operatorname{atan2}(s_y - y,\, s_x - x) - \theta\bigr)$$

where $\psi = 0$ faces the source directly, $\psi > 0$ places the source on the left, and $\psi < 0$ places the source on the right. Differentiating along noiseless differential-drive kinematics $(\dot{x} = v\cos\theta,\; \dot{y} = v\sin\theta,\; \dot{\theta} = \omega)$ yields the exact polar equations:

$$\dot{r} = -v(r, \psi)\cos\psi, \qquad \dot{\psi} = \frac{v(r, \psi)}{r}\sin\psi - \omega(r, \psi).$$

- **Invariant manifolds:** By reflection symmetry of bilateral sensors, $\omega(r, 0) = \omega(r, \pm\pi) = 0$, so the rays $\psi = 0$ (head-on approach) and $\psi = \pm\pi$ (straight retreat) are invariant lines.
- **Equilibria and stability:** Fixed points satisfy $\dot{r} = 0$ and $\dot{\psi} = 0$ (either $v = 0$, as in Vehicle 3a *Love* resting at the lamp, or $\psi = \pm\pi/2$ balanced against $v/r = \pm\omega$, or off-axis equilibria). Linearization of the $2 \times 2$ Jacobian classifies stable/unstable nodes, spirals, centers, and saddles.
- **Supercritical orbital attractor:** Vehicle 4a (*Orbiter*), with crossed excitatory wiring and peaked sensor response (`peak`), repels near the lamp ($r < r^*$) and attracts outside ($r > r^*$), converging onto a stable closed orbit around $r^* \approx 18.4$ in `worlds/one-light.yaml`.

### Act II — Symmetry Breaking & Lagrangian Coherent Structures

Adding a second or third lamp breaks $SO(2)$ rotational invariance, lifting the dynamics to an irreducible 3D flow on $\mathbb{R}^2 \times S^1$. Hidden separatrices between competing attractors are extracted from the spatial flow map $\Phi_T(x_0, y_0) = (u_x(T), u_y(T))$ integrated in **unwrapped** arena coordinates so periodic boundaries (`edges: wrap`) never introduce spurious derivative seams.

From the right Cauchy–Green deformation tensor $\Delta = (D\Phi_T)^\top (D\Phi_T)$, the forward Finite-Time Lyapunov Exponent is

$$\sigma_T(x_0, y_0) = \frac{1}{|T|} \ln \sqrt{\max\bigl(\lambda_{\max}(\Delta),\, 10^{-12}\bigr)}.$$

Ridges of $\sigma_T$ form **repelling Lagrangian Coherent Structures (LCS)** along which ensembles of trajectories shear apart into distinct lamp basins.

### Act III — Memory as Phase-Space Dimensionality Expansion

Internal threshold latches, energy stores, and plastic wires expand the phase space beyond $(x, y, \theta)$:

1. **Irreversible fold/latch hysteresis (Vehicle 5 *Moth*):** Self-excitatory unit `burned` ($b \in \{0, 1\}$) switches irreversibly from $0 \to 1$ once `pain` crosses threshold $18$, jumping the vehicle from the lower attraction sheet onto the upper repulsion sheet at the same physical radius $r$.
2. **Slow–fast relaxation oscillation (Vehicle 5 *Forager*):** Internal energy $E \in [0, 100]$ evolves slowly while the `full`/`hungry` Schmitt-trigger latch ($E_{\text{high}} = 90$, $E_{\text{low}} = 30$) alternates fast phototaxis with exploratory zigzagging around a closed $(r, E)$ hysteresis loop.
3. **Piecewise-smooth synaptic bifurcation (Vehicles 7 & 11):** Mnemotrix and Ergotrix weights $w(t) \in [0, 1]$ gate signal transmission strictly above `ACTIVE_STRENGTH = 0.7` ($w_c = 0.7$). Crossing $w(t) = w_c$ triggers an abrupt structural bifurcation in vector-field topology (attraction $\to$ preemptive avoidance in *Wary Moth*, stop-at-yellow anticipation in *Cautious Driver*).

### Act IV — Homeostatic Criticality & Neural Ring Order Parameter

Vehicle 12 (*Daydreamer*) couples eight threshold units $a_k(t) \in \{0, 1\}$ ($k = 0, \dots, 7$) around an explosive $k \to (k+1, k+3) \bmod 8$ ring regulated by the accumulating `caudate` threshold control:

$$\dot{\theta}_{\text{off}} = \text{rate}\cdot\bigl(A(t) - \text{target}\bigr), \qquad A(t) = \sum_{k=0}^7 a_k(t).$$

Macroscopic wave coherence around the cortical ring is quantified by the complex Kuramoto-style order parameter

$$Z(t) = R(t)e^{i\Phi(t)} = \frac{1}{8}\sum_{k=0}^7 a_k(t)\,e^{i 2\pi k / 8},$$

separating three dynamical regimes across control parameters: **quiescent extinction** ($\langle A \rangle \approx 0$), **critical trains of thought** ($0 < \langle A \rangle \lesssim 2.5$, traveling waves with non-zero $R(t)$ and rotating phase $\Phi(t)$), and **epileptic flood** ($\langle A \rangle \to 8$, $R(t) \to 0$ by symmetry cancellation).

---

## Command-Line Interface

```bash
# Static posters (Act 1..4, Master Synthesis Atlas, or all)
python -m braitenberg.dynamics poster \
    --act 1|2|3|4|atlas|all \
    --theme cosmic|textbook \
    --aspect 16:9|1:1|9:16 \
    --out-dir <dir> [--dpi 300] [--format png|svg]

# Synchronized animations (GIF and/or 60 FPS H.264 MP4)
python -m braitenberg.dynamics animate \
    --scene orbiter|ftle-swarm|plasticity|daydreamer|all \
    --theme cosmic|textbook \
    --aspect 16:9|1:1|9:16 \
    --fps 60 --seconds 8 \
    --out-dir <dir> [--formats gif,mp4]

# Full curated gallery (cosmic + textbook posters, README GIFs, multi-aspect MP4s)
python -m braitenberg.dynamics showcase --out-dir experiments/strogatz-dynamics/media
```

### Visual Themes & Framing

- **`cosmic` (default):** Abyssal space-indigo `#060818` canvas, `#0a0f29` panels, 3-pass neon glow strokes (`plot_glow_line`, `plot_glow_scatter`, `plot_comet_trails`), and abyssal-to-cyan-to-gold FTLE colormaps.
- **`textbook`:** Warm archival paper `#fcfbf7`, crisp charcoal `#18181b` axes and streamlines, editorial serif math typography, and canonical Strogatz equilibrium markers (filled disc = stable, open circle = unstable, cross-ticked open circle = saddle).
- **Framing (`--aspect`):** Every poster and animation adapts its grid geometry cleanly across `16:9` widescreen, `1:1` square, and `9:16` vertical formats without overlapping text or clipped margins.

---

## Acceptance Criteria

1. **Architecture boundary:** Importing and running `braitenberg.dynamics` (`phase_plane`, `lagrangian`, `neuro`, `themes`, `studio`, `__main__`) never imports `pygame` and never opens a GUI window.
2. **Polar phase-plane fidelity:** `polar_field` $(\dot{r}, \dot{\psi})$ agrees with deterministic `Vehicle.step` kinematics (`noise_std = 0`), and `find_fixed_points_and_cycles` detects Vehicle 4a's attracting limit cycle in `worlds/one-light.yaml`.
3. **Seam-free FTLE & ensembles:** `compute_ftle_and_basins` and `simulate_swarm_ensemble` track unwrapped displacements so periodic arena boundaries (`edges: wrap`) produce zero artificial seam spikes or wrap-across trail artifacts.
4. **Hybrid memory & criticality:** `characterize_hysteresis_and_learning` captures Vehicle 5's hysteresis jump, Forager's $(r, E)$ cycle, and Vehicle 7/11's $w_c = 0.7$ crossing; `analyze_daydreamer_criticality` computes bounded $|Z(t)| \in [0, 1]$ and distinguishes silent, critical, and flooded phases.
5. **Dual-theme & multi-aspect rendering:** `poster` renders non-empty PNG and SVG outputs across both `cosmic` and `textbook` themes and all three aspect ratios (`16:9`, `1:1`, `9:16`) at exact expected aspect proportions.
6. **Headless animation export:** `animate` exports valid multi-frame GIFs (and H.264 `yuv420p` MP4s when `ffmpeg` is present) for `orbiter`, `ftle-swarm`, `plasticity`, and `daydreamer`, and `showcase` populates the standard gallery filenames.
