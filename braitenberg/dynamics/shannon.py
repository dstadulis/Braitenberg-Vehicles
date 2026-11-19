"""Claude Shannon ensemble connectome & axiomatic single-experiment dynamics engine.

Why sample means fail on neural & behavioral ensembles:
1. Marginal collapse: mapping an empirical distribution p(a_k) to its sample mean E[a_k]
   confounds an uncertain 50/50 bifurcation split (H = 1 bit) with a half-lit analog unit.
2. Synergy & exclusivity erasure: marginal means discard Watanabe's Total Correlation
   TC(A) = sum_k H(A_k) - H(W), making mutually exclusive reflex modes ([1, 0] vs [0, 1],
   e.g. seek_left vs seek_right) indistinguishable from co-activating modes ([1, 1] vs [0, 0]).
3. Zero-bit bias illusion: a constant tonic unit (e.g. Daydreamer's awake = 1 across all N
   vehicles) has sample mean 1.0 yet carries strictly H = 0.0 bits of information.

This module preserves 100% of ensemble structure alongside a deterministic Hero #0 tracer via:
- Strict Axiom 1 enforcement (one isolated scenario & one homogeneous focal vehicle species),
- Exact Joint Neural Codebooks & per-unit Codeword Arc decompositions,
- Marginal / Joint Entropies, Watanabe Total Correlation (Synergy), and Spatial-Neural MI,
- Bundled-wire Channel Mutual Information I(S_t; D_t) & Schreiber Transfer Entropy T_{S->D},
- 1D Empirical Density Barcodes over [0, 1] (preserving bimodal plasticity across w_c = 0.7),
- Lossless sorted N x M swarm microstate bit-rasters.

This module never imports pygame (see AGENTS.md).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from braitenberg.definition import (
    ACTIVE_STRENGTH,
    ENERGY_SENSOR,
    VehicleDefinition,
    element_name,
    load_definition,
)
from braitenberg.diagram import bundles
from braitenberg.dynamics.lagrangian import compute_ftle_and_basins, extract_primary_sources
from braitenberg.dynamics.neuro import compute_ring_order_parameter
from braitenberg.dynamics.phase_plane import (
    find_fixed_points_and_cycles,
    polar_coords,
    polar_field,
    resolve_primary_source,
    wrap_pi,
)
from braitenberg.scenario import Scenario, SimulationSettings, load_scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Vehicle, clamp01

REPO_ROOT = Path(__file__).resolve().parents[2]


class AxiomViolationError(ValueError):
    """Raised when Axiom 1 (strict single-experiment / single-species isolation) is violated."""


@dataclass(frozen=True)
class ExperimentSpec:
    """Canonical specification for a pure single-species dynamical experiment."""

    experiment_id: str
    title: str
    subtitle: str
    scenario_path: str
    focal_vehicle_path: str
    phase_axes: tuple[str, str]


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "v4a_orbiter": ExperimentSpec(
        experiment_id="v4a_orbiter",
        title="Vehicle 4a: Orbiter Swarm",
        subtitle="Three-Lamp FTLE Chaos, Cylindrical (r, ψ) Limit Cycle & Sensorimotor Channel Information",
        scenario_path="scenarios/dynamics-orbiter-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle4a-orbiter.yaml",
        phase_axes=("r", "psi"),
    ),
    "v5_moth": ExperimentSpec(
        experiment_id="v5_moth",
        title="Vehicle 5: Moth Swarm",
        subtitle="Self-Excitatory Latch Catastrophe (pain -> burned) & Seeking/Avoidance Bifurcation",
        scenario_path="scenarios/dynamics-moth-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle5-moth.yaml",
        phase_axes=("r", "dr"),
    ),
    "v5_forager": ExperimentSpec(
        experiment_id="v5_forager",
        title="Vehicle 5: Forager Swarm",
        subtitle="Slow-Fast (r, E) Relaxation Limit Cycle & Hungry/Full Hysteresis Codebook",
        scenario_path="scenarios/dynamics-forager-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle5-forager.yaml",
        phase_axes=("r", "energy"),
    ),
    "v7_wary_moth": ExperimentSpec(
        experiment_id="v7_wary_moth",
        title="Vehicle 7: Wary Moth Swarm",
        subtitle="Hebbian Mnemotrix Plasticity (sees_red -> danger) Crossing the w_c = 0.7 Gate",
        scenario_path="scenarios/dynamics-wary-moth-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle7-wary-moth.yaml",
        phase_axes=("r", "w_red"),
    ),
    "v9_shy": ExperimentSpec(
        experiment_id="v9_shy",
        title="Vehicle 9: Shy Swarm",
        subtitle="Retinal Symmetry Detection (4×24 Camera + 4×4 Odd Arrays) & Multi-Stage Flee Reflex",
        scenario_path="scenarios/dynamics-shy-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle9-shy.yaml",
        phase_axes=("r", "psi"),
    ),
    "v12_daydreamer": ExperimentSpec(
        experiment_id="v12_daydreamer",
        title="Vehicle 12: Daydreamer Swarm",
        subtitle="Homeostatic Caudate Criticality on an 8-Idea Ring & Kuramoto Order Parameter Z(t)",
        scenario_path="scenarios/dynamics-daydreamer-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle12-daydreamer.yaml",
        phase_axes=("Re(Z)", "Im(Z)"),
    ),
}

PHASE_EXPERIMENTS: dict[str, ExperimentSpec] = {
    "v2a_fear": ExperimentSpec(
        experiment_id="v2a_fear",
        title="Vehicle 2a: Fear",
        subtitle="Repulsive Cylindrical (r, ψ) Scattering, Turn-Rate Profile & Rutherford Deflection",
        scenario_path="scenarios/vehicle2-fear-and-aggression.yaml",
        focal_vehicle_path="vehicles/vehicle2a-fear.yaml",
        phase_axes=("r", "psi"),
    ),
    "v3a_love": ExperimentSpec(
        experiment_id="v3a_love",
        title="Vehicle 3a: Love",
        subtitle="Stable Arrest Attractor (ψ = 0, v → 0), Off-Axis Saddles & Equilibrium Distance r*(I₀)",
        scenario_path="scenarios/vehicle3-love-and-explorer.yaml",
        focal_vehicle_path="vehicles/vehicle3a-love.yaml",
        phase_axes=("r", "psi"),
    ),
    "v4a_orbiter": ExperimentSpec(
        experiment_id="v4a_orbiter",
        title="Vehicle 4a: Orbiter",
        subtitle="Supercritical Cylindrical Limit Cycle, Peak Bifurcation, 3-Lamp FTLE Ridges & Poincaré Map",
        scenario_path="scenarios/dynamics-orbiter-swarm.yaml",
        focal_vehicle_path="vehicles/vehicle4a-orbiter.yaml",
        phase_axes=("r", "psi"),
    ),
    "v5_moth": ExperimentSpec(
        experiment_id="v5_moth",
        title="Vehicle 5: Moth",
        subtitle="Irreversible Burn-Latch Fold Catastrophe & Pre/Post-Burn Topological Phase Flip",
        scenario_path="scenarios/vehicle5-moth.yaml",
        focal_vehicle_path="vehicles/vehicle5-moth.yaml",
        phase_axes=("r", "dr"),
    ),
    "v5_forager": ExperimentSpec(
        experiment_id="v5_forager",
        title="Vehicle 5: Forager",
        subtitle="Slow–Fast (r, E) Relaxation Limit Cycle, Arena Rosette & Hysteresis Duty Cycle",
        scenario_path="scenarios/vehicle5-forager.yaml",
        focal_vehicle_path="vehicles/vehicle5-forager.yaml",
        phase_axes=("r", "energy"),
    ),
    "v7_wary_moth": ExperimentSpec(
        experiment_id="v7_wary_moth",
        title="Vehicle 7: Wary Moth",
        subtitle="Hebbian Mnemotrix Bifurcation ($w_c = 0.7$), Hot/Cold Extinction & Approach Envelope",
        scenario_path="scenarios/vehicle7-wary-moth.yaml",
        focal_vehicle_path="vehicles/vehicle7-wary-moth.yaml",
        phase_axes=("r_red", "w_red"),
    ),
    "v11_cautious_driver": ExperimentSpec(
        experiment_id="v11_cautious_driver",
        title="Vehicle 11: Cautious Driver",
        subtitle="Ergotrix Temporal Sequence Staircases ($w_c = 0.7$), Preemptive Braking & 3×3 Synaptic Matrix",
        scenario_path="scenarios/vehicle11-cautious-driver.yaml",
        focal_vehicle_path="vehicles/vehicle11-cautious-driver.yaml",
        phase_axes=("d_line", "v"),
    ),
    "v12_daydreamer": ExperimentSpec(
        experiment_id="v12_daydreamer",
        title="Vehicle 12: Daydreamer",
        subtitle="Caudate Homeostatic Criticality, Cortical Mandala Z(t), Idea Cascade & Thought Loops",
        scenario_path="scenarios/vehicle12-daydreamer.yaml",
        focal_vehicle_path="vehicles/vehicle12-daydreamer.yaml",
        phase_axes=("Re(Z)", "Im(Z)"),
    ),
}

_DEFINITION_CACHE: dict[Path, VehicleDefinition] = {}


def _cached_load_definition(path: str | Path) -> VehicleDefinition:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if resolved not in _DEFINITION_CACHE:
        _DEFINITION_CACHE[resolved] = load_definition(resolved)
    return _DEFINITION_CACHE[resolved]


def connectome_signature(definition: VehicleDefinition) -> tuple:
    """Return an immutable structural signature of a vehicle definition's connectome.

    Two vehicles share a connectome signature iff they have the exact same Braitenberg
    ``general.type``, ``general.name``, sensors/cameras, actuators, modules, arrays,
    units, and wire topology (learned synaptic weights and random jitters may vary across
    individuals of the same species).
    """
    d = definition
    sensors_sig = tuple(
        (s.name, s.type, s.camera.shape if s.camera is not None else None)
        for s in d.sensors
    )
    actuators_sig = tuple(
        (
            a.name,
            a.type,
            tuple(
                (
                    c.source,
                    int(c.sign),
                    c.learning.type if c.learning is not None else None,
                    c.port,
                    c.label,
                    bool(c.active) if c.learning is None else True,
                )
                for c in a.inputs
            ),
        )
        for a in d.actuators
    )
    modules_sig = tuple(
        (name, tuple(m.inputs), tuple(m.outputs))
        for name, m in sorted(d.modules.items())
    )
    arrays_sig = tuple(sorted((name, tuple(shape)) for name, shape in d.arrays.items()))
    units_sig = tuple(
        (
            u.name,
            bool(u.is_control),
            u.array,
            u.module,
            tuple(u.controls) if u.controls else (),
            tuple(
                (
                    c.source,
                    int(c.sign),
                    c.learning.type if c.learning is not None else None,
                    c.port,
                    c.label,
                    bool(c.active) if c.learning is None else True,
                )
                for c in u.inputs
            ),
        )
        for u in d.units
    )
    return (
        str(d.general.type),
        str(d.general.name),
        sensors_sig,
        actuators_sig,
        modules_sig,
        arrays_sig,
        units_sig,
    )


def _resolve_definition_arg(focal_definition) -> VehicleDefinition:
    if isinstance(focal_definition, VehicleDefinition):
        return focal_definition
    if isinstance(focal_definition, Vehicle):
        return focal_definition.definition
    if isinstance(focal_definition, ExperimentSpec):
        return _cached_load_definition(focal_definition.focal_vehicle_path)
    if isinstance(focal_definition, (str, Path)):
        if isinstance(focal_definition, str):
            if focal_definition in EXPERIMENTS:
                return _cached_load_definition(EXPERIMENTS[focal_definition].focal_vehicle_path)
            if focal_definition in PHASE_EXPERIMENTS:
                return _cached_load_definition(PHASE_EXPERIMENTS[focal_definition].focal_vehicle_path)
        return _cached_load_definition(focal_definition)
    raise AxiomViolationError(f"Unsupported focal_definition type: {type(focal_definition)!r}")


def select_focal_indices(
    sim_vehicles: Sequence[Vehicle],
    focal_definition: VehicleDefinition | Vehicle | ExperimentSpec | str | Path | None = None,
) -> list[int]:
    """Return indices of vehicles matching the homogeneous focal species connectome."""
    if not sim_vehicles:
        return []
    ref_def = sim_vehicles[0].definition if focal_definition is None else _resolve_definition_arg(focal_definition)
    ref_sig = connectome_signature(ref_def)
    return [i for i, v in enumerate(sim_vehicles) if connectome_signature(v.definition) == ref_sig]


def select_focal_vehicles(
    sim_vehicles: Sequence[Vehicle],
    focal_definition: VehicleDefinition | Vehicle | ExperimentSpec | str | Path | None = None,
) -> list[Vehicle]:
    """Select the homogeneous focal vehicle population, excluding stimulus/environmental agents.

    Defaults to matching ``sim_vehicles[0].definition`` so that in scenarios such as
    ``dynamics-shy-swarm.yaml`` (or ``vehicle9-shy.yaml``), ``Vehicle 9: Shy`` individuals
    form the pure focal ensemble while ``Chaser`` and ``Walker`` vehicles act strictly as
    moving environmental stimuli.
    """
    idxs = select_focal_indices(sim_vehicles, focal_definition=focal_definition)
    return [sim_vehicles[i] for i in idxs]


_CANONICAL_SIG_CACHE: dict[tuple, ExperimentSpec] | None = None


def _match_canonical_spec(definition: VehicleDefinition) -> ExperimentSpec:
    global _CANONICAL_SIG_CACHE
    if _CANONICAL_SIG_CACHE is None:
        cache: dict[tuple, ExperimentSpec] = {
            connectome_signature(_cached_load_definition(sp.focal_vehicle_path)): sp
            for sp in PHASE_EXPERIMENTS.values()
        }
        cache.update(
            {
                connectome_signature(_cached_load_definition(sp.focal_vehicle_path)): sp
                for sp in EXPERIMENTS.values()
            }
        )
        _CANONICAL_SIG_CACHE = cache
    sig = connectome_signature(definition)
    if sig in _CANONICAL_SIG_CACHE:
        return _CANONICAL_SIG_CACHE[sig]
    slug = f"v{definition.general.type}_{definition.general.name}".lower().replace(" ", "_")
    return ExperimentSpec(
        experiment_id=slug,
        title=f"Vehicle {definition.general.type}: {definition.general.name}",
        subtitle=definition.general.description or f"Homogeneous {definition.general.name} ensemble",
        scenario_path="",
        focal_vehicle_path=str(definition.path) if definition.path else "",
        phase_axes=("r", "psi"),
    )


def _resolve_single_spec(spec_or_specs) -> ExperimentSpec:
    if getattr(spec_or_specs, "comparative_synthesis", False) or getattr(
        spec_or_specs, "series", None
    ) == "phase_atlas":
        raise AxiomViolationError(
            "Axiom 1 violation: comparative phase_atlas plates synthesize multiple vehicles "
            "and must not be treated as single-experiment plates."
        )
    if isinstance(spec_or_specs, dict) and (
        spec_or_specs.get("comparative_synthesis") or spec_or_specs.get("series") == "phase_atlas"
    ):
        raise AxiomViolationError(
            "Axiom 1 violation: comparative phase_atlas metadata cannot pass single-experiment validation."
        )

    if isinstance(spec_or_specs, ExperimentSpec):
        if (
            not spec_or_specs.experiment_id
            or not isinstance(spec_or_specs.phase_axes, tuple)
            or len(spec_or_specs.phase_axes) != 2
        ):
            raise AxiomViolationError(f"Invalid ExperimentSpec: {spec_or_specs!r}")
        return spec_or_specs

    if isinstance(spec_or_specs, str):
        key = spec_or_specs.strip()
        if not key or "," in key or "+" in key or key.lower() == "all":
            raise AxiomViolationError(
                f"Axiom 1 violation: expected strictly one experiment ID, got {spec_or_specs!r}"
            )
        if key in EXPERIMENTS:
            return EXPERIMENTS[key]
        if key in PHASE_EXPERIMENTS:
            return PHASE_EXPERIMENTS[key]
        valid_ids = list(dict.fromkeys((*EXPERIMENTS.keys(), *PHASE_EXPERIMENTS.keys())))
        raise AxiomViolationError(
            f"Unknown experiment_id {key!r}; valid canonical IDs are {valid_ids}"
        )

    if isinstance(spec_or_specs, Iterable) and not isinstance(spec_or_specs, (bytes, dict)):
        items = list(spec_or_specs)
        if len(items) != 1:
            raise AxiomViolationError(
                f"Axiom 1 violation: visualization must isolate strictly 1 experiment, got {len(items)}"
            )
        return _resolve_single_spec(items[0])

    raise AxiomViolationError(
        f"Axiom 1 violation: unsupported experiment specification {spec_or_specs!r}"
    )


def validate_single_experiment_axiom(
    spec_or_specs,
    vehicles: Sequence[Vehicle] | None = None,
    scenario: Scenario | str | Path | Sequence | None = None,
) -> ExperimentSpec:
    """Enforce Axiom 1: strictly ONE experiment scenario and ONE homogeneous focal species.

    Raises
    ------
    AxiomViolationError
        If multiple specs or scenarios are passed, if ``vehicles`` is empty, or if
        vehicles do not all share the exact same ``general.type``, ``general.name``,
        part names/kinds, and wire topology.
    """
    # Allow passing a homogeneous list[Vehicle] directly when spec_or_specs is omitted or is vehicles
    if vehicles is None and isinstance(spec_or_specs, Sequence) and not isinstance(spec_or_specs, (str, bytes)):
        if len(spec_or_specs) == 0:
            raise AxiomViolationError("Axiom 1 violation: empty specification / vehicle sequence.")
        if all(isinstance(v, Vehicle) for v in spec_or_specs):
            vehicles = spec_or_specs
            spec_or_specs = None

    spec: ExperimentSpec | None = None
    expected_sig: tuple | None = None

    if spec_or_specs is not None:
        spec = _resolve_single_spec(spec_or_specs)
        if spec.focal_vehicle_path:
            expected_def = _cached_load_definition(spec.focal_vehicle_path)
            expected_sig = connectome_signature(expected_def)

    if scenario is not None:
        if isinstance(scenario, Iterable) and not isinstance(scenario, (str, bytes, Path, Scenario, dict)):
            sc_list = list(scenario)
            if len(sc_list) != 1:
                raise AxiomViolationError(
                    f"Axiom 1 violation: expected strictly 1 scenario, got {len(sc_list)}"
                )
            scenario = sc_list[0]

        if isinstance(scenario, (str, Path)):
            sc_path = Path(scenario)
            if not sc_path.is_absolute():
                sc_path = (REPO_ROOT / sc_path).resolve()
            sc_obj = load_scenario(sc_path)
        elif isinstance(scenario, Scenario):
            sc_obj = scenario
        else:
            raise AxiomViolationError(f"Axiom 1 violation: invalid scenario {scenario!r}")

        if not sc_obj.vehicles:
            raise AxiomViolationError(f"Axiom 1 violation: scenario {sc_obj.name!r} has no vehicles.")

        first_sig = connectome_signature(sc_obj.vehicles[0].definition)
        if expected_sig is not None and first_sig != expected_sig:
            raise AxiomViolationError(
                f"Axiom 1 violation: scenario {sc_obj.name!r} hero vehicle "
                f"({sc_obj.vehicles[0].definition.general.type}: {sc_obj.vehicles[0].definition.general.name}) "
                f"does not match experiment {spec.experiment_id!r}."
            )
        if spec is None:
            spec = _match_canonical_spec(sc_obj.vehicles[0].definition)
            expected_sig = first_sig

    if vehicles is not None:
        if not isinstance(vehicles, Sequence):
            vehicles = list(vehicles)
        if len(vehicles) == 0:
            raise AxiomViolationError("Axiom 1 violation: focal vehicle population cannot be empty.")
        if not all(isinstance(v, Vehicle) for v in vehicles):
            raise AxiomViolationError("Axiom 1 violation: all items in vehicles must be Vehicle instances.")

        ref_veh = vehicles[0]
        ref_def = ref_veh.definition
        ref_sig: tuple | None = None
        for idx, veh in enumerate(vehicles[1:], start=1):
            if veh.definition is ref_def:
                continue
            if ref_sig is None:
                ref_sig = connectome_signature(ref_def)
            sig_i = connectome_signature(veh.definition)
            if sig_i != ref_sig:
                raise AxiomViolationError(
                    f"Axiom 1 violation: heterogeneous vehicles mixed in ensemble! "
                    f"Vehicle #0 is 'Vehicle {ref_def.general.type}: {ref_def.general.name}' "
                    f"while Vehicle #{idx} is 'Vehicle {veh.definition.general.type}: {veh.definition.general.name}' "
                    f"(or has differing parts/wires)."
                )
        if expected_sig is not None:
            if ref_sig is None:
                ref_sig = connectome_signature(ref_def)
            if ref_sig != expected_sig:
                raise AxiomViolationError(
                    f"Axiom 1 violation: vehicle species "
                    f"'Vehicle {ref_def.general.type}: {ref_def.general.name}' "
                    f"does not match experiment {spec.experiment_id!r}."
                )
        if spec is None:
            spec = _match_canonical_spec(ref_def)

    if spec is None:
        raise AxiomViolationError("Axiom 1 violation: no experiment spec, scenario, or vehicles provided.")

    return spec


# ---------------------------------------------------------------------------
# Exact Base-2 Shannon Information Primitives
# ---------------------------------------------------------------------------


def binary_entropy_bits(p: float | np.ndarray) -> float | np.ndarray:
    """Compute exact base-2 binary entropy H_2(p) in [0.0, 1.0] bits (0.0 at p=0 and p=1)."""
    arr = np.asarray(p, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = (arr > 0.0) & (arr < 1.0)
    if np.any(mask):
        pm = arr[mask]
        qm = 1.0 - pm
        val = -(pm * np.log2(pm) + qm * np.log2(qm))
        out[mask] = np.clip(val, 0.0, 1.0)
    if np.ndim(p) == 0:
        return 0.0 if float(out) == 0.0 else float(out)
    return out


def discrete_entropy_bits(labels: np.ndarray) -> float:
    """Exact empirical Shannon entropy H(X) in bits (base 2) for discrete integer labels."""
    arr = np.asarray(labels)
    n = arr.size
    if n <= 1:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    if counts.size <= 1:
        return 0.0
    probs = counts.astype(float) / float(n)
    ent = float(-np.sum(probs * np.log2(probs)))
    return 0.0 if ent <= 1e-15 else ent


def discrete_mi_bits(x: np.ndarray, y: np.ndarray) -> float:
    """Exact empirical Mutual Information I(X; Y) = H(X) + H(Y) - H(X, Y) >= 0 in bits."""
    x_arr = np.asarray(x, dtype=np.int64).ravel()
    y_arr = np.asarray(y, dtype=np.int64).ravel()
    n = x_arr.size
    if n <= 1 or y_arr.size != n:
        return 0.0
    hx = discrete_entropy_bits(x_arr)
    if hx == 0.0:
        return 0.0
    hy = discrete_entropy_bits(y_arr)
    if hy == 0.0:
        return 0.0
    max_y = int(y_arr.max()) - int(y_arr.min()) + 1
    joint = (x_arr - int(x_arr.min())) * max(max_y, 1) + (y_arr - int(y_arr.min()))
    hxy = discrete_entropy_bits(joint)
    mi = hx + hy - hxy
    return 0.0 if mi <= 1e-14 else float(mi)


def discrete_transfer_entropy_bits(
    d_now: np.ndarray,
    s_prev: np.ndarray,
    d_prev: np.ndarray,
) -> float:
    """Exact empirical Schreiber Transfer Entropy T_{S -> D} = I(D_t; S_{t-1} | D_{t-1}) >= 0 in bits."""
    dn = np.asarray(d_now, dtype=np.int64).ravel()
    sp = np.asarray(s_prev, dtype=np.int64).ravel()
    dp = np.asarray(d_prev, dtype=np.int64).ravel()
    n = dn.size
    if n <= 1 or sp.size != n or dp.size != n:
        return 0.0
    dn0 = dn - int(dn.min())
    sp0 = sp - int(sp.min())
    dp0 = dp - int(dp.min())
    b_dn = int(dn0.max()) + 1
    b_sp = int(sp0.max()) + 1
    if b_dn <= 1 or b_sp <= 1:
        return 0.0

    # I(D_t; S_{t-1} | D_{t-1}) = H(D_t, D_{t-1}) + H(S_{t-1}, D_{t-1}) - H(D_t, S_{t-1}, D_{t-1}) - H(D_{t-1})
    h_dp = discrete_entropy_bits(dp0)
    h_dn_dp = discrete_entropy_bits(dp0 * b_dn + dn0)
    h_sp_dp = discrete_entropy_bits(dp0 * b_sp + sp0)
    h_all = discrete_entropy_bits((dp0 * b_sp + sp0) * b_dn + dn0)
    te = h_dn_dp + h_sp_dp - h_all - h_dp
    return 0.0 if te <= 1e-14 else float(te)


def _quantize_4bins(values: np.ndarray, is_binary: bool = False) -> np.ndarray:
    """Quantize a 1D signal across N vehicles into at most 4 discrete bins {0, 1, 2, 3}."""
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if is_binary:
        return (arr > 0.5).astype(np.int64)
    if float(np.ptp(arr)) <= 1e-12:
        return np.zeros(n, dtype=np.int64)

    # Separate exact zeros from positive activations when >=1 vehicle is silent and >=1 is active
    zero_mask = (arr >= 0.0) & (arr <= 1e-9)
    pos_mask = arr > 1e-9
    if np.any(zero_mask) and np.any(pos_mask) and (np.sum(zero_mask) + np.sum(pos_mask) == n):
        out = np.zeros(n, dtype=np.int64)
        pos_vals = arr[pos_mask]
        if float(np.ptp(pos_vals)) <= 1e-12:
            out[pos_mask] = 1
        else:
            edges = np.unique(np.quantile(pos_vals, [1.0 / 3.0, 2.0 / 3.0]))
            out[pos_mask] = 1 + np.searchsorted(edges, pos_vals, side="right")
        return out

    edges = np.unique(np.quantile(arr, [0.25, 0.5, 0.75]))
    if edges.size == 0:
        edges = np.linspace(float(arr.min()), float(arr.max()), 5)[1:-1]
    return np.searchsorted(edges, arr, side="right").astype(np.int64)


def _partition_phase_space_4x4(u_coord: np.ndarray, v_coord: np.ndarray) -> np.ndarray:
    """Partition 2D phase coordinates (N,) into a 4x4 quantile/grid region index in {0..15}."""
    u = np.asarray(u_coord, dtype=float).ravel()
    v = np.asarray(v_coord, dtype=float).ravel()
    n = u.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    def _bin_axis(arr: np.ndarray) -> np.ndarray:
        finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.ptp(finite)) <= 1e-12:
            return np.zeros(n, dtype=np.int64)
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
        if edges.size == 0:
            edges = np.linspace(float(finite.min()), float(finite.max()), 5)[1:-1]
        return np.searchsorted(edges, finite, side="right").astype(np.int64)

    bu = _bin_axis(u)
    bv = _bin_axis(v)
    return 4 * bu + bv


def _sensor_ref_for(vehicle: Vehicle, sensor, signal_name: str) -> float:
    if sensor.response.type != "linear":
        return float(sensor.response.height)
    if sensor.type == ENERGY_SENSOR and vehicle.definition.energy is not None:
        return float(vehicle.definition.energy.max)
    return float(vehicle.references.get(signal_name, 1.0))


def lookup_wire_info(snapshot: dict, wire, metric: str = "info_bits") -> float:
    """Ergonomic helper to query wire MI / TE / combined info_bits for a Diagram Wire."""
    table_map = {
        "info_bits": snapshot.get("wire_info_bits", {}),
        "mi_bits": snapshot.get("wire_mi_bits", {}),
        "te_bits": snapshot.get("wire_te_bits", {}),
        "active_fraction": snapshot.get("wire_active_fraction", {}),
    }
    table = table_map.get(metric, snapshot.get("wire_info_bits", {}))
    key3 = (wire.source, wire.target, int(wire.sign))
    if key3 in table:
        return float(table[key3])
    pair_str = f"{wire.source}->{wire.target}"
    if metric == "info_bits" and pair_str in snapshot.get("wire_info_by_pair", {}):
        return float(snapshot["wire_info_by_pair"][pair_str])
    return 0.0


# ---------------------------------------------------------------------------
# Live & Batch Shannon Ensemble Snapshot Engine
# ---------------------------------------------------------------------------


def compute_shannon_snapshot(
    focal_vehicles: list[Vehicle],
    prev_snapshot: dict | None = None,
    phase_coords: tuple[np.ndarray, np.ndarray] | None = None,
    hero_index: int = 0,
    max_codewords: int = 6,
    barcode_bins: int = 20,
) -> dict:
    """Compute a lossless Claude Shannon information snapshot across a homogeneous swarm.

    Designed for both offline batch rollouts (`tri-studio`) and 60 FPS real-time HUD
    rendering (`hud-shannon`) with sub-millisecond overhead.
    """
    validate_single_experiment_axiom(None, vehicles=focal_vehicles)

    n_veh = len(focal_vehicles)
    if hero_index < 0 or hero_index >= n_veh:
        raise IndexError(f"hero_index {hero_index} out of range for {n_veh} focal vehicles.")
    max_codewords = max(1, int(max_codewords))
    barcode_bins = max(2, int(barcode_bins))

    hero = focal_vehicles[hero_index]
    d = hero.definition
    hero_diagram = hero.diagram()

    sensors_by_name = {s.name: s for s in d.sensors}
    actuators_by_name = {a.name: a for a in d.actuators}
    units_by_name = {u.name: u for u in d.units}
    unit_pos = {u.name: k for k, u in enumerate(d.units)}
    array_elements = {
        arr_name: [u.name for u in d.units if u.array == arr_name]
        for arr_name in d.arrays
    }

    # Pre-extract continuous normalized levels in [0, 1], raw values, and 2D grids for all parts
    part_levels: dict[str, np.ndarray] = {}
    part_raw_vals: dict[str, np.ndarray] = {}
    grid_prob: dict[str, np.ndarray] = {}
    grid_entropy: dict[str, np.ndarray] = {}
    grid_mean_level: dict[str, np.ndarray] = {}

    for p in hero_diagram.parts:
        if p.kind == "sensor":
            s = sensors_by_name[p.name]
            raw = np.array([float(v.sensor_values[p.name]) for v in focal_vehicles], dtype=float)
            lv = np.empty(n_veh, dtype=float)
            for i, v in enumerate(focal_vehicles):
                ref = _sensor_ref_for(v, s, p.name)
                val = raw[i]
                lv[i] = clamp01(val / ref) if ref > 0.0 else (1.0 if val > 0.0 else 0.0)
            part_raw_vals[p.name] = raw
            part_levels[p.name] = lv

        elif p.kind == "camera":
            s = sensors_by_name[p.name]
            rows, cols = s.camera.shape
            px_names = hero.pixel_names.get(s.name) or s.pixels()
            refs_vec = np.array([_sensor_ref_for(hero, s, nm) for nm in px_names], dtype=float)
            raw_px = np.array(
                [[v.sensor_values[nm] for nm in px_names] for v in focal_vehicles],
                dtype=float,
            )
            cam_flat = np.where(
                refs_vec[None, :] > 0.0,
                np.clip(raw_px / np.maximum(refs_vec[None, :], 1e-12), 0.0, 1.0),
                np.where(raw_px > 0.0, 1.0, 0.0),
            )
            cam_levels = cam_flat.reshape((n_veh, rows, cols))
            cam_on = (cam_levels > 0.5).astype(float)
            p_grid = np.mean(cam_on, axis=0)
            grid_prob[p.name] = p_grid
            grid_entropy[p.name] = np.asarray(binary_entropy_bits(p_grid), dtype=float)
            grid_mean_level[p.name] = np.mean(cam_levels, axis=0)
            active_frac = np.mean(cam_on, axis=(1, 2))
            part_raw_vals[p.name] = np.sum(cam_on, axis=(1, 2))
            part_levels[p.name] = active_frac

        elif p.kind == "array":
            shape = p.shape if p.shape is not None else (len(array_elements[p.name]),)
            rows, cols = shape if len(shape) == 2 else (1, shape[0])
            elems = array_elements[p.name]
            arr_vals = np.array(
                [[float(v.unit_outputs[nm]) for nm in elems] for v in focal_vehicles],
                dtype=float,
            ).reshape((n_veh, rows, cols))
            arr_on = (arr_vals > 0.5).astype(float)
            p_grid = np.mean(arr_on, axis=0)
            grid_prob[p.name] = p_grid if len(shape) == 2 else p_grid.reshape(shape)
            grid_entropy[p.name] = np.asarray(binary_entropy_bits(grid_prob[p.name]), dtype=float)
            grid_mean_level[p.name] = grid_prob[p.name].copy()
            part_raw_vals[p.name] = np.sum(arr_on, axis=(1, 2))
            part_levels[p.name] = np.mean(arr_on, axis=(1, 2))

        elif p.kind == "module":
            use = d.modules[p.name]
            out_names = [f"{p.name}.{o}" for o in use.outputs]
            mod_vals = np.array(
                [[float(v.unit_outputs[nm]) for nm in out_names] for v in focal_vehicles],
                dtype=float,
            )
            mod_on = (mod_vals > 0.5).astype(float)
            part_raw_vals[p.name] = np.sum(mod_on, axis=1)
            part_levels[p.name] = np.mean(mod_on, axis=1) if out_names else np.zeros(n_veh, dtype=float)

        elif p.kind == "control":
            u = units_by_name[p.name]
            k_u = unit_pos[p.name]
            offsets = np.array([float(v.unit_outputs[p.name]) for v in focal_vehicles], dtype=float)
            lv = np.array(
                [clamp01(float(v.activity[k_u]) / (2.0 * u.target)) if u.target > 0 else 0.0 for v in focal_vehicles],
                dtype=float,
            )
            part_raw_vals[p.name] = offsets
            part_levels[p.name] = lv

        elif p.kind == "motor":
            a = actuators_by_name[p.name]
            raw = np.array([float(v.motor_speeds[p.name]) for v in focal_vehicles], dtype=float)
            lv = np.clip(raw / a.max_speed, 0.0, 1.0) if a.max_speed > 0 else np.zeros(n_veh, dtype=float)
            part_raw_vals[p.name] = raw
            part_levels[p.name] = lv

        else:
            # Standard binary unit
            raw = np.array([float(v.unit_outputs[p.name]) for v in focal_vehicles], dtype=float)
            part_raw_vals[p.name] = raw
            part_levels[p.name] = np.clip(raw, 0.0, 1.0)

    # 3. Discrete Feature / Neuron Matrix B of shape (N, M) in {0, 1}
    internal_parts = [p for p in hero_diagram.parts if p.kind in ("unit", "array", "module", "control")]
    has_internal_units = len(internal_parts) > 0

    feature_names: list[str] = []
    b_cols: list[np.ndarray] = []

    if has_internal_units:
        for p in internal_parts:
            feature_names.append(p.name)
            if p.kind == "unit":
                b_cols.append((part_raw_vals[p.name] > 0.5).astype(int))
            elif p.kind in ("array", "module"):
                b_cols.append((part_raw_vals[p.name] > 0.0).astype(int))
            elif p.kind == "control":
                b_cols.append((part_raw_vals[p.name] > 0.0).astype(int))
    else:
        # Memoryless vehicle (e.g. Vehicle 4a Orbiter): 4-column sensorimotor state
        sensor_parts = [p for p in hero_diagram.parts if p.kind in ("sensor", "camera")]
        left_lv = part_levels[sensor_parts[0].name] if len(sensor_parts) >= 1 else np.zeros(n_veh)
        right_lv = (
            part_levels[sensor_parts[1].name]
            if len(sensor_parts) >= 2
            else left_lv
        )
        turn_rates = np.array([float(v.turn_rate) for v in focal_vehicles], dtype=float)
        # Compute motor differential if move() hasn't stepped yet (e.g. fresh Vehicle instances)
        if np.all(turn_rates == 0.0) and len(d.actuators) >= 2:
            xs = [a.position[0] * hero.half_width for a in d.actuators]
            sxx = sum(x * x for x in xs)
            if sxx > 0.0:
                turn_rates = np.array(
                    [
                        sum(v.motor_speeds[a.name] * x for a, x in zip(d.actuators, xs)) / sxx
                        for v in focal_vehicles
                    ],
                    dtype=float,
                )
        turn_eps = 0.02
        feature_names = ["left_eye_high", "right_eye_high", "turn_left", "turn_right"]
        b_cols = [
            (left_lv > 0.5).astype(int),
            (right_lv > 0.5).astype(int),
            (turn_rates > turn_eps).astype(int),
            (turn_rates < -turn_eps).astype(int),
        ]

    B = np.column_stack(b_cols).astype(int) if b_cols else np.zeros((n_veh, 0), dtype=int)
    n_features = B.shape[1]

    # 4. Exact Joint Neural Codebook (lossless joint state preservation)
    if n_features > 0:
        unique_rows, inv_lex, counts_lex = np.unique(B, axis=0, return_inverse=True, return_counts=True)
        # Sort descending by count, then descending by active bit count, then lexicographic index
        bit_sums = np.sum(unique_rows, axis=1)
        order_cw = np.lexsort((np.arange(len(counts_lex)), -bit_sums, -counts_lex))
        sorted_unique = unique_rows[order_cw]
        sorted_counts = counts_lex[order_cw]
        remap = np.empty(len(order_cw), dtype=int)
        remap[order_cw] = np.arange(len(order_cw), dtype=int)
        raw_codeword_ids = remap[inv_lex]
    else:
        sorted_unique = np.zeros((1, 0), dtype=int)
        sorted_counts = np.array([n_veh], dtype=int)
        raw_codeword_ids = np.zeros(n_veh, dtype=int)

    k_raw = len(sorted_counts)
    probs_all = sorted_counts.astype(float) / float(n_veh)

    codebook_full: list[dict] = []
    for c_idx in range(k_raw):
        bits_tuple = tuple(int(b) for b in sorted_unique[c_idx])
        active_u = [feature_names[j] for j, b in enumerate(bits_tuple) if b == 1]
        codebook_full.append(
            {
                "codeword_id": c_idx,
                "bits": bits_tuple,
                "count": int(sorted_counts[c_idx]),
                "prob": float(probs_all[c_idx]),
                "active_units": active_u,
                "is_residual": False,
            }
        )

    if k_raw <= max_codewords:
        codebook = list(codebook_full)
        vehicle_codeword_ids = raw_codeword_ids.copy()
    else:
        codebook = list(codebook_full[:max_codewords])
        res_count = int(np.sum(sorted_counts[max_codewords:]))
        res_prob = float(np.sum(probs_all[max_codewords:]))
        res_active = [
            feature_names[j]
            for j in range(n_features)
            if np.any(sorted_unique[max_codewords:, j] == 1)
        ]
        codebook.append(
            {
                "codeword_id": max_codewords,
                "bits": (),
                "count": res_count,
                "prob": res_prob,
                "active_units": res_active,
                "is_residual": True,
            }
        )
        vehicle_codeword_ids = np.minimum(raw_codeword_ids, max_codewords)

    # Helper to decompose ANY binary activation vector (N,) into exact (palette_codeword_id, prob_mass) arcs
    def _build_codeword_arcs(active_binary_vec: np.ndarray) -> list[tuple[int, float]]:
        arcs: list[tuple[int, float]] = []
        act = (np.asarray(active_binary_vec) > 0.5)
        if not np.any(act):
            return arcs
        max_bucket = min(k_raw - 1, max_codewords)
        for b_id in range(max_bucket + 1):
            n_on = int(np.sum(act & (vehicle_codeword_ids == b_id)))
            if n_on > 0:
                arcs.append((b_id, float(n_on) / float(n_veh)))
        return arcs

    feature_firing_rate: dict[str, float] = {}
    feature_entropy: dict[str, float] = {}
    part_codeword_arcs: dict[str, list[tuple[int, float]]] = {}

    for j, fname in enumerate(feature_names):
        col = B[:, j]
        p_on = float(np.mean(col))
        feature_firing_rate[fname] = p_on
        feature_entropy[fname] = float(binary_entropy_bits(p_on))
        part_codeword_arcs[fname] = _build_codeword_arcs(col)

    part_firing_rate: dict[str, float] = dict(feature_firing_rate)
    part_entropy: dict[str, float] = {}

    # Populate part_firing_rate, part_entropy, and part_codeword_arcs for every Diagram part
    for p in hero_diagram.parts:
        if p.name in feature_names:
            # Internal unit / array / module / control
            if p.name in grid_entropy:
                # For arrays, part_entropy reports mean cell binary entropy (and grid_entropy holds 2D grid)
                part_entropy[p.name] = float(np.mean(grid_entropy[p.name]))
            else:
                part_entropy[p.name] = feature_entropy[p.name]
        elif not has_internal_units:
            # Memoryless vehicle (e.g. v4a_orbiter): map eyes & motors directly to sensorimotor features
            sensor_names = [sp.name for sp in hero_diagram.parts if sp.kind in ("sensor", "camera")]
            motor_names = [mp.name for mp in hero_diagram.parts if mp.kind == "motor"]
            if len(sensor_names) >= 1 and p.name == sensor_names[0]:
                col = B[:, 0]
            elif len(sensor_names) >= 2 and p.name == sensor_names[1]:
                col = B[:, 1]
            elif len(motor_names) >= 2 and p.name == motor_names[0]:
                # Left motor faster -> turning right (turn_right = B[:, 3])
                col = B[:, 3]
            elif len(motor_names) >= 2 and p.name == motor_names[1]:
                # Right motor faster -> turning left (turn_left = B[:, 2])
                col = B[:, 2]
            else:
                col = (part_levels[p.name] > 0.5).astype(int)
            p_on = float(np.mean(col))
            part_firing_rate[p.name] = p_on
            part_entropy[p.name] = (
                float(np.mean(grid_entropy[p.name]))
                if p.name in grid_entropy
                else float(binary_entropy_bits(p_on))
            )
            part_codeword_arcs[p.name] = _build_codeword_arcs(col)
        else:
            # Sensor / Camera / Motor on a vehicle with internal units
            if p.kind == "camera":
                col = (part_raw_vals[p.name] > 0.0).astype(int)
                part_firing_rate[p.name] = float(np.mean(col))
                part_entropy[p.name] = float(np.mean(grid_entropy[p.name]))
                part_codeword_arcs[p.name] = _build_codeword_arcs(col)
            else:
                col = (part_levels[p.name] > 0.5).astype(int)
                p_on = float(np.mean(col))
                part_firing_rate[p.name] = p_on
                part_entropy[p.name] = float(binary_entropy_bits(p_on))
                part_codeword_arcs[p.name] = _build_codeword_arcs(col)

    # 5. Exact Shannon Entropies & Watanabe Synergy / Total Correlation (all in bits, base 2)
    pos_probs = probs_all[probs_all > 0.0]
    joint_entropy_bits = 0.0 if pos_probs.size <= 1 else float(-np.sum(pos_probs * np.log2(pos_probs)))
    if joint_entropy_bits <= 1e-15:
        joint_entropy_bits = 0.0

    marginal_entropy_sum_bits = float(sum(feature_entropy[nm] for nm in feature_names))
    if marginal_entropy_sum_bits <= 1e-15:
        marginal_entropy_sum_bits = 0.0

    total_correlation_bits = max(0.0, marginal_entropy_sum_bits - joint_entropy_bits)
    if total_correlation_bits <= 1e-14:
        total_correlation_bits = 0.0

    # Spatial-Neural Mutual Information I(X; W) across 4x4 binned phase coordinates
    if phase_coords is not None:
        px_arr = np.asarray(phase_coords[0], dtype=float).ravel()
        py_arr = np.asarray(phase_coords[1], dtype=float).ravel()
        if px_arr.size != n_veh or py_arr.size != n_veh:
            raise ValueError(
                f"phase_coords length ({px_arr.size}, {py_arr.size}) must match focal_vehicles ({n_veh})."
            )
    else:
        px_arr = np.array([float(v.x) for v in focal_vehicles], dtype=float)
        py_arr = np.array([float(v.y) for v in focal_vehicles], dtype=float)

    spatial_bins = _partition_phase_space_4x4(px_arr, py_arr)
    spatial_mi_bits = discrete_mi_bits(spatial_bins, raw_codeword_ids)

    # 6. Bundled-Wire Channel Mutual Information I(S_t; D_t) & Transfer Entropy T_{S -> D}
    quantized_signals: dict[str, np.ndarray] = {}
    for p in hero_diagram.parts:
        if p.kind == "unit":
            quantized_signals[p.name] = _quantize_4bins(part_raw_vals[p.name], is_binary=True)
        elif p.kind == "control":
            quantized_signals[p.name] = _quantize_4bins(part_raw_vals[p.name], is_binary=False)
        else:
            quantized_signals[p.name] = _quantize_4bins(part_levels[p.name], is_binary=False)

    prev_q = prev_snapshot.get("quantized_signals") if isinstance(prev_snapshot, dict) else None
    has_valid_prev = (
        isinstance(prev_q, dict)
        and all(k in prev_q and len(prev_q[k]) == n_veh for k in quantized_signals)
    )

    # Pre-index learning input strengths across vehicles for (source_part, target_part, sign)
    learning_strengths_by_edge: dict[tuple[str, str, int], np.ndarray] = {}
    for u in d.units:
        t_part = u.module or u.array or u.name
        for j_in, c in enumerate(u.inputs):
            if c.learning is not None:
                s_part = c.source.split(".", 1)[0] if "." in c.source else c.source.split("[", 1)[0]
                ekey = (s_part, t_part, int(c.sign))
                arr_s = np.array([float(v.strengths[(u.name, j_in)]) for v in focal_vehicles], dtype=float)
                if ekey in learning_strengths_by_edge:
                    learning_strengths_by_edge[ekey] = np.maximum(learning_strengths_by_edge[ekey], arr_s)
                else:
                    learning_strengths_by_edge[ekey] = arr_s

    bundled = bundles(hero_diagram.wires)
    wire_mi_bits: dict[tuple[str, str, int], float] = {}
    wire_te_bits: dict[tuple[str, str, int], float] = {}
    wire_info_bits: dict[tuple[str, str, int], float] = {}
    wire_active_fraction: dict[tuple[str, str, int], float] = {}
    wire_info_by_pair: dict[str, float] = {}
    wire_metrics: list[dict] = []

    for w in bundled:
        wkey = (w.source, w.target, int(w.sign))
        sq = quantized_signals.get(w.source)
        dq = quantized_signals.get(w.target)

        if sq is None or dq is None:
            mi_val, te_val, act_frac = 0.0, 0.0, 0.0
        elif w.learning is not None:
            strengths_vec = learning_strengths_by_edge.get(wkey)
            if strengths_vec is None:
                strengths_vec = np.array([float(v.strength(w.source, w.target)) for v in focal_vehicles], dtype=float)
            act_mask = strengths_vec > ACTIVE_STRENGTH
            act_frac = float(np.mean(act_mask))
            if not np.any(act_mask):
                mi_val = 0.0
                te_val = 0.0
            else:
                # Effective gated signal + conditional MI within active-synapse subpopulation
                s_gated = np.where(act_mask, sq + 1, 0)
                mi_sub = discrete_mi_bits(sq[act_mask], dq[act_mask]) if int(np.sum(act_mask)) >= 2 else 0.0
                mi_gated = discrete_mi_bits(s_gated, dq)
                mi_val = act_frac * max(mi_sub, mi_gated)
                if has_valid_prev:
                    sp = prev_q[w.source]
                    dp = prev_q[w.target]
                    sp_gated = np.where(act_mask, sp + 1, 0)
                    te_sub = (
                        discrete_transfer_entropy_bits(dq[act_mask], sp[act_mask], dp[act_mask])
                        if int(np.sum(act_mask)) >= 2
                        else 0.0
                    )
                    te_gated = discrete_transfer_entropy_bits(dq, sp_gated, dp)
                    te_val = act_frac * max(te_sub, te_gated)
                else:
                    te_val = 0.0
        elif not w.active and not w.control:
            # Inactive fixed connection (strength <= 0.7): gated to 0.0 bits
            act_frac = 0.0
            mi_val = 0.0
            te_val = 0.0
        else:
            act_frac = 1.0
            mi_val = discrete_mi_bits(sq, dq)
            if has_valid_prev:
                te_val = discrete_transfer_entropy_bits(dq, prev_q[w.source], prev_q[w.target])
                if w.watched:
                    # Bidirectional watched control bundle (caudate <-> idea_k): include reverse TE
                    te_rev = discrete_transfer_entropy_bits(sq, prev_q[w.target], prev_q[w.source])
                    te_val = max(te_val, te_rev)
            else:
                te_val = 0.0

        info_val = max(mi_val, te_val)
        wire_mi_bits[wkey] = max(mi_val, wire_mi_bits.get(wkey, 0.0))
        wire_te_bits[wkey] = max(te_val, wire_te_bits.get(wkey, 0.0))
        wire_info_bits[wkey] = max(info_val, wire_info_bits.get(wkey, 0.0))
        wire_active_fraction[wkey] = max(act_frac, wire_active_fraction.get(wkey, 0.0))

        pair_key = f"{w.source}->{w.target}"
        wire_info_by_pair[pair_key] = max(info_val, wire_info_by_pair.get(pair_key, 0.0))

        wire_metrics.append(
            {
                "source": w.source,
                "target": w.target,
                "sign": int(w.sign),
                "learning": w.learning,
                "control": bool(w.control),
                "watched": bool(w.watched),
                "active": bool(w.active),
                "active_fraction": act_frac,
                "mi_bits": mi_val,
                "te_bits": te_val,
                "info_bits": info_val,
                "hero_level": float(w.level),
                "hero_strength": float(w.strength),
            }
        )

    # 7. 1D Empirical Density Barcodes over [0, 1] for all readout parts + learning connections
    def _make_barcode(
        vals_01: np.ndarray,
        raw_vals: np.ndarray,
        kind: str,
        is_learning: bool = False,
        learning_type: str | None = None,
    ) -> dict:
        clipped = np.clip(np.asarray(vals_01, dtype=float), 0.0, 1.0)
        counts, _ = np.histogram(clipped, bins=barcode_bins, range=(0.0, 1.0))
        hist = counts.astype(float) / float(n_veh)
        pos_h = hist[hist > 0.0]
        ent = 0.0 if pos_h.size <= 1 else float(-np.sum(pos_h * np.log2(pos_h)))
        if ent <= 1e-15:
            ent = 0.0
        return {
            "hist": hist,
            "hero_val": float(clipped[hero_index]),
            "hero_raw": float(raw_vals[hero_index]),
            "min": float(np.min(clipped)),
            "max": float(np.max(clipped)),
            "raw_min": float(np.min(raw_vals)),
            "raw_max": float(np.max(raw_vals)),
            "entropy_bits": ent,
            "is_learning": bool(is_learning),
            "learning_type": learning_type,
            "gate": float(ACTIVE_STRENGTH) if is_learning else None,
            "kind": kind,
        }

    barcodes: dict[str, dict] = {}
    for p in hero_diagram.parts:
        barcodes[p.name] = _make_barcode(
            part_levels[p.name],
            part_raw_vals[p.name],
            kind=p.kind,
            is_learning=False,
        )

    if d.energy is not None and "energy" not in barcodes:
        e_raw = np.array([float(v.energy or 0.0) for v in focal_vehicles], dtype=float)
        e_01 = np.clip(e_raw / max(float(d.energy.max), 1e-9), 0.0, 1.0)
        barcodes["energy"] = _make_barcode(e_01, e_raw, kind="energy", is_learning=False)

    seen_learn_names: set[str] = set()
    for u in d.units:
        t_name = u.module or u.array or u.name
        for j_in, c in enumerate(u.inputs):
            if c.learning is not None:
                s_name = c.source.split(".", 1)[0] if "." in c.source else c.source.split("[", 1)[0]
                base_lbl = f"{s_name}->{t_name}"
                lbl = base_lbl if base_lbl not in seen_learn_names else f"{base_lbl}[{j_in}]"
                seen_learn_names.add(lbl)
                s_vals = np.array([float(v.strengths[(u.name, j_in)]) for v in focal_vehicles], dtype=float)
                barcodes[lbl] = _make_barcode(
                    s_vals,
                    s_vals,
                    kind="learning",
                    is_learning=True,
                    learning_type=c.learning.type,
                )

    # 8. Lossless Sorted N x M Swarm Microstate Bit-Raster
    if phase_coords is not None:
        sec_key = np.nan_to_num(py_arr, nan=0.0, posinf=0.0, neginf=0.0)
        sort_order = np.lexsort((np.arange(n_veh), sec_key, raw_codeword_ids))
    else:
        sort_order = np.lexsort((np.arange(n_veh), raw_codeword_ids))

    raster_matrix = B[sort_order].copy()
    raster_codeword_ids = vehicle_codeword_ids[sort_order].copy()
    raster_raw_codeword_ids = raw_codeword_ids[sort_order].copy()
    hero_sorted_row = int(np.flatnonzero(sort_order == hero_index)[0])

    return {
        "hero_index": int(hero_index),
        "hero_label": hero.label,
        "n_vehicles": n_veh,
        "n_features": n_features,
        "hero_diagram": hero_diagram,
        "binary_matrix": B,
        "feature_names": list(feature_names),
        "feature_firing_rate": feature_firing_rate,
        "feature_entropy": feature_entropy,
        "part_firing_rate": part_firing_rate,
        "part_entropy": part_entropy,
        "grid_prob": grid_prob,
        "grid_entropy": grid_entropy,
        "grid_mean_level": grid_mean_level,
        "codebook": codebook,
        "codebook_full": codebook_full,
        "n_unique_codewords": k_raw,
        "part_codeword_arcs": part_codeword_arcs,
        "joint_entropy_bits": joint_entropy_bits,
        "marginal_entropy_sum_bits": marginal_entropy_sum_bits,
        "total_correlation_bits": total_correlation_bits,
        "spatial_mi_bits": spatial_mi_bits,
        "wire_mi_bits": wire_mi_bits,
        "wire_te_bits": wire_te_bits,
        "wire_info_bits": wire_info_bits,
        "wire_active_fraction": wire_active_fraction,
        "wire_info_by_pair": wire_info_by_pair,
        "wire_metrics": wire_metrics,
        "barcodes": barcodes,
        "raster_matrix": raster_matrix,
        "raster_codeword_ids": raster_codeword_ids,
        "raster_raw_codeword_ids": raster_raw_codeword_ids,
        "raster_vehicle_indices": sort_order,
        "raster_unit_names": list(feature_names),
        "hero_sorted_row": hero_sorted_row,
        "vehicle_codeword_ids": vehicle_codeword_ids,
        "vehicle_raw_codeword_ids": raw_codeword_ids,
        "quantized_signals": quantized_signals,
    }


# ---------------------------------------------------------------------------
# Phase-Coordinate Helpers & Deterministic Experiment Rollout Engine
# ---------------------------------------------------------------------------


def _nearest_source_polar(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    th_vec: np.ndarray,
    sources: list[tuple[float, float]],
    arena,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_nearest, psi_nearest) for each vehicle relative to its nearest active source."""
    n = len(x_vec)
    if not sources:
        cx, cy = 0.5 * arena.width, 0.5 * arena.height
        return polar_coords(x_vec, y_vec, th_vec, (cx, cy), arena=arena)
    if len(sources) == 1:
        return polar_coords(x_vec, y_vec, th_vec, sources[0], arena=arena)

    r_mat = np.empty((n, len(sources)), dtype=float)
    psi_mat = np.empty((n, len(sources)), dtype=float)
    for k, sp in enumerate(sources):
        rk, psik = polar_coords(x_vec, y_vec, th_vec, sp, arena=arena)
        r_mat[:, k] = rk
        psi_mat[:, k] = psik
    best_k = np.argmin(r_mat, axis=1)
    idx = np.arange(n)
    return r_mat[idx, best_k], psi_mat[idx, best_k]


def compute_focal_phase_coords(
    experiment_id: str,
    focal_vehicles: Sequence[Vehicle],
    world,
    env_vehicles: Sequence[Vehicle] = (),
    time_now: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute instantaneous phase-space coordinates for a homogeneous focal swarm."""
    n = len(focal_vehicles)
    arena = world.arena
    x_vec = np.array([float(v.x) for v in focal_vehicles], dtype=float)
    y_vec = np.array([float(v.y) for v in focal_vehicles], dtype=float)
    th_vec = np.array([float(v.heading) for v in focal_vehicles], dtype=float)
    sp_vec = np.array([float(v.speed) for v in focal_vehicles], dtype=float)
    energy_vec = np.array(
        [float(v.energy) if v.energy is not None else np.nan for v in focal_vehicles],
        dtype=float,
    )

    vdef = focal_vehicles[0].definition
    _, sources = extract_primary_sources(vdef, world, time=time_now)

    if experiment_id == "v9_shy":
        # Relative target distance & bearing (r_target, psi_bearing) to moving stimulus agents
        targets = list(env_vehicles)
        if targets:
            tx = np.array([float(t.x) for t in targets], dtype=float)
            ty = np.array([float(t.y) for t in targets], dtype=float)
            r_mat = np.empty((n, len(targets)), dtype=float)
            psi_mat = np.empty((n, len(targets)), dtype=float)
            score_mat = np.empty((n, len(targets)), dtype=float)
            for k in range(len(targets)):
                rk, psik = polar_coords(x_vec, y_vec, th_vec, (tx[k], ty[k]), arena=arena)
                r_mat[:, k] = rk
                psi_mat[:, k] = psik
                # Prefer targets visible near retina's forward cone when similarly close
                in_cone = (np.abs(psik) <= math.radians(35.0)) & (rk <= 34.0)
                score_mat[:, k] = np.where(in_cone, rk * 0.55, rk)
            best_k = np.argmin(score_mat, axis=1)
            idx = np.arange(n)
            r_vec = r_mat[idx, best_k]
            psi_vec = psi_mat[idx, best_k]
        else:
            r_vec, psi_vec = _nearest_source_polar(x_vec, y_vec, th_vec, sources, arena)
        dr_vec = -sp_vec * np.cos(psi_vec)
        return {
            "r": r_vec,
            "psi": psi_vec,
            "dr": dr_vec,
            "energy": energy_vec,
            "phase_x": r_vec,
            "phase_y": psi_vec,
        }

    if experiment_id == "v7_wary_moth":
        red_srcs = (
            [(float(s.position[0]), float(s.position[1])) for s in world.stimuli["red"].sources]
            if "red" in world.stimuli and world.stimuli["red"].sources
            else sources
        )
        r_vec, psi_vec = _nearest_source_polar(x_vec, y_vec, th_vec, red_srcs, arena)
        dr_vec = -sp_vec * np.cos(psi_vec)
        w_red = np.array([float(v.strength("sees_red", "danger")) for v in focal_vehicles], dtype=float)
        w_white = np.array([float(v.strength("sees_white", "danger")) for v in focal_vehicles], dtype=float)
        return {
            "r": r_vec,
            "psi": psi_vec,
            "dr": dr_vec,
            "energy": energy_vec,
            "w_red": w_red,
            "w_white": w_white,
            "phase_x": r_vec,
            "phase_y": w_red,
        }

    if experiment_id == "v12_daydreamer":
        r_vec, psi_vec = _nearest_source_polar(x_vec, y_vec, th_vec, sources, arena)
        dr_vec = -sp_vec * np.cos(psi_vec)
        idea_names = [f"idea_{k}" for k in range(8) if any(u.name == f"idea_{k}" for u in vdef.units)]
        idea_mat = (
            np.array([[float(v.unit_outputs[nm]) for nm in idea_names] for v in focal_vehicles], dtype=float)
            if idea_names
            else np.zeros((n, 8), dtype=float)
        )
        ring = compute_ring_order_parameter(idea_mat)
        z_vec = np.asarray(ring["Z"], dtype=complex)
        re_z = np.real(z_vec)
        im_z = np.imag(z_vec)
        caudate_off = np.array([float(v.unit_outputs.get("caudate", 0.0)) for v in focal_vehicles], dtype=float)
        return {
            "r": r_vec,
            "psi": psi_vec,
            "dr": dr_vec,
            "energy": energy_vec,
            "Z": z_vec,
            "R": np.asarray(ring["R"], dtype=float),
            "Phi": np.asarray(ring["Phi"], dtype=float),
            "total_activity": np.asarray(ring["total_activity"], dtype=float),
            "caudate_offset": caudate_off,
            "phase_x": re_z,
            "phase_y": im_z,
        }

    # v4a_orbiter, v5_moth, v5_forager (and generic fallback)
    r_vec, psi_vec = _nearest_source_polar(x_vec, y_vec, th_vec, sources, arena)
    dr_vec = -sp_vec * np.cos(psi_vec)
    if experiment_id == "v5_moth":
        px, py = r_vec, dr_vec
    elif experiment_id == "v5_forager":
        px, py = r_vec, energy_vec
    else:
        px, py = r_vec, psi_vec
    return {
        "r": r_vec,
        "psi": psi_vec,
        "dr": dr_vec,
        "energy": energy_vec,
        "phase_x": px,
        "phase_y": py,
    }


def _build_phase_background(experiment_id: str, focal_def: VehicleDefinition, world) -> dict:
    """Precompute geometric manifolds, nullclines, thresholds, and FTLE fields per experiment."""
    if experiment_id == "v4a_orbiter":
        pf = polar_field(focal_def, world, r_grid=np.linspace(2.0, 46.0, 64), psi_grid=np.linspace(-math.pi, math.pi, 65))
        eq = find_fixed_points_and_cycles(focal_def, world, r_bounds=(2.0, 46.0))
        ftle = compute_ftle_and_basins(focal_def, world, nx=96, ny=96, horizon=11.0, dt=0.05)
        return {
            "kind": "polar_limit_cycle",
            "polar_field": pf,
            "fixed_points": eq["fixed_points"],
            "limit_cycle": eq["limit_cycle"],
            "ftle": ftle,
            "r_limits": (0.0, 48.0),
            "psi_limits": (-math.pi, math.pi),
        }

    if experiment_id == "v5_moth":
        # Two eyes sum >= 18 for pain, >= 10 for near in worlds/two-lights.yaml (I=10, falloff=15)
        stim, _ = resolve_primary_source(focal_def, world)
        src = world.stimuli[stim].sources[0] if stim and world.stimuli[stim].sources else None
        inten = float(src.intensity) if src else 10.0
        fall = float(src.falloff) if src else 15.0
        r_pain = fall * math.sqrt(max(0.0, 2.0 * inten / 18.0 - 1.0))
        r_near = fall * math.sqrt(max(0.0, 2.0 * inten / 10.0 - 1.0))
        v_max = max((a.max_speed for a in focal_def.actuators), default=6.0)
        return {
            "kind": "burn_catastrophe",
            "pain_radius": r_pain,
            "near_radius": r_near,
            "dr_zero": 0.0,
            "seek_speed": -0.75 * v_max,
            "flee_speed": 0.75 * v_max,
            "r_limits": (0.0, 56.0),
            "dr_limits": (-v_max * 1.05, v_max * 1.05),
        }

    if experiment_id == "v5_forager":
        stim, _ = resolve_primary_source(focal_def, world)
        src = world.stimuli[stim].sources[0] if stim and world.stimuli[stim].sources else None
        inten = float(src.intensity) if src else 10.0
        fall = float(src.falloff) if src else 15.0
        r_sees = fall * math.sqrt(max(0.0, 2.0 * inten / 3.0 - 1.0))
        e_cfg = focal_def.energy
        e_max = float(e_cfg.max) if e_cfg else 100.0
        # Balance radius where recharge rate = basal + cruise consumption (~4.25/s)
        drain_est = (e_cfg.basal + e_cfg.consumption * 4.5) if e_cfg else 4.25
        gain_0 = (e_cfg.charge_gain * inten) if e_cfg else 20.0
        r_balance = fall * math.sqrt(max(0.0, gain_0 / max(drain_est, 1e-6) - 1.0))
        return {
            "kind": "forager_relaxation",
            "energy_low": 30.0,
            "energy_high": 90.0,
            "energy_max": e_max,
            "sees_radius": r_sees,
            "balance_radius": r_balance,
            "r_limits": (0.0, 62.0),
            "energy_limits": (0.0, e_max),
        }

    if experiment_id == "v7_wary_moth":
        red_src = world.stimuli["red"].sources[0] if "red" in world.stimuli and world.stimuli["red"].sources else None
        temp_src = (
            world.stimuli["temperature"].sources[0]
            if "temperature" in world.stimuli and world.stimuli["temperature"].sources
            else None
        )
        r_sees_red = (
            float(red_src.falloff) * math.sqrt(max(0.0, float(red_src.intensity) / 2.5 - 1.0))
            if red_src
            else 26.0
        )
        r_pain = (
            float(temp_src.falloff) * math.sqrt(max(0.0, float(temp_src.intensity) / 5.0 - 1.0))
            if temp_src
            else 4.0
        )
        return {
            "kind": "mnemotrix_bifurcation",
            "critical_weight": float(ACTIVE_STRENGTH),
            "pain_radius": r_pain,
            "sees_red_radius": r_sees_red,
            "r_limits": (0.0, 62.0),
            "w_limits": (0.0, 1.02),
        }

    if experiment_id == "v9_shy":
        cam = next((s for s in focal_def.sensors if s.camera is not None), None)
        cam_range = float(cam.camera.range) if cam and cam.camera else 30.0
        fov_half = math.radians(0.5 * float(cam.field_of_view)) if cam else math.radians(30.0)
        fovea_half = fov_half * (4.0 / 24.0)  # middle 4 of 24 columns -> +-5 deg
        return {
            "kind": "shy_symmetry_cone",
            "camera_range": cam_range,
            "fov_half_rad": fov_half,
            "fovea_half_rad": fovea_half,
            "r_limits": (0.0, 58.0),
            "psi_limits": (-math.pi, math.pi),
        }

    if experiment_id == "v12_daydreamer":
        angles = 2.0 * np.pi * np.arange(8, dtype=float) / 8.0
        chords = [((k, (k + 1) % 8), (k, (k + 3) % 8)) for k in range(8)]
        flat_chords = [pair for grp in chords for pair in grp]
        return {
            "kind": "kuramoto_ring_disc",
            "n_ideas": 8,
            "idea_angles": angles,
            "idea_labels": [f"idea_{k}" for k in range(8)],
            "excitatory_chords": flat_chords,
            "single_idea_radius": 1.0 / 8.0,
            "pair_idea_radius": float(math.cos(math.pi / 8.0) / 4.0),
            "z_limits": (-1.05, 1.05),
        }

    return {"kind": "generic"}


def simulate_experiment_rollout(
    experiment_id: str,
    steps: int = 400,
    stride: int = 2,
    seed: int = 0,
) -> dict:
    """Run one pure single-species experiment rollout and record full Strogatz x Shannon state.

    Validates Axiom 1 before stepping ``Simulation(scenario)``, separates focal vehicles from
    any moving stimulus targets (e.g. ``Chaser``/``Walker`` in ``v9_shy``), and captures
    synchronized trajectories, phase coordinates, environmental stimuli, phase-portrait
    backgrounds, and exact ``shannon_snapshots`` at sampled frames.
    """
    spec = validate_single_experiment_axiom(experiment_id)
    steps = max(1, int(steps))
    stride = max(1, int(stride))

    sc_path = (REPO_ROOT / spec.scenario_path).resolve()
    base_sc = load_scenario(sc_path)
    scenario = replace(
        base_sc,
        simulation=SimulationSettings(dt=base_sc.simulation.dt, seed=int(seed)),
    )
    validate_single_experiment_axiom(spec, scenario=scenario)

    sim = Simulation(scenario)
    expected_def = _cached_load_definition(spec.focal_vehicle_path)
    focal_indices = select_focal_indices(sim.vehicles, expected_def)
    focal_vehicles = [sim.vehicles[i] for i in focal_indices]
    validate_single_experiment_axiom(spec, vehicles=focal_vehicles, scenario=scenario)

    focal_set = set(focal_indices)
    env_indices = [i for i in range(len(sim.vehicles)) if i not in focal_set]
    env_vehicles = [sim.vehicles[i] for i in env_indices]

    n_focal = len(focal_vehicles)
    n_env = len(env_vehicles)
    arena = sim.world.arena

    frame_steps = list(range(0, steps, stride))
    if not frame_steps:
        frame_steps = [0]
    n_frames = len(frame_steps)
    sample_set = {s: idx for idx, s in enumerate(frame_steps)}

    t_arr = np.empty(n_frames, dtype=float)
    x_arr = np.empty((n_frames, n_focal), dtype=float)
    y_arr = np.empty((n_frames, n_focal), dtype=float)
    ux_arr = np.empty((n_frames, n_focal), dtype=float)
    uy_arr = np.empty((n_frames, n_focal), dtype=float)
    heading_arr = np.empty((n_frames, n_focal), dtype=float)
    speed_arr = np.empty((n_frames, n_focal), dtype=float)
    turn_arr = np.empty((n_frames, n_focal), dtype=float)
    r_arr = np.empty((n_frames, n_focal), dtype=float)
    psi_arr = np.empty((n_frames, n_focal), dtype=float)
    dr_arr = np.empty((n_frames, n_focal), dtype=float)
    energy_arr = np.full((n_frames, n_focal), np.nan, dtype=float)
    phase_x_arr = np.empty((n_frames, n_focal), dtype=float)
    phase_y_arr = np.empty((n_frames, n_focal), dtype=float)
    codeword_id_arr = np.empty((n_frames, n_focal), dtype=int)

    extra_series: dict[str, np.ndarray] = {}
    if spec.experiment_id == "v7_wary_moth":
        extra_series["w_red"] = np.empty((n_frames, n_focal), dtype=float)
        extra_series["w_white"] = np.empty((n_frames, n_focal), dtype=float)
    elif spec.experiment_id == "v12_daydreamer":
        extra_series["Z"] = np.empty((n_frames, n_focal), dtype=complex)
        extra_series["R"] = np.empty((n_frames, n_focal), dtype=float)
        extra_series["Phi"] = np.empty((n_frames, n_focal), dtype=float)
        extra_series["caudate_offset"] = np.empty((n_frames, n_focal), dtype=float)

    env_x = np.empty((n_frames, n_env), dtype=float)
    env_y = np.empty((n_frames, n_env), dtype=float)
    env_ux = np.empty((n_frames, n_env), dtype=float)
    env_uy = np.empty((n_frames, n_env), dtype=float)
    env_heading = np.empty((n_frames, n_env), dtype=float)

    cur_ux = np.array([float(v.x) for v in focal_vehicles], dtype=float)
    cur_uy = np.array([float(v.y) for v in focal_vehicles], dtype=float)
    cur_env_ux = np.array([float(v.x) for v in env_vehicles], dtype=float)
    cur_env_uy = np.array([float(v.y) for v in env_vehicles], dtype=float)

    # Record world stimulus sources and active-schedule state per frame
    stimuli_sources: list[dict] = []
    for stim_name, st in sim.world.stimuli.items():
        for s_idx, src in enumerate(st.sources):
            stimuli_sources.append(
                {
                    "stimulus": stim_name,
                    "index": s_idx,
                    "position": (float(src.position[0]), float(src.position[1])),
                    "intensity": float(src.intensity),
                    "falloff": float(src.falloff),
                    "has_schedule": src.schedule is not None,
                    "_src_ref": src,
                }
            )
    source_active_arr = np.ones((n_frames, len(stimuli_sources)), dtype=bool)

    shannon_snapshots: list[dict] = []
    prev_step_snap: dict | None = None

    for step in range(steps):
        prev_fx = np.array([v.x for v in focal_vehicles], dtype=float)
        prev_fy = np.array([v.y for v in focal_vehicles], dtype=float)
        prev_ex = np.array([v.x for v in env_vehicles], dtype=float)
        prev_ey = np.array([v.y for v in env_vehicles], dtype=float)

        sim.step()

        ddx, ddy = arena.delta(
            np.array([v.x for v in focal_vehicles], dtype=float) - prev_fx,
            np.array([v.y for v in focal_vehicles], dtype=float) - prev_fy,
        )
        cur_ux += ddx
        cur_uy += ddy

        if n_env > 0:
            eddx, eddy = arena.delta(
                np.array([v.x for v in env_vehicles], dtype=float) - prev_ex,
                np.array([v.y for v in env_vehicles], dtype=float) - prev_ey,
            )
            cur_env_ux += eddx
            cur_env_uy += eddy

        f_idx = sample_set.get(step)
        if f_idx is not None:
            pcoords = compute_focal_phase_coords(
                spec.experiment_id,
                focal_vehicles,
                sim.world,
                env_vehicles=env_vehicles,
                time_now=sim.time,
            )
            snap = compute_shannon_snapshot(
                focal_vehicles,
                prev_snapshot=prev_step_snap,
                phase_coords=(pcoords["phase_x"], pcoords["phase_y"]),
                hero_index=0,
            )
            shannon_snapshots.append(snap)
            prev_step_snap = snap

            t_arr[f_idx] = sim.time
            x_arr[f_idx] = [v.x for v in focal_vehicles]
            y_arr[f_idx] = [v.y for v in focal_vehicles]
            ux_arr[f_idx] = cur_ux
            uy_arr[f_idx] = cur_uy
            heading_arr[f_idx] = wrap_pi(np.array([v.heading for v in focal_vehicles], dtype=float))
            speed_arr[f_idx] = [v.speed for v in focal_vehicles]
            turn_arr[f_idx] = [v.turn_rate for v in focal_vehicles]
            r_arr[f_idx] = pcoords["r"]
            psi_arr[f_idx] = pcoords["psi"]
            dr_arr[f_idx] = pcoords["dr"]
            energy_arr[f_idx] = pcoords["energy"]
            phase_x_arr[f_idx] = pcoords["phase_x"]
            phase_y_arr[f_idx] = pcoords["phase_y"]
            codeword_id_arr[f_idx] = snap["vehicle_codeword_ids"]

            for k_ex, arr_ex in extra_series.items():
                arr_ex[f_idx] = pcoords[k_ex]

            if n_env > 0:
                env_x[f_idx] = [v.x for v in env_vehicles]
                env_y[f_idx] = [v.y for v in env_vehicles]
                env_ux[f_idx] = cur_env_ux
                env_uy[f_idx] = cur_env_uy
                env_heading[f_idx] = wrap_pi(np.array([v.heading for v in env_vehicles], dtype=float))

            for s_i, s_meta in enumerate(stimuli_sources):
                source_active_arr[f_idx, s_i] = bool(s_meta["_src_ref"].active(sim.time))
        else:
            # Only compute quantized signals if the NEXT step will be a sampled frame
            if (step + 1) in sample_set:
                prev_step_snap = compute_shannon_snapshot(focal_vehicles, prev_snapshot=None, hero_index=0)

    clean_sources = [
        {k: v for k, v in s_meta.items() if k != "_src_ref"}
        for s_meta in stimuli_sources
    ]
    _, primary_sources = extract_primary_sources(expected_def, sim.world)
    phase_background = _build_phase_background(spec.experiment_id, expected_def, sim.world)

    # Identify an information-rich representative frame (after initial transient) for static posters
    warmup_f = min(n_frames - 1, max(1, n_frames // 3))
    scores = [
        shannon_snapshots[f]["joint_entropy_bits"]
        + 0.5 * shannon_snapshots[f]["total_correlation_bits"]
        + 0.5 * shannon_snapshots[f]["spatial_mi_bits"]
        for f in range(n_frames)
    ]
    best_frame = warmup_f + int(np.argmax(scores[warmup_f:])) if n_frames > warmup_f else n_frames - 1

    return {
        "spec": spec,
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "subtitle": spec.subtitle,
        "phase_axes": spec.phase_axes,
        "scenario": scenario,
        "world": sim.world,
        "focal_definition": expected_def,
        "n_vehicles": n_focal,
        "n_frames": n_frames,
        "dt": float(sim.dt),
        "stride": stride,
        "seed": int(seed),
        "frame_steps": np.asarray(frame_steps, dtype=int),
        "t": t_arr,
        "x": x_arr,
        "y": y_arr,
        "ux": ux_arr,
        "uy": uy_arr,
        "heading": heading_arr,
        "speed": speed_arr,
        "turn_rate": turn_arr,
        "r": r_arr,
        "psi": psi_arr,
        "dr": dr_arr,
        "energy": energy_arr,
        "phase_x": phase_x_arr,
        "phase_y": phase_y_arr,
        "vehicle_codeword_ids": codeword_id_arr,
        "hero_index": 0,
        "hero_label": focal_vehicles[0].label,
        "focal_labels": [v.label for v in focal_vehicles],
        "sources": primary_sources,
        "stimuli_sources": clean_sources,
        "source_active": source_active_arr,
        "env_targets": {
            "n_targets": n_env,
            "labels": [v.label for v in env_vehicles],
            "types": [f"Vehicle {v.definition.general.type}: {v.definition.general.name}" for v in env_vehicles],
            "sizes": [(float(v.half_width * 2.0), float(v.half_length * 2.0)) for v in env_vehicles],
            "x": env_x,
            "y": env_y,
            "ux": env_ux,
            "uy": env_uy,
            "heading": env_heading,
        },
        "phase_background": phase_background,
        "shannon_snapshots": shannon_snapshots,
        "best_poster_frame": int(best_frame),
        **extra_series,
    }
