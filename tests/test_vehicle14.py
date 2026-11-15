"""Acceptance criteria of specs/vehicle-14.md."""

import time
from functools import lru_cache

import numpy as np
import pytest

from braitenberg.definition import load_definition
from braitenberg.diagram import definition_svg
from braitenberg.scenario import load_scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Vehicle
from conftest import REPO, run_python

DT = 0.01
OPTIMIST = REPO / "vehicles" / "vehicle14-optimist.yaml"
SCENARIO = REPO / "scenarios" / "vehicle14-optimist.yaml"
SEEDS = range(5)
SECONDS = 600
UPPER = ("p_upper", "p_food")
LOWER = ("p_lower", "p_food")
FORK_UPPER = ("p_fork", "p_upper")
FORK_LOWER = ("p_fork", "p_lower")
WIRES = (UPPER, LOWER, FORK_UPPER, FORK_LOWER)
UNITS = ("at_fork", "deciding", "look_upper", "look_lower", "p_upper", "p_lower", "pleasing",
         "go_upper", "go_lower", "in_upper", "in_lower", "smells")


def scenario(variant="default", seed=0):
    """The default scenario; "control" is never pleased."""
    s = load_scenario(SCENARIO)
    s.simulation.seed = seed
    if variant == "control":
        pleasing = next(u for u in s.vehicles[0].definition.units if u.name == "pleasing")
        pleasing.inputs = [c for c in pleasing.inputs if c.source != "p_food"]
    return s


class Track:
    """Per step: time, the strengths of WIRES, the outputs of UNITS, and whether food is on."""

    def __init__(self, sim, seconds, move_food_at=None):
        v = sim.vehicles[0]
        food = sim.world.stimuli["food"]
        n = round(seconds / DT)
        self.time = np.empty(n)
        self.wire = {w: np.empty(n) for w in WIRES}
        self.out = {u: np.empty(n, dtype=bool) for u in UNITS}
        self.food_on = np.empty(n, dtype=bool)
        for k in range(n):
            if move_food_at is not None and k == round(move_food_at / DT):
                food.sources[0].position = (52.0, 7.0)
            sim.step()
            self.time[k] = sim.time
            for w in WIRES:
                self.wire[w][k] = v.strength(*w)
            for u in UNITS:
                self.out[u][k] = v.unit_outputs[u] > 0
            self.food_on[k] = food.sources[0].active(sim.time)

    def onsets(self, unit):
        x = self.out[unit]
        return list(np.flatnonzero(x[1:] & ~x[:-1]) + 1)

    def laps(self):
        """(start, entry, lane) per lap: a lap starts after the vehicle entered a lane, or at the
        start of the run, and ends when it enters the next one. Entries within 5 s of the one
        before, a mark flickering at the edge of its range, are the same entry."""
        entries = sorted([(k, "upper") for k in self.onsets("in_upper")] +
                         [(k, "lower") for k in self.onsets("in_lower")])
        laps, start, last = [], 0, -10 ** 9
        for k, lane in entries:
            if k - last < round(5 / DT):
                continue
            laps.append((start, k, lane))
            start = last = k
        return laps

    def learned(self, wire):
        """The first step at which the wire exceeds 0.7, or None."""
        above = np.flatnonzero(self.wire[wire] > 0.7)
        return int(above[0]) if len(above) else None

    def any_on(self, unit, a, b):
        return bool(self.out[unit][a:b].any())


@lru_cache(maxsize=None)
def run(variant, seed, seconds=SECONDS, move_food_at=None):
    return Track(Simulation(scenario(variant, seed)), seconds, move_food_at)


def meals(track):
    return len(track.onsets("smells"))


def confident_laps(track):
    """Laps that start with the upper wire active."""
    return [lap for lap in track.laps() if track.wire[UPPER][lap[0]] > 0.7]


# 1. Explores first
@pytest.mark.slow
def test_explores_first():
    lower_before = 0
    for seed in SEEDS:
        track = run("default", seed)
        since = track.learned(UPPER)
        assert since is not None
        assert not track.any_on("go_upper", 0, since) and not track.any_on("go_lower", 0, since)
        lower_before += any(lane == "lower" for _, entry, lane in track.laps() if entry < since)
    assert lower_before >= 2


# 2. Learns the road
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_learns_the_road(seed):
    track = run("default", seed)
    assert track.learned(FORK_UPPER) is not None
    assert track.learned(UPPER) is not None and track.time[track.learned(UPPER)] < 300
    assert track.learned(LOWER) is None


# 3. Scans one at a time
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_scans_one_at_a_time(seed):
    track = run("default", seed)
    deciding = track.out["deciding"]
    assert not (track.out["p_upper"] & track.out["p_lower"] & deciding).any()
    for start, entry, _ in confident_laps(track):
        on = next(k for k in track.onsets("deciding") if start <= k < entry)
        chosen = next(k for k in track.onsets("go_upper") if on <= k < entry)
        assert track.any_on("look_lower", on, chosen) and track.any_on("look_upper", on, chosen)


# 4. Chooses the pleasing lane
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_chooses_the_pleasing_lane(seed):
    track = run("default", seed)
    laps = confident_laps(track)
    assert laps
    for start, entry, _ in laps:
        on = next(k for k in track.onsets("deciding") if start <= k < entry)
        assert track.any_on("go_upper", on, on + round(2 / DT))
        assert not track.any_on("go_lower", start, entry)
    assert sum(lane == "upper" for *_, lane in laps) >= 0.9 * len(laps)


# 5. Even when it is not there
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_goes_even_when_the_food_is_not_there(seed):
    track = run("default", seed)
    absent = [lap for lap in confident_laps(track) if not track.food_on[lap[1]]]
    assert absent
    assert sum(lane == "upper" for *_, lane in absent) >= 0.9 * len(absent)


# 6. Eats more
@pytest.mark.slow
def test_eats_more_than_the_control():
    optimist = sum(meals(run("default", seed)) for seed in SEEDS)
    control = sum(meals(run("control", seed)) for seed in SEEDS)
    assert optimist >= 1.5 * control


# 7. Follows the food
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_follows_the_food(seed):
    track = run("default", seed, 900, 300)
    moved = round(300 / DT)
    after = [lap for lap in track.laps() if lap[1] > moved]
    dropped = next(k for k in range(moved, len(track.time)) if track.wire[UPPER][k] <= 0.7)
    assert dropped <= after[4][1]
    assert track.learned(LOWER) is not None
    assert all(lane == "lower" for *_, lane in after[-3:])


# 8. The circuit
def optimist(strengths=()):
    v = Vehicle(load_definition(OPTIMIST), 50, 20, 0, dt=DT)
    for wire, s in dict(strengths).items():
        unit, source = wire[1], wire[0]
        k = next(j for j, c in enumerate(v.definition.units[v.unit_names.index(unit)].inputs)
                 if c.source == source and c.learning)
        v.strengths[(unit, k)] = s
    return v


def force(v, outputs):
    """Make units' outputs at this step what `outputs` says, whatever they computed."""
    rows = len(v.activations)
    for name, value in outputs.items():
        k = v.unit_names.index(name)
        v.activations[(v.t - v.steps[k]) % rows, k] = value
        v.fixed_activations[(v.t - v.steps[k]) % rows, k] = value


NOTHING = {"fork_eye": 0.0, "blue_eye": 0.0, "red_eye": 0.0, "nose": 0.0, "ir_left": 0.0,
           "ir_right": 0.0}


def play(v, seconds, forced=lambda t: {}):
    """Step v for `seconds`, forcing outputs by time; returns each unit's output per step."""
    rng = np.random.default_rng(0)
    seen = {u: [] for u in v.unit_names}
    for n in range(round(seconds / DT)):
        force(v, forced(n * DT))
        v.step(NOTHING, DT, rng)
        for u in v.unit_names:
            seen[u].append(v.unit_outputs[u] > 0)
    return {u: np.array(x) for u, x in seen.items()}


def held(name, until):
    return lambda t: {name: 1.0} if t < until else {}


def onsets(x):
    return list(np.flatnonzero(x[1:] & ~x[:-1]) + 1)


def test_looks_take_turns():
    v = optimist({FORK_UPPER: 1.0, FORK_LOWER: 1.0})
    seen = play(v, 4, held("at_fork", 4))
    assert seen["deciding"][50:].all()
    assert len(onsets(seen["look_upper"])) >= 2 and len(onsets(seen["look_lower"])) >= 1
    assert not (seen["look_upper"] & seen["look_lower"]).any()
    assert not (seen["p_upper"] & seen["p_lower"]).any()
    assert len(onsets(seen["p_upper"])) >= 2 and len(onsets(seen["p_lower"])) >= 1


def test_chooses_the_lane_that_leads_to_food():
    v = optimist({FORK_UPPER: 1.0, FORK_LOWER: 1.0, UPPER: 1.0})
    seen = play(v, 4, held("at_fork", 0.2))
    pulses = onsets(seen["pleasing"])
    assert pulses
    for k in pulses:
        assert seen["look_upper"][k - 10:k + 1].any() and not seen["look_lower"][k]
        end = k + int(np.argmin(seen["pleasing"][k:]))
        assert (end - k) * DT <= 0.1
    chose = onsets(seen["go_upper"])[0]
    assert not seen["go_lower"].any()
    later = chose + round(0.2 / DT)
    assert not seen["deciding"][later] and not seen["look_upper"][later]
    assert not seen["look_lower"][later]


def test_without_a_pleasing_lane_it_does_not_choose():
    v = optimist({FORK_UPPER: 1.0, FORK_LOWER: 1.0})
    seen = play(v, 5, lambda t: {"at_fork": 1.0} if t < 0.2 else
                {"in_lower": 1.0} if 4 <= t < 4.2 else {})
    assert seen["deciding"][round(3.9 / DT)]
    assert not seen["deciding"][-1]
    assert not seen["go_upper"].any() and not seen["go_lower"].any()


def test_go_holds_until_a_lane_or_its_time():
    v = optimist()
    seen = play(v, 4, lambda t: {"choose_upper": 1.0} if t < 0.05 else
                {"in_upper": 1.0} if 1 <= t < 1.2 else {})
    on = np.flatnonzero(seen["go_upper"])
    assert on[0] * DT < 0.2 and 0.9 < on[-1] * DT < 1.2
    v = optimist()
    seen = play(v, 4, lambda t: {"choose_upper": 1.0} if t < 0.05 else {})
    assert 1.4 < np.flatnonzero(seen["go_upper"])[-1] * DT < 1.8


# 9. Thoughts are not visits
def test_a_thought_starts_no_wait():
    v = optimist({FORK_UPPER: 1.0, UPPER: 0.8})
    seen = play(v, 8, lambda t: {"p_fork": 1.0, "look_upper": 1.0} if t < 0.5 else {})
    assert seen["p_upper"].any()
    assert v.strength(*UPPER) == pytest.approx(0.8)   # no wait started, so none failed


def test_a_visit_starts_a_wait():
    v = optimist({UPPER: 0.8})
    play(v, 8, lambda t: {"in_upper": 1.0} if t < 0.5 else {})
    assert v.strength(*UPPER) == pytest.approx(0.7)   # the visit predicted food, none came


def test_the_fork_wire_learns_from_visits_only():
    v = optimist({FORK_UPPER: 1.0})
    play(v, 4, lambda t: {"p_fork": 1.0} if t < 0.2 else
         {"look_upper": 1.0} if 1 <= t < 1.5 else {})
    assert v.strength(*FORK_UPPER) == pytest.approx(1.0)
    v = optimist({FORK_UPPER: 0.0})
    play(v, 4, lambda t: {"p_fork": 1.0} if t < 0.2 else
         {"look_upper": 1.0} if 1 <= t < 1.5 else {})
    assert v.strength(*FORK_UPPER) == pytest.approx(0.0)
    play(v, 12, lambda t: {"p_fork": 1.0} if t < 0.2 else
         {"in_upper": 1.0} if 3 <= t < 3.5 else {})
    assert v.strength(*FORK_UPPER) == pytest.approx(0.4)


# 10. Headless and fast
def test_runs_without_pygame():
    result = run_python(
        "import sys\n"
        "from braitenberg.scenario import load_scenario\n"
        "from braitenberg.sim import Simulation\n"
        "Simulation(load_scenario('scenarios/vehicle14-optimist.yaml')).run(100)\n"
        "assert 'pygame' not in sys.modules\n")
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_runs_faster_than_ten_times_real_time():
    sim = Simulation(load_scenario(SCENARIO))
    start = time.perf_counter()
    sim.run(round(SECONDS / DT))
    assert time.perf_counter() - start < 60


# Definition and diagram
def test_definition_loads_and_draws():
    d = load_definition(OPTIMIST)
    assert d.general.type == "14"
    svg = definition_svg(d)
    for name in ("deciding", "pleasing", "go_upper", "p_food"):
        assert name in svg


def test_food_moves_in_the_second_scenario():
    s = load_scenario(REPO / "scenarios" / "vehicle14-optimist-moved.yaml")
    upper, lower = s.world.stimuli["food"].sources
    assert upper.position[1] > 20 > lower.position[1]
    assert upper.active(100) and not lower.active(100)
    assert lower.active(200) and not upper.active(200)


# The impatient vehicle (specs/vehicle-14.md, *Second example: the impatient vehicle*)

IMPATIENT = REPO / "vehicles" / "vehicle14-impatient.yaml"
DOORSTEP = REPO / "scenarios" / "vehicle14-impatient.yaml"
COMES_BACK = ("p_wait", "p_carrier")
BRINGS_FOOD = ("p_carrier", "p_fed")
WAITING_UNITS = ("carrier", "fed", "go_tap", "tapping")


@lru_cache(maxsize=None)
def wait(control=False, seconds=SECONDS):
    """Per step: time, the two wires' strengths, the outputs of WAITING_UNITS, and the vehicle's
    place and heading."""
    s = load_scenario(DOORSTEP)
    if control:
        pleasing = next(u for u in s.vehicles[0].definition.units if u.name == "pleasing")
        pleasing.inputs = [c for c in pleasing.inputs if c.source != "p_fed"]
    sim = Simulation(s)
    v = sim.vehicles[0]
    n = round(seconds / DT)
    out = {u: np.empty(n, dtype=bool) for u in WAITING_UNITS}
    wires = {w: np.empty(n) for w in (COMES_BACK, BRINGS_FOOD)}
    place = np.empty((n, 3))
    for k in range(n):
        sim.step()
        for u in WAITING_UNITS:
            out[u][k] = v.unit_outputs[u] > 0
        for w in wires:
            wires[w][k] = v.strength(*w)
        place[k] = v.x, v.y, np.degrees(v.heading)
    return out, wires, place


def rises(x):
    return list(np.flatnonzero(x[1:] & ~x[:-1]) + 1)


# I1. Learns that the carrier comes back and brings food
@pytest.mark.slow
def test_impatient_learns_before_it_taps_at_nothing():
    out, wires, _ = wait()
    passes = rises(out["carrier"])
    learned = max(int(np.flatnonzero(wires[w] > 0.7)[0]) for w in wires)
    assert learned < passes[2]
    assert not out["go_tap"][:learned].any()


# I2. Taps at nothing while it waits
@pytest.mark.slow
def test_impatient_taps_while_it_waits():
    out, _, _ = wait()
    passes = rises(out["carrier"])
    assert not (out["go_tap"] & out["carrier"]).any()
    taps = rises(out["go_tap"])
    for a, b in zip(passes[2:], passes[3:]):
        assert sum(a < k < b for k in taps) >= 3


# I3. Tapping does not bring the carrier sooner
@pytest.mark.slow
def test_tapping_does_not_hasten_the_carrier():
    impatient, _, _ = wait()
    control, _, _ = wait(control=True)
    a, b = rises(impatient["carrier"]), rises(control["carrier"])
    assert len(a) == len(b) and all(abs(x - y) * DT <= 0.2 for x, y in zip(a, b))
    assert len(rises(impatient["fed"])) == len(rises(control["fed"]))
    assert not control["go_tap"].any()


# I4. It stays in its pen, facing the path
@pytest.mark.slow
def test_impatient_stays_in_its_pen():
    _, _, place = wait()
    x, y, heading = place.T
    assert ((45 < x) & (x < 55) & (17 < y) & (y < 26)).all()
    assert (np.abs((heading - 90 + 180) % 360 - 180) < 15).all()


# I5. Its thoughts do not teach it otherwise
@pytest.mark.slow
def test_taps_at_nothing_do_not_unlearn():
    _, wires, _ = wait()
    learned = int(np.flatnonzero(wires[BRINGS_FOOD] > 0.7)[0])
    assert (wires[BRINGS_FOOD][learned:] > 0.7).all()
    assert (wires[COMES_BACK][learned:] > 0.7).all()


# I6. The circuit
def impatient(strengths=()):
    v = Vehicle(load_definition(IMPATIENT), 50, 21, np.pi / 2, dt=DT)
    for (source, unit), s in dict(strengths).items():
        k = next(j for j, c in enumerate(v.definition.units[v.unit_names.index(unit)].inputs)
                 if c.source == source and c.learning)
        v.strengths[(unit, k)] = s
    return v


EMPTY = {"eye": np.zeros((1, 9)), "door_left": 1.0, "door_right": 1.0}


def watch(v, seconds, forced=lambda t: {}):
    rng = np.random.default_rng(0)
    seen = {u: [] for u in v.unit_names}
    for n in range(round(seconds / DT)):
        force(v, forced(n * DT))
        v.step(EMPTY, DT, rng)
        for u in v.unit_names:
            seen[u].append(v.unit_outputs[u] > 0)
    return {u: np.array(x) for u, x in seen.items()}


def test_imagines_the_carrier_and_taps():
    seen = watch(impatient({COMES_BACK: 1.0, BRINGS_FOOD: 1.0}), 20)
    taps = rises(seen["go_tap"])
    assert len(taps) >= 2
    for k in taps:
        assert seen["look"][k - 20:k].any()
        end = k + int(np.argmin(seen["go_tap"][k:]))
        assert 1.3 <= (end - k) * DT <= 1.8
    assert seen["tap_left"].any() and seen["tap_right"].any()
    assert not (seen["tap_left"] & seen["tap_right"]).any()


def test_without_food_in_mind_it_does_not_tap():
    for strengths in ({COMES_BACK: 1.0}, {BRINGS_FOOD: 1.0}, {}):
        seen = watch(impatient(strengths), 20)
        assert not seen["go_tap"].any() and not seen["tapping"].any()


def test_impatient_runs_without_pygame():
    result = run_python(
        "import sys\n"
        "from braitenberg.scenario import load_scenario\n"
        "from braitenberg.sim import Simulation\n"
        "Simulation(load_scenario('scenarios/vehicle14-impatient.yaml')).run(100)\n"
        "assert 'pygame' not in sys.modules\n")
    assert result.returncode == 0, result.stderr
