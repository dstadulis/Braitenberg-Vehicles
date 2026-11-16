"""Acceptance criteria of specs/vehicle-10.md."""

from functools import lru_cache

import pytest

from braitenberg.definition import load_definition
from braitenberg.diagram import definition_svg
from braitenberg.scenario import load_scenario
from braitenberg.sim import Simulation
from braitenberg.vehicle import Vehicle
from conftest import REPO, run_python

GATHERER = REPO / "vehicles" / "vehicle10-gatherer.yaml"
SCENARIO = REPO / "scenarios" / "vehicle10-gatherer.yaml"
DT = 0.01
PHASE = 600                     # berries sweet for 600 s, then honey on the stones for 600 s
SEEDS = range(5)
SENSE = {"red": "sees_red", "blue": "sees_blue", "yellow": "sees_yellow", "sweet": "smells_sweet"}
FEATURES = list(SENSE)
WIRES = [(a, b) for a in FEATURES for b in FEATURES if a != b]
BUSH = [("red", "sweet"), ("sweet", "red"), ("blue", "yellow"), ("yellow", "blue")]
STONE = [(a, b) for a in ("blue", "yellow", "sweet") for b in ("blue", "yellow", "sweet") if a != b]
NEVER = [("red", "blue"), ("red", "yellow"), ("blue", "red"), ("yellow", "red")]


def strength(v, wire):
    a, b = wire
    return v.strength(SENSE[a], "thinks_" + b)


@lru_cache(maxsize=None)
def run(seed, control=False):
    """Run the default scenario for both phases; per step: (time, strengths by wire, outputs)."""
    scenario = load_scenario(SCENARIO)
    scenario.simulation.seed = seed
    for u in scenario.vehicles[0].definition.units:
        for c in u.inputs:
            if c.learning and control:
                c.learning.learn = 0.0
    sim = Simulation(scenario)
    v = sim.vehicles[0]
    units = ["thinks_" + f for f in FEATURES] + list(SENSE.values()) + ["wants_red", "hunger.full"]
    track = []
    for _ in range(round(2 * PHASE / DT)):
        sim.step()
        out = v.unit_outputs
        track.append((sim.time, {w: strength(v, w) for w in WIRES},
                      {u: out[u] for u in units}))
    return track


def phase(track, n):
    return [step for step in track if (n - 1) * PHASE <= step[0] < n * PHASE]


def meals(steps):
    full = [out["hunger.full"] for _, _, out in steps]
    return sum(1 for a, b in zip(full, full[1:]) if b and not a)


# 1. Ideas emerge
@pytest.mark.parametrize("seed", SEEDS)
def test_two_ideas_emerge(seed):
    _, st, _ = phase(run(seed), 1)[-1]
    assert all(st[w] > 0.7 for w in BUSH)


# 2. Nothing else is learned
@pytest.mark.parametrize("seed", SEEDS)
def test_nothing_else_is_learned(seed):
    track = run(seed)
    assert all(st[w] <= 0.7 for _, st, _ in phase(track, 1) for w in WIRES if w not in BUSH)
    assert all(st[w] == 0 for _, st, _ in track for w in NEVER)


# 3. Part calls up the whole
@pytest.mark.parametrize("seed", SEEDS)
def test_red_calls_up_sweet(seed):
    assert any(out["thinks_sweet"] and out["sees_red"] and not out["smells_sweet"]
               for _, _, out in phase(run(seed), 1))


# 4. The idea pays
def test_eats_more_often_than_control():
    learner = sum(meals(phase(run(seed), 1)) for seed in SEEDS)
    control = sum(meals(phase(run(seed, control=True), 1)) for seed in SEEDS)
    assert learner >= 1.5 * control


# 5. The old idea falls apart
@pytest.mark.parametrize("seed", SEEDS)
def test_red_stops_meaning_sweet(seed):
    steps = phase(run(seed), 2)
    gone = next(k for k, (_, st, _) in enumerate(steps) if st[("red", "sweet")] <= 0.7)
    for _, st, out in steps[gone + round(0.2 / DT):]:      # once the drop has passed through
        if st[("red", "sweet")] > 0.7:
            break
        assert not out["wants_red"]


# 6. A new idea with three members
@pytest.mark.parametrize("seed", SEEDS)
def test_stones_become_one_idea_with_sweet(seed):
    steps = phase(run(seed), 2)
    assert any(all(st[w] > 0.7 for w in STONE) for _, st, _ in steps)
    assert any(out["thinks_sweet"] and out["sees_blue"] and not out["smells_sweet"]
               for _, _, out in steps)


# Rule-level criteria, on the gatherer's own brain fed by hand
def brain():
    return Vehicle(load_definition(GATHERER), 50, 50, 0, dt=DT)


def feel(v, seconds, **on):
    """Hold the senses named in `on` (red, blue, yellow, sweet) on, the others off."""
    readings = {s.name: 0.0 for s in v.definition.sensors}
    for f in on:
        if f == "sweet":
            readings.update(nose_left=10.0, nose_right=10.0)
        else:
            readings[f + "_eye"] = 10.0
    for _ in range(round(seconds / DT)):
        v.think(readings)


# 7. Symmetry of the rule
def test_together_both_ways_alone_one_way():
    v = brain()
    feel(v, 2, red=True, sweet=True)
    assert strength(v, ("red", "sweet")) > 0.7 and strength(v, ("sweet", "red")) > 0.7
    feel(v, 1, red=True)                         # let the sweet pass out of the windows
    before = {w: strength(v, w) for w in WIRES}
    feel(v, 15, red=True)
    assert strength(v, ("red", "sweet")) == pytest.approx(before[("red", "sweet")] - 0.15, abs=0.001)
    assert strength(v, ("sweet", "red")) == before[("sweet", "red")]


# 8. Learned from events only (the Mnemotrix rule since this spec)
def test_an_association_does_not_teach_another():
    v = brain()
    v.strengths[("thinks_red", v_index(v, "thinks_red", "smells_sweet"))] = 1.0
    feel(v, 5, blue=True, sweet=True)          # thinks_red is on, through sweet only
    assert v.unit_outputs["thinks_red"] == 1
    assert strength(v, ("blue", "red")) == 0
    assert strength(v, ("sweet", "red")) == pytest.approx(1.0 - 0.01 * 5, abs=0.01)


def v_index(v, unit, source):
    u = next(u for u in v.definition.units if u.name == unit)
    return next(j for j, c in enumerate(u.inputs) if c.source == source)


def test_three_members_any_one_calls_up_the_others():
    v = brain()
    for a, b in STONE:
        v.strengths[("thinks_" + b, v_index(v, "thinks_" + b, SENSE[a]))] = 1.0
    feel(v, 1, blue=True)
    out = v.unit_outputs
    assert out["thinks_yellow"] and out["thinks_sweet"] and not out["thinks_red"]


# 9. No reverberation
def test_ideas_go_off_with_the_senses():
    v = brain()
    for a, b in WIRES:
        v.strengths[("thinks_" + b, v_index(v, "thinks_" + b, SENSE[a]))] = 1.0
    feel(v, 1, red=True, blue=True, yellow=True, sweet=True)
    feel(v, 0.2)
    assert not any(v.unit_outputs["thinks_" + f] for f in FEATURES)


# 10. Display and headless
def test_diagram_reports_twelve_mnemotrix_wires():
    v = brain()
    wires = [w for w in v.diagram().wires if w.learning]
    assert len(wires) == 12 and {w.learning for w in wires} == {"mnemotrix"}
    assert "M learning (Mnemotrix)" in definition_svg(load_definition(GATHERER))


def test_runs_without_pygame():
    result = run_python(
        "import sys\n"
        "from braitenberg.scenario import load_scenario\n"
        "from braitenberg.sim import Simulation\n"
        "Simulation(load_scenario('scenarios/vehicle10-gatherer.yaml')).run(100)\n"
        "assert 'pygame' not in sys.modules\n")
    assert result.returncode == 0, result.stderr
