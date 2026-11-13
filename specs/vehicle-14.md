# Vehicle 14 — Egotism and Optimism: the optimist

**Status:** specified, not implemented. The values below are starting points, to be tuned on seeds 0 to 4 (see *Timing*).

## Summary

Braitenberg's last chapter starts from what is still missing in the vehicles. Even the most complex ones, of type 13, show no **personality**. They seem driven by necessity rather than drawn by goals, in spite of a predictor that seems to draw motives from a future state of the world. People are interesting because they seem to be after something. So he gives his last vehicle "a certain amount of systematic egotism, with a touch of the pleasure principle".

He needs nothing new, only the parts already built:

- **Prediction**, from the Ergotrix wires ([vehicle-11.md](vehicle-11.md)).
- **A Darwinian evaluator**, from Vehicle 13 ([vehicle-13.md](vehicle-13.md)): ancient knowledge of what is good and what is bad.
- **A succession of mental states**, from Vehicle 12 ([vehicle-12.md](vehicle-12.md)). Until now, where the predictor pointed to two equally likely next states, the choice between them was left to chance.

Vehicle 14 gets one rule: **when choosing among several equally likely next brain states, choose the most pleasing one.** The way to do it has three steps:

1. **Hold** the present state for a short time, in short-term memory.
2. **Scan:** the predictor runs quickly through its predictions, and the evaluator scores each one as favorable or unfavorable.
3. **Choose:** the predictor goes through them again and stops at the one with the highest score, which becomes the next state of the brain.

The brain's states include the vehicle's own movements, so choosing a pleasing thought means doing what leads to it. "While dreaming and sleepwalking, the vehicle transforms the world (and its own position in the world)" so that the world ends up in a more favorable state. From outside, the vehicle moves with consistent determination, always after something that the observer cannot guess at the outset, and that may not even be there when it arrives. Most of the time the chain of optimistic predictions proves right, and Vehicle 14 reaches goals that Vehicle 13 "couldn't even dream of".

This spec builds that rule into one vehicle, the **optimist**, on a looping road that forks in two lanes. Food is in the upper lane, out of reach of every sense at the fork. At first the optimist takes either lane, as chance has it, and learns three things: that the fork is followed by the upper lane or the lower one, and that the upper lane is followed by food. From then on, at each fork it imagines both lanes, one after the other, finds that one of them leads to food, and takes it. It keeps doing so on the laps when the food is not there. When the food moves to the other lane, its optimism fails a few times, it goes back to chance, and soon it learns the new way.

It needs no new kind of part: threshold units and Ergotrix wires are enough.

## From the book to the vehicle

| Book | Here |
|---|---|
| Equally likely next states | The two lanes after the fork: the predictor has seen both follow it |
| Hold the present state | `deciding`, which keeps `p_fork` on from the fork until a choice is made |
| The predictor goes through its predictions | `phase`, `look_upper` and `look_lower`, which let the predictor think of one lane at a time |
| The Darwinian evaluator | `pleasing`: food is good. Built in, as Vehicle 13's `upset` |
| Stop at the highest score | `choose_upper`, `choose_lower`: the lane being thought of when the evaluator is pleased |
| The next state, which implies movement | `go_upper`, `go_lower`: the chosen thought is enacted, and steers the vehicle into its lane |

**Highest score, or first pleasing one.** The evaluator here has two values: pleasing or not. So scanning until something pleases and stopping there gives the same choice as scanning everything and taking the best, and needs no memory of scores. With graded values the two differ (see *Out of scope*).

**Why one at a time.** Both lanes are predicted from the fork. If the predictor thought of both at once, the predicted food would come from one of them, but the evaluator could not tell which. Thinking of one lane at a time is what lets each score belong to its prediction. This is the book's "goes quickly through its various predictions".

**Why not the caudate.** Braitenberg describes the optimistic slant as given to the threshold control of Vehicle 12. A caudate that controlled the predicting units would also lower their thresholds whenever they were silent, and they would think on their own while the vehicle drives: each such thought starts an Ergotrix wait that fails, and erases what the wire learned (the open question in [vehicle-12.md](vehicle-12.md), *Open questions*). Here the scan runs only while the vehicle holds a decision, and stops when it has chosen. Optimism in the daydreamer itself is left for later (see *Out of scope*).

## The world

`worlds/fork-road.yaml`: an arena 100 × 40 with `edges: wrap`. A road runs left to right and wraps round, so the vehicle drives laps. A narrow stem opens into a fork, two lanes round a divider, which close again into the stem.

```
y=40 ┌────────────────────────────────────────────────────────────┐ ceiling
     │████████████                  upper lane        ╲███████████│
     │████████████        ◢■■■■■■■■■■■■■■■■■■■■■■■◣ food ╲██████████│
y=20 │   stem  ─▶   fork ◀ ■■■■■■■ divider ■■■■■■■■ ▶      stem ─▶ │
     │████████████        ◥■■■■■■■■■■■■■■■■■■■■■■■◤      ╱██████████│
     │████████████                  lower lane        ╱███████████│
y=0  └────────────────────────────────────────────────────────────┘ floor
     x=0         20      28   34                   70  78  80     100
```

### Obstacles

All boxes have `height` 2 and are gray.

| Box | Position | Length × width | Heading | Is |
|---|---|---|---|---|
| Floor, ceiling | `[50, 1]`, `[50, 39]` | 100 × 2 | 0 | The road's outer walls |
| Stem, lower | `[90, 8.5]` and `[10, 8.5]` | 20 × 13 | 0 | The stem, y 15 to 25, from x 80 round to x 20 (two boxes each, since a box cannot cross the wrapping edge) |
| Stem, upper | `[90, 31.5]` and `[10, 31.5]` | 20 × 13 | 0 | |
| Divider | `[52, 20]` | 36 × 16 | 0 | x 34 to 70, y 12 to 28 |
| Nose | `[34, 20]` | 11.3 × 11.3 | 45 | Turns the divider's front into a point at x 26, so that a vehicle driving straight at it slides off to one side instead of stopping |
| Tail | `[70, 20]` | 11.3 × 11.3 | 45 | The same at the back, pointing at x 78 |
| Funnels | `[75, 31.5]` and `[75, 8.5]` | 19 × 2 | -43 and 43 | Lead each lane back into the stem |

The stem centers the vehicle before every fork, whichever lane it came from. Without it, a vehicle leaving the upper lane would go on along the top of the road and enter the upper lane again, and chance would not decide anything.

### Stimuli

| Stimulus | Sources | Meaning |
|---|---|---|
| `yellow` | `[28, 20]`, intensity 10, falloff 8 | A post at the fork, at the tip of the nose |
| `blue` | `[42, 33]`, intensity 10, falloff 5 | The mark at the entrance of the upper lane |
| `red` | `[42, 7]`, intensity 10, falloff 5 | The mark at the entrance of the lower lane |
| `food` | `[52, 33]`, intensity 10, falloff 4, `schedule: {period: 120, on_at: 0, off_at: 90}` | Food, in the upper lane, there three quarters of the time |
| `proximity` | none; `surfaces` with intensity 10, falloff 2 | Walls, for infrared |

Fields pass through obstacles ([worlds.md](worlds.md), *Blocking*), so the lanes are far enough apart that nothing in one is sensed from the other: from the lower lane, food reads at most 0.2 and the blue mark 0.6.

Nothing at the fork says where the food is. The food is 24 units past the post, behind the divider, and smelled only within about 5 units.

## The optimist

### Parts

- **Body:** `size` 4, `noise_std` 0.3, so that, before it has learned, it takes either lane as chance has it.
- **Sensors:**
  - `fork_eye` at `[0, 0.9]`, reading `yellow`, `field_of_view` 60: the post while it is ahead.
  - `blue_eye` and `red_eye` at `[-0.3, 0]` and `[0.3, 0]`, reading `blue` and `red`, all around.
  - `nose` at `[0, 0.5]`, reading `food`, all around.
  - `ir_left` and `ir_right` at `[-0.6, 0.9]` and `[0.6, 0.9]`, reading `proximity`, `field_of_view` 180, as the Vehicle 13 grazer's.
- **Motors:** `left_motor` at `[-0.5, -0.8]` and `right_motor` at `[0.5, -0.8]`, `base_speed` 3, `gain` 1, `max_speed` 4.

### Units

#### R, what it senses

| Unit | Inputs | Threshold | Delay | Fires when |
|---|---|---|---|---|
| `at_fork` | `fork_eye` | 2 | 0.05 | the fork post is ahead, within about 15 units |
| `in_upper` | `blue_eye` | 5 | 0.05 | the upper lane's mark is within about 5 units: it is entering the upper lane |
| `in_lower` | `red_eye` | 5 | 0.05 | the same for the lower lane |
| `smells` | `nose` | 4 | 0.05 | food within about 5 units |

#### P, what it predicts

| Unit | Inputs | Threshold | Delay | Fires when |
|---|---|---|---|---|
| `p_fork` | `at_fork`, `deciding` | 1 | 0.05 | at the fork, or still deciding what to do there |
| `p_upper` | `in_upper` twice, `look_upper`, inhibitory `p_lower`; **Ergotrix** from `p_fork` | 2 | 0.1 | in the upper lane; or, once learned, while it looks at the upper lane from the fork |
| `p_lower` | `in_lower` twice, `look_lower`, inhibitory `p_upper`; **Ergotrix** from `p_fork` | 2 | 0.1 | the same for the lower lane |
| `p_food` | `smells`; **Ergotrix** from `p_upper` and from `p_lower` | 1 | 0.05 | food, or, once learned, a lane it leads to |

The Ergotrix wires all start at strength 0:

| Wire | `within` | `learn` | `forget` | Learns |
|---|---|---|---|---|
| `p_fork → p_upper`, `p_fork → p_lower` | 10 | 0.4 | 0.05 | the fork is followed by that lane |
| `p_upper → p_food`, `p_lower → p_food` | 5 | 0.4 | 0.1 | that lane is followed by food |

A lane is predicted only while it is **looked at**: the wire from `p_fork` gives 1 and `look_upper` 1, which reaches the threshold of 2, and neither does alone. What the vehicle senses counts twice, so being in a lane fires its unit whatever it thinks. The two lanes inhibit each other, so the predictor never thinks of both at once, and a lane thought of goes off before the other one comes on.

As in Vehicle 11, the wires learn only from their destinations' fixed inputs: `p_fork → p_upper` from `in_upper`, never from a look. `look_upper` is a fixed input too, but alone it does not reach the threshold, so it never counts as the destination turning on.

#### Hold and scan

| Unit | Inputs | Threshold | Delay | Fires when |
|---|---|---|---|---|
| `deciding` | `at_fork`, itself, inhibitory `go_upper`, `go_lower`, `in_upper`, `in_lower` | 1 | 0.05 | from the fork until it has chosen, or has reached a lane without choosing |
| `phase` | `deciding`, inhibitory itself | 1 | 0.5 | on and off by turns while deciding, about half a second each |
| `look_upper` | `deciding`, `phase` | 2 | 0.05 | deciding, while `phase` is on |
| `look_lower` | `deciding`, inhibitory `phase` | 1 | 0.05 | deciding, while `phase` is off |

So while it decides, the predictor looks at the lower lane, then the upper, then the lower again, and so on. A look lights its lane only if `p_fork` predicts it: before the fork wires have learned, the looks find nothing.

#### D, the evaluator, and the choice

| Unit | Inputs | Threshold | Delay | Fires when |
|---|---|---|---|---|
| `pleasing_was` | `p_food` | 1 | 0.05 | `p_food`, a moment late |
| `pleasing` | `p_food`, inhibitory `pleasing_was` | 1 | 0.05 | food has **just** come to mind, or to the nose: a pulse of about 0.05 s |
| `choose_upper` | `p_upper`, `look_upper`, `pleasing` | 3 | 0.05 | the upper lane, being looked at, has just led to food |
| `choose_lower` | `p_lower`, `look_lower`, `pleasing` | 3 | 0.05 | the same for the lower lane |
| `go_upper` | `choose_upper`, itself, inhibitory `in_upper`, `in_lower` | 1 | 0.05 | from the choice until it reaches a lane |
| `go_lower` | `choose_lower`, itself, inhibitory `in_upper`, `in_lower` | 1 | 0.05 | the same |

The evaluator scores an **onset**, not a state, for the same reason as the scan: if food were scored while it stays in mind, the lane looked at next, as `p_food` is still fading, would be credited with it. `pleasing` is sharpened by its `was` copy as Vehicle 13's echo units are ([vehicle-13.md](vehicle-13.md), *The short-term memory and P*).

`pleasing` is D, as `upset` was in Vehicle 13: it is not learned. That food is good is the ancient knowledge that Braitenberg gives the vehicle ready-made. What the vehicle learns is where food comes from.

#### Movement

`left_motor` takes, inhibitory, `go_upper` and `ir_right` twice; `right_motor` takes, inhibitory, `go_lower` and `ir_left` twice. So it drives at 3 and keeps clear of walls as the grazer does; `go_upper` slows the left wheel to 2, which turns it left, toward the upper lane, and `go_lower` turns it right. Neither decides anything about the lane once the vehicle is in it: walls and infrared take it from there.

### How it behaves

- **Before it has learned,** the fork means nothing. `deciding` turns on at the post, the looks find nothing, and it is still deciding when it reaches a lane, which the noise and the nose choose. Each lap, the fork wire to the lane it took gains 0.4 and the other loses 0.05, so both pass 0.7 after a few laps of each. In the upper lane, when food is there, `p_upper → p_food` gains 0.4; two meals make it active.
- **Learned.** At the post, `deciding` holds `p_fork`, and the scan begins. Looking at the lower lane, `p_lower` lights, and nothing follows. Looking at the upper lane, `p_upper` lights, `p_food` follows one delay later, and `pleasing` pulses: `choose_upper` fires, `go_upper` latches and ends the scan. About a second after the post, the vehicle turns left, toward food it cannot sense. It is after something the observer cannot guess at the fork.
- **Every lap.** It takes the upper lane whenever it remembers food there. The lower fork wire, never followed, falls below 0.7 after about six laps, and the vehicle stops even imagining the lower lane: there is nothing there it wants.
- **When the food is not there.** A quarter of the time, the food is off. The vehicle goes to the upper lane just the same, "running after a dream", and the wire loses 0.1. Most of the time its optimism is right, and the wire stays active.
- **When the food moves.** If the food is moved to the lower lane, each visit to the empty upper lane costs the wire 0.1, and each thought of it at the fork another 0.1 (see *Timing*). After two laps it is below 0.7, nothing pleases at the fork, and the vehicle goes back to chance. As soon as chance takes it to the lower lane, the fork wire to it relearns and two meals there make `p_lower → p_food` active. From then on it takes the lower lane. Its preference was never wired in: only what it learned decides it.
- **The indifferent vehicle** (the control) is the same definition with nothing feeding `pleasing`. It learns the same wires and scans the same way, but nothing ever pleases it, so it never chooses: it takes either lane, as Vehicle 13 would. Over a run it eats about half as often.

### Timing

1. **The decision comes before the nose.** `at_fork` turns on about 15 units before the post, 5 s at 3 units a second. The scan takes at most two phases, about 1.1 s, and the turn into the upper lane, which must go from y 20 to above 28, needs the rest. If tuning shows it is too late, the post moves back or `phase` gets shorter. The nose makes sure a late or missing choice still ends in one lane or the other.
2. **A thought does not take a real visit's credit.** Each look at the upper lane is an onset of `p_upper`, which starts a wait on `p_upper → p_food`. Its `within` must end before the vehicle enters the lane: the first look is at about x 15, and with `within` 5 the wait ends near x 30, before `in_upper` at about x 37. Otherwise the real entry would come while the wire is waiting, change nothing ([vehicle-11.md](vehicle-11.md), *The rule*), and the thought would be judged by the meal, 13 s later, and fail. As it is, the thought fails, `-0.1`, and the real visit, from x 37 to food smelled at about x 47, 3.3 s, pairs, `+0.4`: a net gain each lap with food. The fork wires' `within` of 10 covers the 8 s from the post to a lane.
3. **One pulse per look.** `pleasing` must turn on while the lane that caused it is still being looked at, and `p_food` must go off before the next look, so that the same lane, looked at again, pleases again. With `phase` 0.5 and the delays above, `p_food` follows its lane by 0.1 s and goes off about 0.15 s after it.

Each thought of the upper lane costs its wire `forget`. This is the open question of Vehicle 12, thinking that erases memory, in a mild form: here a real visit pays for it, but a vehicle that thought more often than it went would forget what it knows (see *Open questions*).

### Diagram and panel

As for the Vehicle 13 once-bitten vehicle. The units are in rows from front to back: R; P, with the fork in the middle and the two lanes to the left and the right, upper on the left, as the vehicle turns left to it; the scan; D and the choice; and the motors. Watch, at each fork, `deciding` light up, `look_lower` and `look_upper` take turns, and `pleasing` pulse with `look_upper`; in the readouts, `p_upper->p_food  E` climb in steps with each meal.

## Definitions, world and scenarios

- `vehicles/vehicle14-optimist.yaml`: the optimist, `general.type` `"14"`, named *Optimist (goes where it imagines food)*.
- `worlds/fork-road.yaml`: the world above.
- `scenarios/vehicle14-optimist.yaml`: the optimist at `[90, 20]`, in the stem, heading 0; `dt` 0.01.
- `specs/images/vehicle14-optimist.svg` and `specs/images/vehicle14-optimist.gif`: the diagram and a recording.

```
python -m braitenberg scenarios/vehicle14-optimist.yaml
python -m braitenberg.diagram vehicles/vehicle14-optimist.yaml --out specs/images/vehicle14-optimist.svg
python -m braitenberg.record scenarios/vehicle14-optimist.yaml --seconds 300 --every 0.25 --speed 8 --scale 0.5 --out specs/images/vehicle14-optimist.gif
```

## Related specs

- [vehicle-11.md](vehicle-11.md): the Ergotrix wire, and its rule on onsets and fixed inputs.
- [vehicle-12.md](vehicle-12.md): trains of thought and threshold control; its open question on thinking that erases memory.
- [vehicle-13.md](vehicle-13.md): the predictor P, the realistic R and the evaluator D; sharpened pulses.
- [worlds.md](worlds.md): obstacles, schedules and the proximity field.

## Acceptance criteria

Criteria 1 to 6 use the default scenario, run headless for 600 s, and must hold for each of seeds 0 to 4. A **lap** is the time between two onsets of `at_fork`; it **takes** a lane when `in_upper` or `in_lower` turns on in it. A **meal** is an onset of `smells`. The **upper wire** is `p_upper → p_food`. The **control** is the same definition without `p_food` among `pleasing`'s inputs.

1. **Explores first:** before the upper wire first exceeds 0.7, the optimist takes each lane at least once, and `go_upper` and `go_lower` never fire.
2. **Learns the road:** both fork wires exceed 0.7 at some time; the upper wire exceeds 0.7 before 300 s; `p_lower → p_food` never exceeds 0.7.
3. **Scans one at a time:** in every lap in which both fork wires are above 0.7, `p_upper` and `p_lower` are never on together, and while `deciding` is on each of them turns on at least once before `go_upper` or `go_lower` does, or before `deciding` ends.
4. **Chooses the pleasing lane:** in every lap that starts with the upper wire above 0.7, `go_upper` turns on within 2 s after `at_fork`, `go_lower` does not, and the optimist takes the upper lane in at least 90% of those laps.
5. **Even when it is not there:** of those laps, the ones in which the food is off when the vehicle enters the upper lane are taken to the upper lane in the same proportion, and at least one such lap occurs in each run.
6. **Eats more:** over seeds 0 to 4 together, the optimist has at least 1.5 times as many meals as the control.
7. **Follows the food:** with the food moved to `[52, 7]`, in the lower lane, at 300 s, and the run going on to 900 s: the upper wire falls to 0.7 or below within 4 laps after the move; `p_lower → p_food` exceeds 0.7 before 900 s; and in the last 3 laps the optimist takes the lower lane.
8. **The circuit:** with chosen unit outputs and the wires held at the given strengths:
   - with both fork wires active and `at_fork` turning on, `look_lower` and `look_upper` take turns while `deciding` is on, and `p_lower` and `p_upper` follow them;
   - with only the upper wire active, `pleasing` pulses once for each look at the upper lane, for at most 0.1 s, and never during a look at the lower lane; `choose_upper` fires on the first such pulse, `go_upper` latches, and `deciding`, `look_upper` and `look_lower` go off;
   - with neither lane wire active, `deciding` stays on until `in_upper` or `in_lower`, and neither `go_` unit fires;
   - `go_upper` holds until `in_upper` or `in_lower` turns on.
9. **Thoughts are not visits:** a look at the upper lane never counts as `p_upper`'s fixed inputs turning on, so `p_fork → p_upper` gains only when `in_upper` turns on.
10. **Headless and fast:** loading and running the scenario never imports pygame, and 600 simulated seconds run headless in under 60 s.

## Open questions

- **Thinking that erases memory.** Each thought of a lane starts an Ergotrix wait that fails. Here a real visit makes up for it each lap. A vehicle that thinks more, scanning many states or for longer, would need the rule suggested in Vehicle 12: only a source driven by its fixed inputs starts a wait. For the existing vehicles, whose wire sources are all driven by fixed inputs, that rule changes nothing; it is a change to the Ergotrix rule, and is left until a vehicle needs it.
- **Which lane is scanned first.** The scan always starts with the lower lane. With food in both lanes, the first pleasing one wins, so the order decides. That is harmless here, but it is a bias that a vehicle with graded values should not have.

## Out of scope

- **The impatient vehicle.** Braitenberg's other scene: a Vehicle 14 waiting for another vehicle that carries an appealing source, which now and then performs the motions it would make when the source arrives, as if it could hasten it, as we run to the door before the bell rings. Here a pleasing thought is enacted only as the lane it chooses; a vehicle whose thoughts of the pleasing event itself drive its motors would show it. It is the natural second example.
- **Graded values.** An evaluator with more than two values, and the book's double pass, scanning everything and then stopping at the best, which needs the scores kept while the scan goes on.
- **Chains longer than two.** The optimist looks one step ahead from each lane: lane, then food. A predictor that scans chains of several steps, choosing at each, reaches farther goals.
- **Optimism in the daydreamer.** The optimistic slant given to Vehicle 12's threshold control itself, so that trains of thought drift toward pleasing ideas with the senses silent: "nice dreams".
- **Satiety.** The optimist is always hungry. With the hunger module ([modules.md](modules.md)), food would please only while it is hungry, and a full vehicle would take either lane.
- **Bad things.** An evaluator that also scores predictions as unpleasant, so that the vehicle avoids a lane that leads to the Vehicle 13 bully, and weighs a pleasing lane against a dangerous one.
- **D by selection.** As in Vehicle 13, D is written by hand. Evolving it with Vehicle 6 is left for later.
