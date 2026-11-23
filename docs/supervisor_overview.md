# Closed-Loop Adaptive RL: progress overview (as of 2026-11-23)

**What belongs in this document.** It prepares the meetings with the supervisor and contains:
- **Architecture decisions** that shape the whole project, including those already taken and
  implemented: what was decided, why, which alternatives were rejected, and what follows from
  it. Each one is listed so the supervisor can confirm or overturn it.
- **Results and findings** that bear on these decisions or on the direction of the project.
- **Open questions** for the next steps.

It does not contain implementation details, parameter tuning or code structure; those live in
the decision log (`docs/decisions.md`, numbers #n) and the commits. Keep items short.

## 1. Summary

- **Plant and data:** a simulated Crazyflie (Crazyflow, JAX) behind a minimal
  `reset / step(u)` interface, a geometric baseline controller, and recorded data sets for
  three conditions (nominal, +30 % mass, constant wind force).
- **Dynamics model:** analytic nominal model (as the LotF surrogate) plus a learned residual
  acceleration (MLP). Trained offline, it cuts the 30-step position prediction error on
  held-out runs by 4–12× under disturbances.
- **Policy gradient:** as in the charter, from finite-horizon rollouts inside the hybrid model,
  starting at the currently measured state. First-order: BPTT through these rollouts.
- **Online model update:** in every control step the residual takes one Adam step on a
  sliding window of the newest real transitions, then the policy takes one SGD step.
- **First closed-loop results (+30 % mass mid-flight):** with the nominal model the
  adaptation steers the wrong way and crashes. With the online model (window 1–2 s) the
  error after the switch drops from 50 cm (baseline) to 11–14 cm. With a 5 s window the
  model learns a spurious dependence on vertical velocity and the adaptation fails again.
- **Not built yet:** the zeroth-order pipeline; the FO vs ZO comparison.

## 2. The system

```
                 ref (p, v, a)
                      │
   x_k ──► policy: u = baseline(x, ref) + π_θ(errors, R, a_ref) ──► Crazyflow plant ──► x_{k+1}
    ▲                                                                                     │
    └─────────────────────────────────────── z⁻¹ ◄───────────────────────────────────────┤
                                                                                          │
   model loop:    sliding window of the last W s of real tuples (x, u, x⁺)   ◄────────────┘
                  → one Adam step on residual φ
   policy loop:   from x_k, roll out H = 30 steps in the hybrid model with π_θ
                  → cost on the predicted states → BPTT → one clipped SGD step on θ

   dynamics model:   x⁺ = nominal(x, u) with a_res = MLP_φ(R, v, ω_cmd, T/mg)
```

Stack: JAX, flax.nnx, optax, Crazyflow; CPU only (macOS). Reference implementation:
Learning on the Fly (LotF, uzh-rpg).

## 3. Architecture decisions for review

Ordered by how much of the project depends on them.

**A1. Policy gradient only from rollouts in the surrogate model (#17; charter).**
- *Decision:* in every control step, start at the measured state, roll out H = 30 steps
  (0.3 s) with the current policy in the hybrid model, and evaluate the tracking cost on the
  predicted states. First-order: BPTT through this rollout; zeroth-order (later): sampling
  on the same rollouts. The buffer of real (x, u, x⁺) tuples feeds only the model.
- *Why:* charter (fixed decision 1, "rollouts live inside the model"); no episodes or resets
  on the plant; the learned model is what makes the gradient good (LotF principle).
- *Consequence:* the policy can only be as good as the model's prediction over H steps
  (see A4, Section 4).
- *To confirm:* H = 30 and how to choose and report it (charter cites "Do Differentiable
  Simulators Give Better Policy Gradients?").

**A2. Residual policy: baseline controller + small neural network (#20).**
- *Decision:* u = baseline(x, ref) + π_θ(·). The network's output layer starts at zero, so the
  untrained policy is exactly the baseline. 2 × 32, tanh, ~1.8k parameters.
- *Why:* without resets the policy must be stable from the first control step. A small,
  smooth net keeps ZO variance low and BPTT derivatives smooth.
- *Alternatives:* LotF uses hover thrust + MLP (2 × 512, ~270k parameters), pretrained
  offline; a pure network would need pretraining before closing the loop.
- *To confirm:* acceptable as "the RL policy"? Which benchmark is fair: the baseline (no
  integrator), a baseline with integrator, or a classical adaptive controller?

**A3. Policy inputs and cost (#20).**
- *Decision:* inputs are tracking errors, attitude and feedforward acceleration, with no
  absolute position. Cost as LotF (smooth-L1 on position and velocity error) plus a penalty on
  the *correction* (weight 0.01; our thrust unit is m·g, not N).
- *Why:* the policy is not tied to one trajectory or place. LotF's policy has no reference
  input and is trajectory-specific.

**A4. Hybrid model: LotF nominal model + residual acceleration (#6, #12).**
- *Decision:* nominal model = point mass, thrust along body z, ω = ω_cmd, thrust acts
  instantly (no rate or rotor dynamics, = LotF `simplified_dyn`). Residual = MLP 2 × 128 that
  adds a world-frame acceleration; translation only; no position input; one net, no ensemble.
- *Why:* comparability with LotF; leaves a known, nameable mismatch. The residual covers mass,
  drag, thrust delay. Without position input the net cannot memorise where it flew.
- *Consequence:* the nominal one-step input Jacobian is 3–15× too large (Section 5.1). A
  position-dependent disturbance (wind zone) would need the position input (#13).
- *To confirm:* is this bias the kind we want for the FO/ZO comparison, or should the nominal
  model get first-order rate dynamics?

**A5. Which derivatives BPTT uses (#11).**
- *Decision:* forward through the hybrid model, backward through the nominal model only (the
  residual is a constant for differentiation, as LotF's `custom_jvp`). The derivative through
  the residual exists as an option.
- *Why:* LotF does it this way; the residual still shapes the gradient through the predicted
  states. The derivative through the network matters only for disturbances that change the
  effect of u (mass), not for additive ones (constant wind), and it needs excitation to be
  identifiable (Section 5.2).

**A6. Update rule: plain SGD, fixed step size, one step per control step (#20).**
- *Decision:* one SGD step per control step (100 Hz), gradient clipping, no Adam.
- *Why:* small, frequent steps as the stability mechanism (charter fixed decision 2, PGAC).
  Adam's per-parameter normalisation would hide the difference between a biased (FO) and a
  noisy (ZO) gradient.
- *To confirm:* is there a link to PGAC step-size conditions that we should respect?

**A7. Plant, control interface and rates (#1, #2, #3, #8).**
- *Decision:* Crazyflow, body rates + collective thrust, policy 100 Hz, physics and inner rate
  loop 500 Hz. Only `plant.py` knows the simulator. Disturbances: true mass and a constant
  world-frame force, changeable mid-run; controller and model only know the nominal mass.
- *Why:* same input as LotF, no Euler singularities (looping possible), light enough for a
  laptop, differentiable (true gradient for evaluation).
- *Alternatives:* attitude control (singular at ±90°), motor-level control (≥ 500 Hz and motor
  dynamics in the model).

**A8. Offline data with held excitation noise (#7).**
- *Decision:* the training run uses a random smooth reference plus held noise on u.
- *Why:* in closed loop u = controller(x); without noise the effect of u (what BPTT needs) is
  not identifiable. LotF gets variety from random resets; we have none.

**A9. Online model update (#22).**
- *Decision:* in every control step, first the model, then the policy. The residual takes one
  Adam step (lr 1e-2, warm start) on a sliding window of the newest W s of real transitions;
  it starts at zero (hybrid = nominal), one network, no excitation.
- *Why:* the model must adapt at least as fast as the policy, because a stale model steers
  the policy the wrong way (Section 5.1). Offline, Adam needed 40–130 full-batch steps from
  zero, i.e. 0.4–1.3 s at one step per control step.
- *Alternatives:* LotF retrains every 3 s for 100 iterations on a 2–5 s buffer with an
  ensemble of 3; exponential forgetting instead of a hard window (noted for later).

## 4. Results so far

**Model, open-loop prediction on held-out runs (#16).** Position error at k = 30 steps
(0.3 s), mean over all start points:

| condition | nominal | hybrid | ratio |
|---|---|---|---|
| +30 % mass, circle / hover | 13.7 / 13.5 cm | 1.1 / 1.0 cm | 0.08 |
| wind, circle / hover | 5.4 / 5.4 cm | 1.3 / 1.2 cm | 0.24 |
| nominal, circle / hover | 2.2 / 1.24 cm | 1.2 / 1.27 cm | 0.56 / 1.03 |

**Cross test.** A residual trained on one condition is worse than no residual on another
condition (e.g. the mass residual on nominal data: 2.2 → 7.0 cm). An offline model alone is
not enough when the situation changes.

**Closed loop (#20).** Circle (r = 0.5 m, 4 s period), mass ×1.3 from t = 10 s, mean position
error:

| | 5–10 s (nominal) | 10–12 s (after switch) | 15–20 s |
|---|---|---|---|
| baseline | 2.2 cm | 32 cm | 50 cm |
| FO, nominal model (residual untrained) | 2.1 cm | 31 cm | crash at 15.4 s |
| FO, online model, window 1 s | 2.3 cm | 6.9 cm | 11.5 cm |
| FO, online model, window 2 s | 2.3 cm | 5.3 cm | 13.7 cm |
| FO, online model, window 5 s | 2.2 cm | 20 cm | 65 cm |
| same + held thrust noise 0.1 m·g, window 1 s | 8.6 cm | 2.2 cm | 1.7 cm |
| same + held thrust noise 0.1 m·g, window 2 s | 5.7 cm | 4.3 cm | 2.7 cm |
| same + held thrust noise 0.1 m·g, window 5 s | 4.0 cm | 13 cm | 58 cm |

H = 30, policy SGD lr 1e-2, model: one Adam step (lr 1e-2) per control step, residual
starts at zero. The baseline has no integrator and keeps a ~49 cm height offset. With the
nominal model the thrust correction turns negative (−0.65 m·g at 15 s) until the drone hits
the floor limit. With the 1–2 s window the online model predicts the mass effect within
~0.5 s, the thrust correction turns positive and reaches +0.25 m·g at 20 s (0.30 needed,
still rising). Excitation noise on the thrust lets the short windows reach the full
correction (0.30 m·g) but costs tracking before the switch (2.2 → 4–9 cm) and does not fix
the 5 s window.

## 5. Findings worth discussing

1. **A model that does not know the disturbance steers the policy the wrong way.** In the
   nominal model the baseline pulls a too-low drone back up by itself; the gradient then
   mainly damps the growing velocity error and asks for *less* thrust. The real drone sinks
   further, the predicted pull-up grows, and so does the damping. One update step from a
   fixed state (H = 30):

   | state, model | thrust correction after one step |
   |---|---|
   | 5 cm below the reference, nominal model | −0.012 m·g |
   | on the reference, residual knows +30 % mass (a_res = −0.23 g) | +0.110 m·g |

   So the sign is right once the model predicts the sinking. This is what the online model
   update has to deliver, and fast: while the model is stale, adaptation is harmful, not just
   slow. Open: should the policy update be gated (e.g. paused while the model's online error
   is high)?
2. **The nominal model's one-step input Jacobian is far off.** Against the true simulator
   gradient on a circle: ∂v_z⁺/∂T is 0.015 (true) vs 0.23 (nominal), ∂ω⁺/∂ω_cmd is 0.14–0.32
   (true) vs 1 (nominal). Cause: rotor and rate dynamics within 10 ms. Over H steps this
   probably averages out partly; not measured yet.
3. **Identifiability of the residual online, confirmed.** In closed loop the data never
   show the drone rising while heavy. With a 5 s window the residual explains the mass effect
   partly through vertical velocity (learned ∂a_res,z/∂v_z ≈ +6 … +16 instead of ≈ 0): it is
   right at the flown state (sinking), but in the imagined rollout the drone rises and the
   model predicts only −0.3 … −0.7 m/s² instead of −2.9. The rollout is then back to the
   nominal case of finding 1. The one-step error at the flown state does not reveal this.
   Candidate remedies: excitation noise in closed loop (A8 online), removing or constraining
   inputs of the residual, shorter windows.

## 6. Alignment with the project charter (2026-11-11)

| Charter | Current state | Status |
|---|---|---|
| Surrogate model used for both ZO and FO gradient estimation (fixed decision 1) | FO on hybrid-model rollouts; ZO planned on the same rollouts. | consistent |
| Finite-horizon rollouts live inside the model | H = 30 steps in the hybrid model from the measured state. | consistent |
| Small, frequent updates (fixed decision 2) | One SGD step per control step. | consistent |
| At least one clearly nonlinear trajectory, e.g. a looping (fixed decision 3) | Circle, hover, sum of sines so far. Input choice allows loopings; the baseline does not handle them yet. | open |
| Simulator: visualisation, light, preferably differentiable | Crazyflow: MuJoCo viewer, CPU, differentiable (verified). Replay rendering of runs planned. | consistent, visualisation open |
| Drone as environment | yes | consistent |

## 7. Open questions for the next steps

1. **Online model update:** sliding window (1–2 s works, 5 s fails, Section 5.3);
   exponential forgetting as an alternative; model every control step and policy every
   second step. Is "the model must adapt at least as fast as the policy" the right principle?
2. **Excitation during online operation** (Section 5.3): acceptable to inject small noise in
   closed loop, or should identifiability come from elsewhere (e.g. fewer residual inputs)?
3. **Derivative through the residual (A5 option):** smooth activation needed (tanh/softplus
   instead of ReLU)? Should the comparison of model vs true Jacobians over H steps become a
   main evaluation?
4. **Postponed on purpose:** measurement noise / state estimation; position-dependent
   disturbances (wind zone: "tracking a disturbance" vs "learning a map"); the looping
   trajectory. Is the order right?

## 8. Next steps

1. **Identifiability of the online model** (Section 5.3), then the wind experiment.
2. **Zeroth-order pipeline** on the same model rollouts, with the same policy and cost.
3. **Evaluation:** tracking over time, online model error, model gradient vs true gradient.
   Rendering of runs for the report.
