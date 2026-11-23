# Handover: Online-Modell + Policy-Schleife (Kernaufgabe)

Für den nächsten Coding Agent. Nach dem Lesen: CLAUDE.md, `docs/decisions.md`, `git log`,
die Projekt-Charter (`../Project Charter Closed-Loop Adaptive RL for Robotic Systems.md`).
Diese Datei ist ein Übergabezettel; wenn die Aufgabe begonnen ist, löschen (Git ist das Archiv).

## Die Aufgabe

Das ist der Kern des Projekts. In jedem Regeltakt:

```
u_k     = act(θ_k, x_k, ref_k)                                   # fly
x_{k+1} = plant.step(u_k)                                        # measure
φ       ← model update on the buffer of recent real (x, u, x⁺)   # 1) model first
θ_{k+1} = θ_k − lr · ∇ rollout_loss(θ; x_{k+1}, ref, hybrid(φ))  # 2) then policy
```

`rollout_loss`: vom gerade gemessenen Zustand H Schritte mit der aktuellen Policy im
Hybridmodell vorausrollen, Kosten auf den vorhergesagten Zuständen, BPTT durch diesen
Rollout (#17). Ableitungen wie LotF nur durch das nominale Modell (J1, #11); Ableitung
durch das Residuum (J2) nur als Option über `grad_through_residual`. Sonst keine Varianten.
Der Puffer echter Tupel dient nur dem Modell-Fit.

Ziel, das der Nutzer sehen will: Modell lernt nach einer Störung → vorhergesagte Rollouts
zeigen die Drift → Gradient korrigiert → Policy passt sich an.

**Erst zuhören, dann Entscheidungen vorlegen (Optionen, Empfehlung, Begründung), auf seine
Wahl warten.** Er wird Parameter selbst durchprobieren.

## Stand

- Commit-Stand bis „Add residual policy with online first-order adaptation“ (#20).
- `pipelines/first_order/gradient.py` rechnet den Gradienten über den Modell-Rollout (#17);
  `run_adaptive.py` übergibt ein untrainiertes Residuum (hybrid = nominal).
- Experiment: Kreis, Masse ×1,3 ab t = 10 s, Policy-Update in jedem Takt ab 2 s.
  Baseline: 2,2 / 32 / 50 cm (5–10 / 10–12 / 15–20 s). FO mit nominalem Modell: Absturz
  bei 15,4 s, Schubkorrektur läuft ins Negative (#20). Erwartet: mit Online-Modell positiv.
- Nächster Schritt: Online-Modell-Update (Entscheidungen unten).

## Kontext, der nicht (leicht) aus dem Code zu sehen ist

1. **Wie das Residuum unter J1 wirkt:** Seine Ableitung ist abgeschnitten, aber es
   bestimmt die vorhergesagten Zustände und damit, wo Kosten und nominale Jacobi
   ausgewertet werden. Mit veraltetem Residuum sieht der Rollout nach einer Störung nur den
   aktuellen Versatz, mit nachgelerntem die weitere Drift.
2. **Was J2 zusätzlich bringen kann (durchgerechnet):** Target a_res = (v⁺_mess − v⁺_nom)/dt
   ≈ −0,23·(T/m)·b₃ bei heavy, linear im Feature T/(m g). Lernt das Netz die Steigung,
   wird ∂v⁺/∂T im Hybrid 0,77·dt/m, also richtig (nominal ~30 % zu groß). Bei konstantem
   Wind ist J1 = J2. Rotation und Ratendynamik korrigiert das Residuum nie (#6, #12).
3. **Identifizierbarkeit online (offen).** Ohne Anregung ist u = Policy(x) eine Funktion von
   x. Das Residuum kann dann die Wirkung von u nicht von der von x trennen: ∂a_res/∂u ist
   beliebig (wichtig für J2), und die Vorhersage für u, die anders sind als die geflogenen,
   ist unzuverlässig (wichtig für den Rollout). Offline deshalb gehaltenes Rauschen auf u
   (#7). Optionen: `HeldNoise` auch online, Variation durch Referenz und Policy-Änderungen
   reicht, oder Ableitung regularisieren. Dem Nutzer vorlegen.
4. **Wahre vs. nominale Ableitung, gemessen über einen Schritt** (`Plant.snapshot/true_step`,
   Kreis, nominal): ∂v_z⁺/∂T wahr 0,015, nominal 0,23. ∂ω⁺/∂ω_cmd wahr 0,32/0,28/0,14,
   nominal 1. Ursache: Rotor- und Ratendynamik. Über H Schritte nicht gemessen (#19).
   Annahme: über H gleicht sich das teilweise aus.
5. **JIT-Falle beim Modell im Rollout:** Das Residuum-Netz φ als Argument (nnx-Modul) an
   die gejittete Funktion übergeben, nicht über eine Closure einfangen. Sonst backt `jit`
   die Gewichte beim ersten Trace als Konstanten ein, und spätere Modell-Updates werden
   still ignoriert. `baseline` ist dagegen bewusst statisch (`static_argnames`, Hash über
   Identität).
6. **Modell-Update-Baustein existiert:** `fit.train_step(model, optimizer, F, Y)` (Adam,
   MSE + Spektralnorm), `fit.training_data(run)` für Features/Targets. Offline gilt:
   Full-Batch, ab ~40–130 Schritten Überanpassung, Fehlerboden ~0,1 (m/s²)² bei allen
   Bedingungen (vermutlich Schubverzögerung, ungeprüft) (#14). Online fehlt alles drum
   herum: Puffer, Vergessen, Schritte pro Takt. Für das Modell ist Adam nicht ausgeschlossen;
   die SGD-Regel (#20, P5) gilt für die Policy.
7. **Offline-Residuum als Start ist zweischneidig:** Kreuztest #16. Ein Residuum aus einer
   anderen Bedingung ist schlechter als gar keins (heavy-Residuum auf nominal: 2,2 → 7,0 cm
   bei k = 30). Nach dem Wechsel muss das Online-Modell alte Daten schnell vergessen.
   Optionen: Start von null (Null-Init, hybrid = nominal), vom nominal-Offline-Modell,
   oder vom passenden.
8. **Aktivierung:** Die gespeicherten Residuen sind ReLU. Für J2 ist eine glatte Aktivierung
   (tanh/softplus) wahrscheinlich nötig (stückweise konstante Ableitung). Sie ist ein
   Konstruktor-Argument, `load_model` baut sie über `getattr(nnx, name)` neu. Neues
   Training nötig (`train_model.py`). Erst prüfen ob wirklich nötig.
9. **Einheiten und Gewichte:** Aktionsgewicht 0,01, nicht LotFs 0,1 (Schub in m·g statt N).
   lr = 1e-2 vorläufig, mit Rollout neu prüfen.
10. **Reihenfolge im Takt:** erst Modell, dann Policy, damit der Gradient das frischeste
    Modell nutzt. Lernen erst ab 2 s (Rotor-Anlauf, #4). Störung ab Start des Laufs
    oder mitten im Lauf über `plant.set_disturbance`.
11. **LotF-Vergleich:** LotF lernt das Residuum offline und nutzt es im Fine-Tuning mit
    Resets. Wie LotF online nachlernt, wurde **nicht** geprüft. Der Code liegt nicht im
    Repo, bei Bedarf klonen (github.com/uzh-rpg/learning_on_the_fly, in den Scratchpad).
    Relevante Stellen: `lotf/objects/quadrotor_obj.py` (`step`, `custom_jvp`),
    `lotf/utils/residual_dynamics.py`, `lotf/algos/bptt.py`.

## Offene Entscheidungen für diesen Schritt (vorlegen, nicht selbst treffen)

- Puffer: Größe, gleitendes Fenster vs. Vergessen über Gewichtung; ein oder mehrere
  Modellschritte pro Takt; Optimierer und lr fürs Modell.
- Start des Residuums (Punkt 7), Aktivierung (Punkt 8).
- Anregung online (Punkt 3).
- Metriken: Tracking-Fehler über die Zeit, Online-k-Schritt-Fehler des Modells
  (`hybrid.rollout_errors`), Gradient gegen wahr (#19, `Plant.true_step`).
- Wie lange nach dem Wechsel das Modell braucht, bis der Gradient hilft.

## Sonst offen (decisions.md, ToDo)

#9 Messrauschen, #13 ortsabhängige Störung (beide bewusst verschoben), #15 `lax.scan`,
#18 Rendern (Replay aus aufgezeichneten Läufen, Anfang Closed-Loop-Teil),
#19 Gradient gegen wahr. Danach: ZO-Zweig (`pipelines/zeroth_order/`) auf denselben
Modell-Rollouts.

## Arbeitsweise (Kurzform aus CLAUDE.md und Erfahrung)

- Deutsch, Code und Kommentare Englisch. Nutzer entscheidet, reviewt und committet; du
  bereitest die Commit-Message vor.
- Messbare Fragen mit Messungen an unseren Daten entscheiden; Wegwerf-Skripte im Scratchpad.
- Weicht ein Messergebnis von einer getroffenen Entscheidung ab: nicht still ändern,
  vorlegen.
- Am Ende jedes Schritts nötige Änderungen an Modulkarte und Blockschaltbild (GoodNotes)
  nennen. Offen: Modulkarte um `policy.py`, `pipelines/first_order/gradient.py`,
  `run_adaptive.py` ergänzen; Blockschaltbild um Policy-Schleife (x_k → Modell-Rollout →
  Gradient → θ, z⁻¹) und Modell-Schleife (Puffer → φ).
