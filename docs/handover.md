# Handover: nach dem Online-Modell-Update

Für den nächsten Coding Agent. Erst lesen: CLAUDE.md, die Projekt-Charter
(`../Project Charter Closed-Loop Adaptive RL for Robotic Systems.md`, maßgeblich),
`docs/decisions.md` (vor allem #11, #17, #20, #22), `docs/supervisor_overview.md`, `git log`.
Diese Datei ist ein Übergabezettel; löschen, sobald die Arbeit weitergeht (Git ist das Archiv).

**Zuerst den Nutzer fragen, was beim Supervisor-Treffen am 2026-11-24 herauskam.** Dort
wurden die Architekturentscheidungen A1–A9 aus `supervisor_overview.md` besprochen; das kann
die Reihenfolge unten ändern.

## Stand (Commit 183e92e)

Pro Regeltakt in `scripts/run_adaptive.py`:

```
u_k = act(θ, x_k, ref_k) (+ e_k, gehaltenes Schubrauschen, optional)  → plant → x_{k+1}
(f, y) = online.sample(x_k, u_k, x_{k+1})  → SlidingWindow (letzte W s)
→ fit.train_step (1 Adam-Schritt, warm)                                → Residuum φ
→ gradient.update: Rollout H = 30 im Hybridmodell ab x_{k+1}, BPTT (J1) → 1 SGD-Schritt auf θ
```

Ergebnisse (Kreis, Masse ×1,3 ab 10 s), Details in #22: nominales Modell → Absturz;
Online-Modell 1 s / 2 s Fenster → 11 / 14 cm (Baseline 50 cm); mit Schubrauschen 0,1 m·g
→ 1,7 / 2,7 cm, aber 4–9 cm statt 2,2 cm vor dem Wechsel; 5-s-Fenster scheitert immer.

## Offen (vom Nutzer noch nicht priorisiert; Optionen vorlegen, nicht selbst wählen)

1. **Rauschamplitude** 0,05 / 0,1 / 0,2 m·g: gibt es einen Bereich, in dem der Gewinn nach
   dem Wechsel den Preis davor überwiegt?
2. **Vermutung prüfen: Schubverzögerung.** Mit Rauschen springt das Target a_res vor dem
   Wechsel um ±2–3 m/s². Vermutung: Rotorverzögerung, die das Nominalmodell nicht kennt (#6);
   das Residuum erklärt sie aus dem aktuellen Zustand und die Policy korrigiert eine
   Störung, die es nicht gibt. Prüfen: Korrelation des Targets mit dem Rauschschritt des
   vorigen Takts.
3. **Warum 5 s scheitert, auch mit Rauschen.** Diagnose ohne Rauschen: das Netz koppelt
   a_res an v_z (∂a_res,z/∂v_z ≈ +6 … +16 statt ≈ 0), im Rollout steigt die Drohne und die
   Störung verschwindet. Vermutung mit Rauschen (ungeprüft): 0,1 m·g bewegt v_z zu wenig.
   Betrifft auch exponentielles Vergessen.
4. **Langsame Restkonvergenz ohne Rauschen:** Schubkorrektur 0,25 statt 0,30 m·g bei 20 s.
   Nicht untersucht.
5. **Rendern (#18):** matplotlib-GIF existiert (`render_runs.py`). Offen: MuJoCo-Replay mit
   Drohnenmodellen für den Bericht (~halber Tag, Farben je Drohne und Offscreen-Rendering auf
   macOS ungeprüft).
6. **Notiert für später** (#22 und ToDo): exponentielles Vergessen; Modell jeden Takt,
   Policy jeden 2. Takt; Start des Residuums aus Offline-Modell (Ablation); Policy-Update
   aussetzen bei schlechtem Modell; Ensemble; J2 mit glatter Aktivierung; #19 Gradient gegen
   wahr; Wind-Experiment; Looping (Charter); ZO-Pipeline auf denselben Rollouts; #9, #13, #15.

## Kontext, der nicht im Code steht

- **Läufe nie parallel starten.** Zwei JAX-Prozesse bremsen sich um >10× (Threadpools über
  alle Kerne). `run_adaptive.py --windows 1 2 5` allein: ~2 min.
- **Der Gradient kommt nur aus Modell-Rollouts** (#17, Charter). Ein Gradient entlang der
  geflogenen Trajektorie ist ausgeschlossen und soll nicht vorgeschlagen werden.
- **Ein veraltetes Modell lenkt die Policy aktiv falsch** (#20): Im Rollout zieht die Baseline
  die zu tiefe Drohne selbst hoch, der Gradient dämpft → weniger Schub. Jede Phase, in der
  das Modell hinterherhinkt, ist gefährlich, nicht nur langsam.
- **Der A-priori-Fehler am geflogenen Zustand reicht als Modellmetrik nicht.** Entscheidend
  ist die Vorhersage entlang des vorgestellten Rollouts (andere u, andere v als geflogen):
  `rollout_a_res_z` in `run_adaptive.py`, viertes Panel.
- **Diagnose-Technik** (Skripte lagen im Scratchpad, nicht im Repo): `run_adaptive` importieren,
  `ra.gradient.update` durch einen Wrapper ersetzen, der am übergebenen Zustand mit `jax.grad`
  die Ableitung des Residuums nach einzelnen Features misst (Feature-Index: R 0–8, v 9–11,
  ω_cmd 12–14, T/(m g) 15), dann `ra.fly(...)` aufrufen.
- **JIT-Falle:** Das Residuum geht als Argument in `gradient.update`, nicht per Closure; sonst
  ignoriert der kompilierte Code spätere Modell-Updates (getestet in `test_policy.py`).
- **Einheiten:** Policy-Ausgang Schub in m_nom·g; Aktionsgewicht 0,01 (nicht LotFs 0,1).
- **LotF online** (Paper, nicht Code): rollender Puffer 2–5 s bei 50 Hz, Residuum alle 3 s
  100 Iterationen, Ensemble aus 3, Policy alle 5 s. Im Repo nur Offline-Training.
- **`supervisor_overview.md`** sagt im Kopf, was hineingehört (Architekturentscheidungen,
  Ergebnisse, offene Fragen). Neue Ergebnisse dort nachtragen.

## Arbeitsweise (Kurzform)

Deutsch, Code und Kommentare Englisch. Nutzer entscheidet Theorie und Architektur, reviewt,
committet; du bereitest die Commit-Message vor. Messbare Fragen mit Messungen entscheiden,
Wegwerf-Skripte im Scratchpad. Debuggen: erwartet → tatsächlich → wo es abweicht → warum →
erst dann Fix. Am Ende jedes Schritts die Änderungen für Modulkarte und Blockschaltbild
(GoodNotes) nennen, aus dem Code geprüft.
