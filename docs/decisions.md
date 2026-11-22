# Entscheidungen

Kurzes Log der Design-Entscheidungen. Pro Eintrag: was, warum, verworfene Alternativen.
Details (Messungen, Diskussion) stehen im Commit, der die Entscheidung umsetzt.
Nummern bleiben fest; ein ToDo, das entschieden wird, wandert mit seiner Nummer nach oben.
Überholte Entscheidungen werden nicht gelöscht, sondern als „ersetzt durch #n“ markiert.

## Entschieden

### 1. Steuerung über Body Rates + kollektiven Schub (Projektstart)
`Control.body_rate` mit `Dynamics.first_principles`. Nominales Modell wird dadurch
Punktmasse + Rotationskinematik (wie LotF), keine Euler-Singularitäten (Looping möglich),
Policy-Rate 100–200 Hz reicht.
Verworfen: `state` (Regler macht alles), `attitude` (Euler-Singularität bei ±90°),
`force_torque`/`rotor_vel` (≥ 500 Hz, Trägheit + Motoren im Surrogat nötig).

### 2. Raten: Policy 100 Hz, Physik und innerer Ratenregler 500 Hz (2026-11-17)
`body_rate_freq` ist die Rate von Crazyflows innerem Regler, nicht die der Policy.
u wird über 5 Physikschritte gehalten. LotF nutzt 50 Hz; Crazyflow empfiehlt ≥ 200 Hz
für body_rate. 100 Hz als Kompromiss aus BPTT-Länge und Modellfehler.

### 3. Plant-Schnittstelle (2026-11-17)
`reset() → x`, `step(u) → x⁺`; nur `plant.py` kennt Crazyflow. Tupel (x, u, x⁺) baut
die Schleife, nicht die Plant. x = [p, R (row-major), v, ω_body], 18-dim.
Crazyflows Lageterme im Ratenregler (`kR`, `ki_m`) auf 0 → reiner Ratenkanal.

### 4. Kein Hover-Init der Rotoren (2026-11-17)
Rotoren starten bei 0 RPM, Drohne sinkt anfangs ~16 cm. Bewusst einfach gehalten.
Folge: erste ~2 s eines Laufs nicht als Trainingsdaten verwenden (Rotordrehzahl ist
versteckter Zustand, nicht in x).

### 5. Baseline-Regler geometrisch, ohne Integrator (2026-11-17)
Position-PD + Vorsteuerung → Soll-Lage auf SO(3) → Raten. Zustandslos, damit
`u = controller(x, ref)` dieselbe Signatur wie eine Policy hat.
Preis: bleibender Fehler bei konstanten Störungen (Kreis: ~1,6 cm).

### 6. ω im nominalen Modell: ω = ω_cmd, Schub sofort (2026-11-19)
Wie das LotF-Surrogat (`simplified_dyn`): R⁺ = R·exp(ω_cmd·dt), a = T/m·b₃ + g, keine
Ratendynamik, keine Totzeit. `f(x, u)` bleibt gedächtnislos; ω in x wird nicht genutzt,
ω⁺ := ω_cmd. Gründe: Vergleichbarkeit mit LotF; das Modell hat eine bekannte, benennbare
Abweichung (fehlende Ratendynamik), die den First-Order-Gradienten verzerrt und den
FO/ZO-Vergleich sichtbar macht; klassisches flaches Quadrotormodell.
Gemessen (nominal/test_circle, Open-Loop mit echten u, Fit auf nominal/train):
| ω-Modell | Lage k=10/30 | Position k=10/30 |
|---|---|---|
| ω = ω_cmd | 1,88°/1,99° | 0,34/2,20 cm |
| 1. Ordnung (τ ≈ 25/26/64 ms) | 0,80°/0,83° | 0,34/2,19 cm |
| 2. Ordnung (roll/pitch: f_n ≈ 9 Hz, ζ ≈ 0,35) | 0,14°/0,30° | 0,34/2,20 cm |
Das ω-Modell ändert die Positionsvorhersage bis k = 30 nicht (heavy: 13,7 cm bei allen);
der Positionsfehler kommt von Masse, Schubverzögerung, Widerstand → Aufgabe des Residuums.
Roll/Pitch sind unterdämpft 2. Ordnung (Ratenregler-D-Anteil + Rotoren); eine 1. Ordnung
erklärt nur 60 % der Varianz von Δω. Die früheren τ ≈ 30/31/69 ms sind dadurch messmethodenabhängig.
Verworfen: 1. Ordnung (halbiert nur den Lagefehler); 2. Ordnung mit Historie (ω, u des
Vorschritts im Modellzustand), bleibt mögliche Ablation für die Gradientenqualität;
Totzeit-Puffer (LotF braucht ihn für Hardware-Latenz, unsere Plant hat keine Latenz).

### 7. Anregung und Datensätze für das Dynamikmodell (2026-11-17)
- Training: variierte Referenz (Summe von Sinus, fester Seed) für Breite der Zustände.
- Zusätzlich gehaltenes Rauschen auf u (neuer Wert alle 10 Schritte, 0,5 rad/s,
  10 % Schub) für Unabhängigkeit von u und x: ohne Rauschen ist u = Regler(x),
  und die Wirkung von u (= was BPTT braucht) ist nicht identifizierbar.
  Weißes Rauschen verworfen: die Regelkreise (τ 30–70 ms) filtern es weg.
- Rauschen sitzt in der Schleife, nicht in der Plant (gilt so auch an echter Hardware);
  e wird separat gespeichert (später für den Zeroth-Order-Gradienten).
- Test: feste Trajektorien (Hover, Kreis), gleiche Rauschstatistik, eigener Seed.
  Nicht in den Trainingsdaten → misst Verallgemeinerung, kein Leak.
- LotF: kein Aktionsrauschen im Code, Vielfalt über zufällige Resets. Für uns
  (ohne Resets) nicht verfügbar.

### 8. Störungen und nominale Masse (2026-11-17)
`plant.set_disturbance(mass=..., force=...)`: wahre Masse und konstante Kraft (Welt).
Bleibt aktiv bis zum nächsten Aufruf oder `reset()`. Die Plant gibt nur noch
`nominal_mass` heraus (fest beim Erzeugen); die wahre Masse wäre sonst ein Orakel für
Regler und Modell. Massenänderung skaliert J nicht mit (vereinfachend).
Offline-Datensätze: je Bedingung (nominal, Masse +30 %, Kraft) eigene Train/Test-Läufe,
Störung ab Start aktiv. Umschalten mitten im Lauf erst im Closed-Loop-Teil.

### 10. Gemeinsamer Vertrag in `state.py` und `constants.py` (2026-11-19)
Layout von x und u (`STATE_DIM`, `CMD_DIM`, `unpack_state`, Frames, Einheiten) in
`state.py`, `GRAVITY` in `constants.py`; beide laden nicht den Simulator (`state.py` nutzt
JAX für `pack_state`, das auch unter `jit` funktionieren muss). Vorher lagen sie in `plant.py`
bzw. `controllers.py`: der Regler lud über `unpack_state` den Simulator mit, und
`nominal.py` hätte von Plant und Baseline-Regler abhängen müssen. Ändert #3 nicht
inhaltlich, nur den Ort der Definition. g = 9,81 wie in Crazyflow.
Verworfen: alles in `plant.py` lassen; Layout in `nominal.py` duplizieren (zwei Quellen,
die still auseinanderlaufen können).

### 12. Residuum: Eingänge, Ziel, Netz (2026-11-19)
`dynamics/residual.py`, `dynamics/hybrid.py`. Wie LotF: MLP 2 × 128, ReLU, Ausgang
Restbeschleunigung (3) im Weltframe, eingespeist über `a_res` in `nominal.step`;
One-Step-MSE + Spektralnorm-Regularisierung (λ = 1e-3). Rotation wird nicht korrigiert.
Ziel: `a_res = (v⁺_gemessen − v⁺_nominal) / dt`, berechnet über `nominal.step`.
Abweichungen von LotF:
- Eingänge R, v, ω_cmd, T/(m·g) (16), **ohne p**: Störungen bisher ortsunabhängig; mit p
  lernt das Netz den Ort der Trainingstrajektorie mit und extrapoliert online beliebig.
  p kommt dazu, sobald eine ortsabhängige Störung untersucht wird (#13). Ohne gemessenes ω.
- Feste physikalische Skalen statt Daten-Normierung (LotF: keine): bleibt online gültig.
- Ein Netz statt Ensemble aus 3 (LotF nutzt nur den Mittelwert).
- Letzte Schicht mit 0 initialisiert: untrainiert gilt hybrid = nominal.
Offline je Bedingung ein Netz (Störung ist kein Eingang). Aktivierung ist Argument, weil
#11 einen glatten Gradienten durch das Netz verlangen könnte.

### 14. Offline-Training des Residuums (2026-11-22)
`dynamics/fit.py` (Loss, `train_step`, `fit`, Speichern), `scripts/train_model.py`.
Wie LotF: Full-Batch (ein Schritt = eine Epoche; LotFs `batch_size = 256` wird nie benutzt),
Adam lr = 1e-2, Loss = MSE + 1e-3 · Σ‖W‖₂ über alle Schichten.
Abweichungen von LotF:
- Validierung: letzte 20 % des Train-Laufs als zusammenhängender Block (Zeilen-Zufallssplit
  testet auf Nachbarn im 10-ms-Abstand). LotF validiert nicht.
- Early Stopping durch Auswahl: bis 500 Schritte, Val alle 10, beste Gewichte behalten.
  Statt fest 100 (LotF): das Optimum hängt von der Bedingung ab (bester Schritt 40/130/60
  für nominal/heavy/wind, Seed 0); danach steigt Val wieder (Überanpassung).
  Val-MSE [(m/s²)²] a_res=0 → bestes: nominal 0,23 → 0,10; heavy 3,25 → 0,07; wind 0,69 → 0,10.
  Rest ~0,1 bei allen gleich, vermutlich Schubverzögerung (Rotordrehzahl, versteckter Zustand).
- Speichern als `.npz` (Gewichte flach, `meta` als JSON, Lernkurve) statt orbax: ein Format
  im Projekt, kein weiterer Import; orbax ist für große, verteilte Checkpoints gebaut.
- Targets werden bei jedem Training aus (X, U, X⁺) neu berechnet, nicht gespeichert: sie
  hängen vom Nominalmodell ab und würden sonst still veralten.
Verworfen: Minibatch (nicht deterministisch, Vorteil klein gegen die Wahl der Schrittzahl).

## ToDo

### 9. Messrauschen / Zustandsschätzung
Die Plant liefert den exakten Zustand; eine echte Drohne liefert eine Schätzung
(IMU, Motion Capture) mit Rauschen und Verzögerung. Für „Plant 1:1 austauschbar“ sollte
`step()` einen verrauschten Zustand liefern und den wahren nur separat zur Auswertung.
Offen: wann einführen, Rauschmodell (Rauschen auf R muss auf SO(3) bleiben),
errors-in-variables beim Lernen. Vor dem Closed-Loop-Teil entscheiden.

### 11. Wodurch fließt der Policy-Gradient?
LotF (`Quadrotor.step`, `custom_jvp`): Vorwärtsschritt mit Residuum, Gradient nur durch
das analytische Modell (Residuum ohne Ableitung). Alternative: Gradient durch das ganze
Hybridmodell. Zentral für den FO/ZO-Vergleich. Vor dem Policy-Teil entscheiden.

### 13. Ortsabhängige Störung
Z. B. Windzone auf einem Teil der Kreisbahn. Zeigt den Unterschied „Störung verfolgen“
(Residuum ohne p, online nachtrainiert) gegen „Karte lernen“ (mit p, sagt die Zone voraus).
Braucht: Plant-Erweiterung (Kraft als Funktion von p), neue Datensätze, Residuum mit/ohne p.
Vor dem Closed-Loop-Teil entscheiden.

### 15. Trainingsschleife in `lax.scan`
`fit` ist eine Python-Schleife um den `nnx.jit`-Schritt (lesbar, Early Stopping und Logging
in normalem Python; ~10 s pro Bedingung). Schneller: ganze Schleife in `jax.lax.scan` wie
LotF, dann Auswertung und beste Gewichte im kompilierten Code. Relevant, sobald das
Training oft läuft (Sweeps, Online-Nachtraining).
