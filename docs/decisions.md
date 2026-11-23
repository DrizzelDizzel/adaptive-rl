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

### 16. Auswertung des Dynamikmodells (2026-11-22)
`hybrid.rollout_errors`, `scripts/eval_model.py`. Open-Loop-Rollout mit den aufgezeichneten
u ab jedem gemessenen Zustand der Testläufe (Fenster überlappend, k = 1…50), Fehler
‖p̂ − p‖, ‖v̂ − v‖, Lagewinkel; Mittelwert und p90 über die Startpunkte (Metrik wie #6).
Kein Bestanden-Kriterium, berichtet wird das Verhältnis hybrid/nominal. Lage nur als
Kontrolle: das Residuum korrigiert sie nicht (#12), beide Modelle müssen gleich sein.
LotF hat keine Mehrschritt-Auswertung des Residuums.
Ergebnis Position k = 30 (test_circle / test_hover), nominal → hybrid:
heavy 13,7 → 1,1 / 13,5 → 1,0 cm; wind 5,4 → 1,3 / 5,4 → 1,2 cm;
nominal 2,2 → 1,2 / 1,24 → 1,27 cm (Hover: kein Gewinn, ab k ≈ 25 leicht schlechter).
Kreuztest (Residuum einer anderen Bedingung, test_circle, k = 30): schlechter als nominal,
z. B. heavy-Residuum auf nominal-Daten 2,2 → 7,0 cm. Ein offline gelerntes Residuum
schadet, sobald sich die Situation ändert → Motivation für Online-Anpassung.
Verworfen: Fenster ohne Überlappung (~58 bei k = 30, verrauscht); feste Schwelle in cm.

### 11. Welche Jacobi-Matrizen benutzt BPTT? (2026-11-22)
A = ∂f/∂x, B = ∂f/∂u im Rollout (#17). Schalter `hybrid.step(..., grad_through_residual)`.
- J1 (Standard, wie LotF `Quadrotor.step`, `custom_jvp`: vorwärts genaues Modell, rückwärts
  `simplified_dyn`): a_res per `stop_gradient` konstant → Jacobi = nominal, exakt, weil a_res
  additiv eingeht. Das Residuum wirkt trotzdem auf den Gradienten: über die vorhergesagten
  Zustände, an denen Kosten und Jacobi ausgewertet werden.
- J2 (nur Option): Kette auch durch das Netz. Zählt bei Störungen, die die Wirkung von u
  ändern (Masse: a_res hängt von T ab); bei additiver, zustands- und eingangsunabhängiger
  Störung (konstanter Wind) ist J1 = J2. Bewertet per Messung gegen die wahre
  Crazyflow-Jacobi (`jax.jacfwd` durch Crazyflows reine Step-Funktion, geprüft: endlich,
  stimmt mit finiten Differenzen), #19.
Erste Messung (ein Schritt, Kreis, nominal): wahres ∂v_z⁺/∂T = 0,015, nominal 0,23;
wahres ∂ω⁺/∂ω_cmd = 0,32/0,28/0,14 (Diagonale), nominal 1. Grund: Rotor- und
Ratendynamik (#6); B des Nominalmodells ist über einen Schritt um Faktor 3–15 zu groß.
Wahre Dynamik zur Auswertung: `Plant.snapshot()` (voller Crazyflow-Zustand inkl. Rotor-
drehzahlen) und `Plant.true_step(state, u) → (state⁺, x⁺)`, rein und differenzierbar. Nur
plant.py kennt Crazyflow (#3); die Regelschleife benutzt beides nie. Die `.npz`-Läufe
reichen dafür nicht: die wahre Jacobi hängt vom versteckten Zustand ab.

### 17. Policy-Gradient aus Rollouts im Hybridmodell (2026-11-23)
Projekt-Charter: „The finite-horizon rollouts live inside the model“, FO und ZO auf dem
Surrogat. In jedem Regeltakt: vom gerade gemessenen x_k H Schritte mit der aktuellen Policy
und der Referenz ref_k … ref_{k+H} im Hybridmodell ausrollen, Kosten auf den vorhergesagten
Zuständen, BPTT durch diesen Rollout (Ableitungen #11). LotF-Prinzip.
- Die echten Tupel (x, u, x⁺) im Puffer dienen nur dem Modell-Fit. Der Gradient braucht
  vom echten Flug nur x_k.
- Das Modell zählt direkt: ein genaueres Residuum sagt Drift durch die Störung voraus, der
  Gradient korrigiert vorausschauend. Mit veraltetem Residuum sieht der Rollout nur den
  aktuellen Versatz.
- Preis: Vorwärtsfehler summieren sich über H (#16: nominal ~2,2 cm, hybrid ~1,2 cm bei
  k = 30 auf dem Kreis).

### 20. Policy und First-Order-Anpassung im Closed Loop (2026-11-22)
`policy.py`, `pipelines/first_order/gradient.py`, `scripts/run_adaptive.py`.
- P1 Residual-Policy: u = Baseline(x, ref) + π_θ, letzte Schicht mit 0 initialisiert →
  untrainiert exakt die Baseline, stabil ab t = 0 (ohne Resets Pflicht). LotF: u_hover + π,
  offline vortrainiert.
- P2 Eingänge: e_p/0,1 m, e_v/0,5 m/s, R, a_ref/g (18); keine absolute Position, kein ω.
  LotF: p, R, v, letzte Aktionen, keine Referenz (Policy bahngebunden).
- P3 2 × 32, tanh (~1,8k Parameter; LotF 2 × 512 ReLU): ZO-Rauschen wächst mit der
  Parameterzahl, BPTT braucht glatte Ableitungen. Ausgang Raten in rad/s, Schub in m_nom·g.
- P4 Kosten wie LotF (smooth_L1 auf e_p, e_v), Aktionsterm auf die Korrektur π statt
  u − u_hover (Baseline-Kommando auf dem Kreis ist weit vom Schweben). Gewicht: siehe unten.
- P5 Rollout H = 30 im Hybridmodell ab dem gemessenen x_k (#17), Ableitung J1 (#11), inkl.
  Rückführung über die Baseline. Ein SGD-Schritt pro Regeltakt (kein Adam: würde
  FO-Verzerrung gegen ZO-Rauschen verdecken), Clipping auf Gradientennorm 1. Lernen ab 2 s (#4).
- P6 Baseline-Regler in JAX (`cascaded_control`), gleiche Werte wie numpy (rel. 6e-7),
  `fly_circle.py` unverändert 1,6 cm (#5).
- P7 Erstes Experiment: Kreis, Masse ×1,3 ab t = 10 s, gegen reine Baseline. #9, #13
  bewusst verschoben, bis FO und ZO laufen.
Gemessen (mittlerer Positionsfehler [cm], 5–10 / 10–12 / 15–20 s): Baseline 2,2 / 32 / 50
(ohne Integrator: Höhenfehler ≈ 0,23·g/k_p).
Aktionsgewicht 0,01 statt LotFs 0,1: LotFs Wert gilt in N und rad/s; unser Schub ist in m·g
normiert, dieselbe Zahl wirkt dadurch ~6× stärker. lr = 1e-2 vorläufig.
FO mit Modell-Rollout (#17), Residuum untrainiert (hybrid = nominal), H = 30, lr = 1e-2
(2026-11-23): 2,1 / 31 cm, dann Absturz bei 15,4 s. Die Schubkorrektur wird negativ
(−0,65 m·g bei 15 s). Grund: Das Nominalmodell weiß nichts von der Masse; im Rollout zieht
die Baseline die zu tief fliegende Drohne selbst hoch, und der Gradient dämpft vor allem den
wachsenden Geschwindigkeitsfehler → weniger Schub → größerer Fehler. Einzelner Update-Schritt
(Testfall): 5 cm unter der Referenz, nominal: −0,012 m·g (H = 30); auf der Referenz, Residuum
kennt −0,23 g: +0,11 m·g. Ein veraltetes Modell lenkt die Policy also aktiv in die falsche
Richtung; erst das Online-Modell macht die Anpassung möglich.

### 22. Online-Update des Dynamikmodells (2026-11-23)
`dynamics/online.py`, `scripts/run_adaptive.py`. Pro Regeltakt erst Modell, dann Policy.
- Grundsatz: Das Modell muss mindestens so schnell nachlernen wie die Policy, eher schneller.
  Ein veraltetes Modell lenkt die Policy in die falsche Richtung (#20).
- Puffer: gleitendes Fenster der letzten W s echter Tupel (x, u, x⁺), als (Features, Target)
  wie offline (#14); Ringpuffer fester Größe (jit kompiliert pro Array-Form einmal).
  LotF: rollender Puffer, 2 s (Hover) / 5 s (Trajektorie) bei 50 Hz.
- Pro Takt k Adam-Schritte (Default 1, lr 1e-2 wie #14) auf dem ganzen Fenster, Gewichte
  warm weiter. LotF: alle 3 s 100 Iterationen, Ensemble aus 3 Netzen, parallele Threads.
  Offline brauchte Adam 40–130 Schritte von null → bei 1 Schritt/Takt ~0,4–1,3 s.
- Start: Null-Init (hybrid = nominal), alles Gelernte stammt aus dem Flug. Offline-Start
  („eventually pretrained“, Charter) später als Ablation. ReLU bleibt (unter J1 wird die
  Ableitung des Netzes nicht benutzt). Ein Netz, kein Ensemble (#12).
- Daten ab 2 s (#4); Modell und Policy lernen, sobald das Fenster voll ist (2 s + W).
- Keine Anregung online und kein Aussetzen der Policy bei schlechtem Modell: erst messen.
Später: exponentielles Vergessen (Gewicht λ^Alter, wie RLS) statt hartem Fenster;
Modell jeden Takt, Policy nur jeden 2. Takt.
Gemessen (Kreis, Masse ×1,3 ab 10 s, H = 30, Policy-lr 1e-2, 1 Modellschritt/Takt;
Positionsfehler [cm] 5–10 / 10–12 / 15–20 s): Baseline 2,2 / 32 / 50; FO nominal Absturz;
Fenster 1 s: 2,3 / 6,9 / 11,5; 2 s: 2,3 / 5,3 / 13,7; 5 s: 2,2 / 20 / 65 (schlechter als
Baseline). Schubkorrektur bei 1 s / 2 s ~+0,25 m·g bei 20 s (nötig 0,30), noch steigend.
Das Modell trifft a_res am geflogenen Zustand bei allen Fenstern nach ~0,5 s (a priori
0,03–0,10 m/s² ab 15 s); die erwartete Mischung alt/neu im Fenster tritt nicht auf.
Ursache des 5-s-Falls (Diagnose, Scratchpad): Das Netz hat a_res an v_z gekoppelt
(∂a_res,z/∂v_z = +6 … +16 statt ≈ 0), weil im Fenster „sinkt“ mit „a_res negativ“ und
„v_z ≈ 0“ mit „a_res = 0“ (vor dem Wechsel) zusammenfällt. Im Rollout steigt die Drohne
(v_z ≈ +0,2–0,3), das Modell sagt dort a_res ≈ −0,3 … −0,7 statt −2,9 voraus → nominaler
Fall (#20), negative Korrektur. Bei 2 s ist die Kopplung schwächer (+2 … +6), der
Rollout-Mittelwert meist −2,3 … −2,9. Identifizierbarkeit (#7) bestätigt; der
A-priori-Fehler am geflogenen Zustand zeigt das nicht. Deshalb zeichnet `run_adaptive.py`
jetzt zusätzlich das mittlere a_res,z entlang des vorgestellten Rollouts auf
(`gradient.imagine`, derselbe Rollout wie im Gradienten); ohne Rauschen −2,5 / −2,6 / −1,0
(10–12 s) für 1 / 2 / 5 s.
Anregung online (`--noise`, gehaltenes Schubrauschen, nur Schub, alle 10 Takte, nur in den
Online-Läufen), 0,1 m·g wie offline (#7): 1 s: 8,6 / 2,2 / 1,7 cm; 2 s: 5,7 / 4,3 / 2,7 cm;
5 s: 4,0 / 13 / 58 cm. Nach dem Wechsel erreicht die Schubkorrektur bei 1 s / 2 s ~0,30 m·g
(ohne Rauschen 0,25). Preis vor dem Wechsel: 2,2 → 4–9 cm, die Korrektur schwankt um ±0,2 m·g.
Das Target a_res schwankt mit dem Rauschen um ±2–3 m/s² (vermutlich Schubverzögerung, die
das Nominalmodell nicht kennt, #6; ungeprüft), das Modell folgt, der Rollout-Mittelwert
zeigt Spitzen bis −15 … −20 m/s². Das 5-s-Fenster bleibt gescheitert (Rollout −1,3 / −1,7).
Laufzeit: zwei Läufe parallel bremsen sich gegenseitig um >10× (JAX-Threadpools über alle
Kerne); Läufe nacheinander starten (~2 min für 5 Läufe).

## ToDo

### 9. Messrauschen / Zustandsschätzung
Die Plant liefert den exakten Zustand; eine echte Drohne liefert eine Schätzung
(IMU, Motion Capture) mit Rauschen und Verzögerung. Für „Plant 1:1 austauschbar“ sollte
`step()` einen verrauschten Zustand liefern und den wahren nur separat zur Auswertung.
Offen: wann einführen, Rauschmodell (Rauschen auf R muss auf SO(3) bleiben),
errors-in-variables beim Lernen. Vor dem Closed-Loop-Teil entscheiden.

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

### 18. Rendern: Referenz, Baseline, Policy/Modell nebeneinander
Für Bericht und Verständnis. Aus aufgezeichneten Läufen (Replay), nicht in der Schleife:
gleiches Format offline und online, Schleife bleibt unberührt; live später nur zweite
Datenquelle. Werkzeug (Crazyflow/MuJoCo mit mehreren Drohnen, matplotlib 3D) dann
entscheiden. Zeitpunkt: Anfang des Closed-Loop-Teils.
Stand 2026-11-23: matplotlib-3D als GIF (`scripts/render_runs.py`, liest die `.npz`, die
`run_adaptive.py` neben die Abbildung schreibt; Pillow, kein ffmpeg). Punkte + 1-s-Spur je
Lauf, Farben wie in den Abbildungen. Offen: MuJoCo-Replay mit Drohnenmodellen für Bericht
und Präsentation.

### 19. Modell-Gradienten gegen den wahren Gradienten messen
Mit `Plant.snapshot`/`true_step` (#11): Sensitivität ∂x_{k+H}/∂u_k für H = 1, 10, 30,
nominal (J1) und hybrid (J2) gegen wahr, auf Kreis und Hover, je Bedingung; relativer
Fehler getrennt für Position und Lage. Offen: ob das Nachwirken der Ratendynamik den
Einschrittfehler (Faktor 3–15) über H ausgleicht. J2 erst mit ReLU, bei rauen Ableitungen
glatte Aktivierung (softplus/tanh, Neutraining). Später ebenso für den Policy-Gradienten
selbst (FO und ZO gegen wahr). Spätestens mit dem ersten FO-Ergebnis einbauen.
