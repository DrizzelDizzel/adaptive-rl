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

## ToDo

### 6. ω im nominalen Modell
Optionen: ω = ω_cmd (LotF, mit Totzeit-Puffer) · Verzögerung 1. Ordnung · zusätzlich
Schubverzögerung. Gemessen: τ ≈ 30/31/69 ms (roll/pitch/yaw), Schub ≈ 42 ms.
1. Ordnung halbiert den Lagefehler (k = 10: 1,55° → 0,74°), Positionsfehler fast gleich.
Relevant v. a. für Looping und Gradientenvergleich. Muss vor `nominal.py` entschieden sein.

### 9. Messrauschen / Zustandsschätzung
Die Plant liefert den exakten Zustand; eine echte Drohne liefert eine Schätzung
(IMU, Motion Capture) mit Rauschen und Verzögerung. Für „Plant 1:1 austauschbar“ sollte
`step()` einen verrauschten Zustand liefern und den wahren nur separat zur Auswertung.
Offen: wann einführen, Rauschmodell (Rauschen auf R muss auf SO(3) bleiben),
errors-in-variables beim Lernen. Vor dem Closed-Loop-Teil entscheiden.
