# Entscheidungen

Kurzes Log der Design-Entscheidungen. Pro Eintrag: was, warum, verworfene Alternativen.
Details (Messungen, Diskussion) stehen im Commit, der die Entscheidung umsetzt.
Status: **gilt** · **offen** · **ersetzt durch #n**

---

### 1. Steuerung über Body Rates + kollektiven Schub (gilt, Projektstart)
`Control.body_rate` mit `Dynamics.first_principles`. Nominales Modell wird dadurch
Punktmasse + Rotationskinematik (wie LotF), keine Euler-Singularitäten (Looping möglich),
Policy-Rate 100–200 Hz reicht.
Verworfen: `state` (Regler macht alles), `attitude` (Euler-Singularität bei ±90°),
`force_torque`/`rotor_vel` (≥ 500 Hz, Trägheit + Motoren im Surrogat nötig).

### 2. Raten: Policy 100 Hz, Physik und innerer Ratenregler 500 Hz (gilt, 2026-11-17)
`body_rate_freq` ist die Rate von Crazyflows innerem Regler, nicht die der Policy.
u wird über 5 Physikschritte gehalten. LotF nutzt 50 Hz; Crazyflow empfiehlt ≥ 200 Hz
für body_rate. 100 Hz als Kompromiss aus BPTT-Länge und Modellfehler.

### 3. Plant-Schnittstelle (gilt, 2026-11-17)
`reset() → x`, `step(u) → x⁺`; nur `plant.py` kennt Crazyflow. Tupel (x, u, x⁺) baut
die Schleife, nicht die Plant. x = [p, R (row-major), v, ω_body], 18-dim.
Crazyflows Lageterme im Ratenregler (`kR`, `ki_m`) auf 0 → reiner Ratenkanal.

### 4. Kein Hover-Init der Rotoren (gilt, 2026-11-17)
Rotoren starten bei 0 RPM, Drohne sinkt anfangs ~16 cm. Bewusst einfach gehalten.
Folge: erste ~2 s eines Laufs nicht als Trainingsdaten verwenden (Rotordrehzahl ist
versteckter Zustand, nicht in x).

### 5. Baseline-Regler geometrisch, ohne Integrator (gilt, 2026-11-17)
Position-PD + Vorsteuerung → Soll-Lage auf SO(3) → Raten. Zustandslos, damit
`u = controller(x, ref)` dieselbe Signatur wie eine Policy hat.
Preis: bleibender Fehler bei konstanten Störungen (Kreis: ~1,6 cm).

### 6. ω im nominalen Modell (offen)
Optionen: ω = ω_cmd (LotF, mit Totzeit-Puffer) · Verzögerung 1. Ordnung · zusätzlich
Schubverzögerung. Gemessen: τ ≈ 30/31/69 ms (roll/pitch/yaw), Schub ≈ 42 ms.
1. Ordnung halbiert den Lagefehler (k = 10: 1,55° → 0,74°), Positionsfehler fast gleich.
Relevant v. a. für Looping und Gradientenvergleich.

### 7. Anregung für Modell-Daten (offen)
LotF: kein Aktionsrauschen im Code, Vielfalt über zufällige Resets — für uns
(ohne Resets) nicht verfügbar. Kandidaten: gehaltenes Rauschen auf u, variierte Referenz.
