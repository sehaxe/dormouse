# Prior art: active aerodynamic stabilization of amateur/university solid rockets, wind effects, learning-based flight control, onboard NN controllers, LLM-in-the-loop

Research for «Поток-М» (Супер ГИРД / Воздушно-инженерная школа): 2-stage solid rocket, L=1870 mm, Ø102 mm, 12.4 kg, ~250 m/s, apogee ~28 s, ~4 km; 4+4 trapezoid fins; planned: active aerodynamic fins + NN controller (PID backup).

- Date of research: 2026-10-17. All sources read directly unless marked otherwise.
- Method: 16+ search rounds (DuckDuckGo HTML), direct fetches of project pages, university PDFs, arXiv abstracts, manufacturer manuals. Russian-language lane included (eLibrary/CyberLeninka, BMSTU journal).
- Web-search tool was HTTP 403; DDG HTML + direct fetch used instead. One page (roscansat.com SuperGIRD league) was unreachable — see Open questions.

---

## TL;DR

1. Active aerodynamic stabilization in amateur/university rockets is **demonstrated** but mostly in narrow regimes: BPS lands TVC rockets (no fins, thrust-vector, 40+ flights); Delft (DARE) flew roll-axis-only canard stabilization (2016) and reports simulated apogee gain; Canterbury (NZ) claims full attitude control; Leeds built a canard system but **never activated control in flight**; Santa Clara, WPI, Madison, Liberty are design/senior projects.
2. Wind is the standard, well-documented reason active control exists: NASA documents weathercocking; NAR caps launch wind at 20 mph; Apogee's heuristics (40° cone, "wind ≤ 20 % of exit speed") are the hobby standard; 4 km apogee + 5 m/s crosswind implies ~km-class drift (arithmetic below — not a sourced measurement).
3. Learning-based flight control is proven on drones (Neural-Fly: winds to 12.1 m/s, 2.5–4× smaller path error), simulated on rockets (RL landing 8 %→97 % success, HIL only). **No** public example of an NN running the inner control loop of a solid amateur rocket was found.
4. Real onboard practice: PID + IMU, not NN. Commercial HPR computers log at 500 Hz (Blue Raven); BPS Signal runs "a high-speed control loop" (rate unpublished) and tunes PID gains from a measured-inertia string test; IMU guidance recommends BMI088; Russian teams publish Arduino+BMP280/BMI270 designs.
5. No evidence of LLMs in any inner control loop. The closest published system is LLM-as-commander over MAVLink (DroneServer, 2026): LLM sends commands to a certified-style autopilot, it does not stabilize the vehicle.

---

## 1. Active aerodynamic stabilization — amateur / university

| Candidate | What was done | Numbers / results | Source (date) |
|---|---|---|---|
| **BPS.space (Joe Barnard) TVC** | Gimbal TVC mount + Signal flight computer; finless active stability; propulsive landing program | TVC mount: ±5°, max bandwidth 13 Hz / nominal 25 Hz, 5–20 N thrust, "over 40 successful flights" over 3 yr of iteration; software "not publicly available yet"; BPS itself says "your first few tests will probably fail". Active control "in model rocketry since the 1980s"; Signal = stability augmentation, **not guidance** | https://bps.space/products/thrust-vector-control (n.d., site ©2026); https://bps.space/pages/about (n.d.) |
| **BPS landing (2022)** | Model rocket soft landing using TVC + custom flight computer | "Successfully lands a model rocket" (media report; BPS pages do not state a landing metric) — secondary source | https://hackaday.com/2022/08/05/bps-space-succesfully-lands-a-model-rocket/ (2022-08-05) |
| **DARE TU Delft — Advanced Control Team (V7S)** | Canard-based Stability Augmentation System, 4 canards, separate servos, SDRE nonlinear controller, XSENS MTi-100 IMU; wind-tunnel test rig; flight tests 2016 | Simulated: **apogee +9.2 %, ground track −24.7 %** vs passive V7; wind tunnel: canards behaved at airspeeds up to 80 m/s; 2016 flights: **roll-axis stabilization only** (software-limited); found canard–fin aerodynamic interference; canards "not optimal" | https://dare.tudelft.nl/projects/act/ (2013–2016) |
| **Leeds University Rocketry (LURA) — Project Aptos / Pathfinder** | Canard active-stability module: aero design (XFLR5), 3 PCBs, smart servos with built-in PID position control, LQR controller (state x = {φ,θ,ψ,p,q,r}, inputs = 4 canard deflections), spin-can to reduce canard downwash; 2 flight tests 2023/24 with canards at 0°; planned deflection limit ±10° | Servo torque: required 7.95 kg·cm → chose 11 kg·cm (SF 1.4); apogee recorded 429.76 m vs OpenRocket 506.71 m (17.9 % error), time-of-apogee error 3 %; simulated 5 m/s crosswind: control reduces initial off-rail pitch by 2.7°; **control never switched on in flight** (servo damaged on recovery); LURA review: only Canterbury (below) "successfully implemented an active control system for three axis control" | https://leedsrocketry.co.uk/wp-content/uploads/2024/08/Development-of-a-Canard-Controlled-Active-Stability-System-for-a-Sounding-Rocket.pdf (2024); control-algorithm report: .../Canard-Control-Development.pdf (2024) |
| **UC Rocketry, University of Canterbury (NZ)** | Rocket series with control: Lisa (roll control), Tasha (full attitude control, incl. "stabilising an unstable rocket configuration"), Melissa (roll control on one launch, NZ I-class record attempt), Milly (I-class world altitude record) | Lisa series "subsonic … capable of roll control … around 600 m"; Tasha: full attitude control; records page (no loop rates/telemetry published) | https://www.ucrocketry.org/rockets (project archive, flights 2009–2014+) |
| **Santa Clara University (2018 thesis)** | Active control stabilization of a high-power rocket: 6DOF model-based design, built system | Thesis: instability "augmented when … thrust-to-weight ratio is less than 5:1"; design+build of active control to hold straight path to apogee; per Leeds review — modelling/control/electronics done, only prototype ("did not perform mechanical design") | https://scholarcommons.scu.edu/mech_senior/81/ (2018-06) |
| **Worcester Polytechnic (2019)** | Tab actuation on passive rear fins; bench-tested | Bench success; **rocket never left the pad**, control not demonstrated (per Leeds review) | cited in Leeds PDF above; primary WPI report not located |
| **UW–Madison (Badger Blast Off, 2023/24)** | Articulating canard fin system + PID + air-brake function for IREC 10 000 ft target; 4-inch test rocket | Design phase as of Oct 2023; states active systems exist at Santa Clara and VOS Visions; goal: maintain vertical orientation + airbrake | https://seniordesign.me.wisc.edu/2023/10/24/badger-blast-off/ (2023-10-24) |
| **Liberty University (2026 honors thesis)** | Design optimization of active-stabilizing canards (iterative trajectory simulation + adaptive surrogate modelling) | Thesis accepted Spring 2026; optimization study, not a flight | https://digitalcommons.liberty.edu/honors/1601/ (2026-05-08) |
| **Bauman MSTU — flight test bench LS3 (Russia)** | Model rocket with automatic control system + **grid fins** (решетчатые стабилизаторы), 4 servo blocks, belt transmission; composite body; designed to work on both boost and coast phases; prototype manufactured | Altitude ≤400 m; liftoff mass ≤1.5 kg; length ≤1000 mm; Ø≤100 mm; engine RD1-50/RD1-100/RD1-300; positioning accuracy **0.2–0.3°** (belt ratio 3:1); descent 4 m/s, landing radius ≤700 m; strength SF 3.17 after re-design; **flight results not published in the paper** | https://engjournal.bmstu.ru/articles/2031/eng/2031.pdf; DOI 10.18698/2308-6033-2020-11-2031 (2020) |
| **Reshetnev SibSU (Krasnoyarsk) 2026** | Active-rudder control unit for rocket models: frame with 4 servos, electronics bay, cover w/ battery; Arduino Nano + BMI270 IMU (6 axes) + 4×MG90S + GPS NEO-7M + BMP280 + nRF24L01; rudders turn 90° on descent to act as a wing | Concept/design paper; no flight numbers | https://cyberleninka.ru/article/n/razrabotka-sistemy-upravleniya-aktivnyh-ruley-v-modelyah-raket (2026) |
| **Reshetnev SibSAU 2017 — onboard measuring complex** | Arduino Nano V3.0 + Bosch BMP280 pressure/temp; firing circuit (IRFZ44N) for recovery; piezo finder; logging of altitude/velocity/time | **10 test flights, 100 % success**; mass 25 g; 25×63×22 mm; −5…+40 °C; 20 min battery; cites commercial Russian controller "ComFly-030" (Real Rockets) and "Altimeter Two" (Jolly Logic) as prior art | https://cyberleninka.ru/article/n/razrabotka-bortovogo-izmeritelnogo-kompleksa-modeli-rakety (2017) |
| **Guided Rocket Avionics (student GitHub, 2026)** | 6DOF sim + EKF + PID for TVC-guided model rocket; STM32F405 firmware + custom PCB planned | Repo README: most components "Not started" (as of research date); 0 stars — listed as example of hobby attempts, low evidence value | https://github.com/Guided-Rocket-Avionics/Simulation (2026) |

**Verdict for this block (VERIFIED):** active fin control has been built and flown by university teams but nearly always in a reduced mode (roll only) or never activated; the only "full" claims are Canterbury's project page (self-reported) and the Leeds literature assessment that Canterbury is the single successful 3-axis case. TVC (BPS) is the best-documented amateur active-stability technology, with the flight software closed-source.

---

## 2. Wind effects and stability practice

| Practice | Numbers | Source (date) |
|---|---|---|
| Stability condition | NASA: stable iff **center of pressure below center of gravity**; swing test (nose leads in rotation); fix by nose weight / bigger fins. NASA's own note: full-scale rockets use TVC, not fins | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/conditions-for-rocket-stability/ (updated 2023-11-20) |
| Stability margin (caliber) | "stability margin should be greater than one [body diameter]" (cited to Guerrero 2018 / NAROM 2018 in the Leeds report); Apogee: making a rocket "more" stable increases weathercocking and lowers flights | Leeds PDF above (2024); https://www.apogeerockets.com/Peak-of-Flight/Newsletter594 (Apogee newsletter) |
| Weathercocking mechanism | Rocket turns into the wind because relative airflow is tilted; tilt angle depends on **wind-vs-rocket-velocity ratio**; weathercocking reduces max altitude. (Page prints formula `tan b = V/w` — see Open questions) | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/rocket-weather-cocking/ (updated 2023-11-20) |
| Launch wind limit (rule) | NAR Model Rocket Safety Code §9: "wind speeds no greater than 20 miles per hour" (32 km/h); model rockets ≤1 500 g / 125 g propellant; launch site 1 000–1 500 ft table | https://www.nar.org/ModelRocketSafetyCode (accessed 2026-10-17) |
| Hobby heuristics on wind vs flight | "Wind speed should be less than 20 % of the rocket's speed" (rocket ≥5× wind speed); a 15 mph wind ⇒ rail-exit speed ~75 mph (110 ft/s ≈ 33.5 m/s) to be "truly safe"; **weathercocking cone: imaginary 40° cone from the pad**, apogee point inside = acceptable | https://www.apogeerockets.com/education/downloads/Newsletter671.pdf (Peak of Flight #671); same cone in …/Newsletter625.pdf (simulations run at 8 mph; 20 % rule restated; 23 m/s exit used) |
| Weathercocking judging methods | Apogee Tech Pub 21 lists 5 methods to judge weathercocking; one uses a **20° cone** at the pad; also "wind causes instability" reference | https://www.apogeerockets.com/downloads/Technical_Publications/Tech_Pub_21.pdf |
| Multi-stage specific risk | Two-stage: weathercocking can mean upper stage igniting near-horizontal; drift minimized by design, not control | https://www.apogeerockets.com/Peak-of-Flight/Newsletter301 (Peak of Flight #301) |
| Simulation practice | RocketPy (open-source 6DOF): Monte Carlo dispersion notebook (=EuRoC example) prints **impact X std = 46.8 m** (includes motor/mass/wind uncertainties); out-of-rail velocity 23.08 ± 3.12 m/s | https://docs.rocketpy.org/en/v0.13.1/notebooks/dispersion_analysis/dispersion_analysis.html (v0.13.1 docs) |
| Active control gain (simulated) | DARE: −24.7 % ground track, +9.2 % apogee (simulation) — see §1 | dare.tudelft.nl (2016) |
| Crosswind drift (5–10 m/s) | **No direct measured number found for a 4 km amateur flight.** First-order: drift ≈ wind × descent time. For 4 km apogee, ~7 m/s under canopy ⇒ ~570 s ⇒ 5 m/s wind ≈ 2.9 km; 10 m/s ≈ 5.7 km. DERIVED ARITHMETIC, not a source measurement (see SPECULATION) | formula form used by public recovery calculators (e.g. https://www.altimetercloud.com/tools/drift_calculator/) |

---

## 3. Learning-based flight control (peer-reviewed / primary)

| Work | Method | Numbers | Source (date) |
|---|---|---|---|
| **Neural-Fly** (Caltech) | Deep learned shared representation of aerodynamics + composite adaptation (online linear coefficient update); trained on 12 min of flight data | Winds up to **12.1 m/s (43.6 km/h, Beaufort 6)** in the Caltech Real Weather Wind Tunnel; **path-tracking error 2.5×–4× smaller** than state-of-the-art nonlinear/adaptive controllers; exponential-stability guarantees; works outdoors with onboard sensors; transfers across drones | arXiv:2205.06908 / Sci. Robotics 7(66) eabm6597 (2022-05-04); press: https://www.caltech.edu/about/news/rapid-adaptation-of-deep-learning-teaches-drones-to-survive-any-weather |
| **Neuroflight** | First open-source NN-based flight-controller firmware; NN trained in sim, compiled to embedded | NN runs at **>2.67 kHz on Arm Cortex-M7**; flight tests: stable flight + aerobatic maneuvers | arXiv:1901.06553 (2019) |
| **Proto-MPC** | Meta-learned encoder-prototype-decoder inside an MPC for winds | Robust trajectory tracking with static and spatially varying side winds — **simulation only** | arXiv:2401.15508 (2024) |
| **RAJS (RL rocket landing)** | RL with guide policy (classical controller as jump-start), high-fidelity rocket model | Landing success **8 % (baseline) → 97 %** (RL); validated incl. **hardware-in-the-loop** — not a flight test | arXiv:2407.15083, IROS 2024 oral (2024-07-21) |
| **SLS Adaptive Augmenting Control (NASA)** | Adaptive law augmenting (not replacing) gain-scheduled PID + bending filters; augments attitude-control response outside design envelope | Flew on **Artemis I, 2022-11-16**; performed as intended through launch phase; heritage Ares I-X; concept flight-tested on F/A-18 research aircraft 2013; motivation: 41 % of historical RLV failures in non-GN&C subsystems might have been mitigated by advanced GN&C | ESA-GNC paper: https://www.esa-gnc.eu/papers-open/sopot2023/222.pdf (2023); https://ntrs.nasa.gov/citations/20140007340 (2014); https://www.nasa.gov/wp-content/uploads/2015/05/sls.pdf (2015) |
| **Other** | arXiv has a dense literature on NN/RL quadrotor control (e.g. vision-only attitude, 2025; RAPTOR foundation policy, 2025) — all multicopter, none solid-rocket inner loop found | — | arXiv searches, 2026-10-17 |

---

## 4. Real flight controllers / onboard NN — what is standard

| System | Onboard compute/sensing | Loop/log rates | Source (date) |
|---|---|---|---|
| **BPS Signal R2** | Active-stability flight computer for TVC rockets; app-configured; PID gains computed in the app from measured inertia (string-pendulum test: 10 rotations) + mass + COM data; abort logic | "runs a high speed control loop" — **rate not published** in manual/tuning docs; tuning doc v1.3 (2018-07-12), manual v1.8 (2019-01-01) | Manual & tuning PDFs: cdn.shopify.com/…/Signal_R2_-_User_Manual_1.8.pdf; …/Signal_R2_-_Tuning_Instructions.pdf |
| **BPS IMU guidance** | Recommends **BMI088** ("best choice", ±24 g, ±2000°/s); avoid MPU series; do not use breakouts; BNO055 fusion "fails under flight loads" | Vendor-agnostic rules, no rates | https://bps-space.github.io/electronics/imu-selection.html (BPS wiki, CC BY-SA) |
| **Featherweight Blue Raven** | Commercial HPR flight computer: 2× 3-axis accel (±400 g, ±32 g), 3-axis gyro, baro; 3D inertial navigation; 4×25 A outputs; Bluetooth | **500 Hz** accel/gyro/quaternion; **50 Hz** velocities/baro/events; 22 flight events; $200 | https://www.madcowrocketry.com/featherweight-blue-raven/ (reseller listing, features status 2023-08-02); manufacturer: featherweightaltimeters.com (page blocked to automated fetch) |
| **Neuroflight** | NN firmware on Cortex-M7 (Betaflight-class FC) | >2.67 kHz NN execution; real flight tests | arXiv:1901.06553 (2019) |
| **Russian practice** | "ComFly-030" (Real Rockets) flight controller (commercial); student builds: Arduino Nano + BMP280 (2017, 10/10 flights), Arduino Nano + BMI270 + MG90S (2026 concept) | no rates published | CyberLeninka 2017 & 2026 articles (see §1) |

**Pattern (VERIFIED):** the flying, shipped practice is PID + MEMS IMU at tens–hundreds of Hz; NNs fly only in research firmware (Neuroflight) or drone racing/acrobatics stacks; no commercial amateur-rocket flight computer ships an NN control loop.

---

## 5. "LLM / machine code in the control loop"

| Item | Finding | Source (date) |
|---|---|---|
| LLM-as-commander over autopilot | DroneServer: MCP↔MAVLink bridge, 98 tools, LLM treated as **untrusted commander**; server-side validation, geofence, audit; 1 000+ simulated flights; 8/11 models complete 90–100 % of 6 flying missions; on real hardware, models flew an unseen mission 10/10 (3 quadcopters, GPS and GPS-denied) | arXiv:2601.15486 (v3 2026-09-11) |
| LLM inside the inner attitude/rate loop on a rocket | **No evidence found.** None of the surveyed rocket/amateur flight controllers, university projects or commercial units uses LLM output as a control signal; the nearest published boundary is the DroneServer paper, which explicitly keeps the LLM outside the loop and validates every command server-side | search rounds 2026-10-17 (no primary source exists in what was found) |

---

## Safe to claim / Do not claim

### Safe to claim (each with its source)
1. Active aerodynamic (fin/canard) stabilization of small rockets is established prior art: DARE flew roll-axis canard stabilization (2016) and simulated +9.2 % apogee / −24.7 % ground track; Canterbury reports full attitude control incl. stabilizing an unstable configuration; BPS flies TVC-only active stability with 40+ flights and sells the hardware (software closed) — §1 sources.
2. It is normal for these systems to be **stability augmentation, not guidance** — BPS states this distinction explicitly (https://bps.space/pages/about).
3. Wind/weathercocking is the canonical disturbance: NASA documents the mechanism and apogee loss; NAR caps launch wind at 20 mph; Apogee's practice numbers (40° cone; wind ≤20 % of exit speed) are citable hobby standards — §2 sources.
4. Learning-based wind rejection is proven **on drones**: Neural-Fly, 12.1 m/s winds, 2.5–4× tracking-error reduction (Science Robotics 2022).
5. Adaptive control on real launch vehicles exists: NASA SLS AAC flew on Artemis I (2022-11-16), built to *augment*, not replace, gain-scheduled control (ESA-GNC/NTRS).
6. RL rocket-landing exists in high-fidelity simulation + HIL: 8 %→97 % success (RAJS, IROS 2024) — not flown.
7. NN inner-loop flight control exists in research firmware: Neuroflight at >2.67 kHz on Cortex-M7, flight-tested 2019.
8. Commercial/onboard logging practice is ~500 Hz IMU on HPR (Blue Raven) with PID-era avionics; recommended IMU class BMI088 (BPS wiki).
9. Neural networks are **not** standard in amateur rocket control loops; no LLM inner-loop flight control evidence was found (absence-of-evidence claim).

### Do not claim
1. **"The NN flies the rocket's inner loop" as an existing amateur practice** — no such public flight case was found; closest is drone research firmware (Neuroflight) and drone-tunnel experiments (Neural-Fly).
2. **"BPS/others publish control-loop rates for amateur rockets"** — BPS explicitly does not publish Signal's loop rate; loop rates are known only for research FCs (2.67 kHz, Neuroflight) and log rates for commercial computers (500 Hz, Blue Raven).
3. **"Active canard control is a solved amateur technique"** — Leeds' system was never armed in flight; Santa Clara/WPI were prototypes; DARE flew roll-only after finding canard–fin interference; BPS tells buyers "your first few tests will probably fail".
4. **"Neural-Fly = rocket flight control"** — it is a quadrotor result; do not transfer the 12.1 m/s / 2.5–4× numbers to a rocket without own testing.
5. **"RL already lands rockets"** — RAJS is sim+HIL (industry rocket models), not a flight demonstration; G-FOLD/Falcon 9 are not RL.
6. **Precise crosswind-drift numbers for a 4 km apogee at 5–10 m/s** — no measured source found; only arithmetic estimates (see SPECULATION).
7. **"LLMs control drones today"** — the only strong paper keeps the LLM outside the loop as an untrusted commander (DroneServer).
8. **VOS Visions "VOS2"** — referenced only by UW–Madison's page (secondary); no primary technical paper was found; do not cite as prior art without locating the original.
9. **"Canterbury = 3-axis flight-proven"** — the claim is the team's own project page + Leeds' literature review; no telemetry/loop data was found to substantiate it.

---

## VERIFIED (sourced)
- Every row of §1–§5 tables with its URL/date. Key anchors: BPS product page (40+ flights, ±5°, 13/25 Hz), DARE (9.2 %/−24.7 %, 80 m/s tunnel, roll-only 2016), Leeds PDF (429.76 vs 506.71 m; 2.7°; ±10°; control not flown), Bauman LS3 (0.2–0.3°, ≤400 m), NAR code (20 mph), NASA weathercocking/stability pages, Apogee #671/#625/#21 (40° cone, 20 % rule), Neural-Fly (12.1 m/s; 2.5–4×), Neuroflight (2.67 kHz), RAJS (8→97 %), SLS AAC (Artemis I 2022-11-16), Blue Raven (500/50 Hz), DroneServer (LLM outside loop).

## SPECULATION / own inference (not sourced)
1. **Crosswind drift magnitude for your rocket.** No measured 5–10 m/s crosswind drift number was found for a 4 km amateur flight. DERIVED (my arithmetic): drift ≈ wind × time under recovery. At ~7 m/s canopy descent from 4 km (≈570 s), 5 m/s wind ⇒ ~2.8 km; 10 m/s ⇒ ~5.7 km. This is an order-of-magnitude estimate; soil reality (dual deploy, descent rate, wind profile) will differ ±50 % or more.
2. **"NN controller is the hard part."** Inference from BPS's own statement that "nearly all of the complexity involved is in the flight software", plus the pattern that every NN flight controller found was drone/quadrotor, not fin-actuated rocket.
3. **Feasibility of NN inner loop on a 4 km solid rocket**: SPECULATION — no direct benchmark exists; based on Neuroflight (2.67 kHz on MCU) and the low dynamics of a 12.4 kg / 102 mm rocket, a small MLP at 100–500 Hz is plausible on Cortex-M4/M7-class hardware, but no public example validates stability under rocket vibration/transonic conditions.
4. **Interpretation of NASA's printed formula** on the weathercocking page (`tan b = V/w`): the physical reading is an inclination depending on the wind/velocity ratio; the fraction as printed may be inverted or figure-specific. Flagged as unverified.

## Open questions
- BPS Signal actual loop rate (unpublished in the manual/tuning docs).
- Rover-ready evidence for UW–Madison/Liberty results (current phases are design/optimization).
- VOS Visions VOS2 primary document (not found).
- roscansat.com SuperGIRD league baseline (page unreachable during research; a search snippet claims the league flyes a mass-size mockup to ≥1 000 m — UNVERIFIED).
- Any amateur/university rocket with an actual NN in the flight control loop — none found (absence of evidence, not proof of absence).

## Source list (read directly)
1. https://bps.space/products/thrust-vector-control
2. https://bps.space/pages/about
3. https://bps.space/pages/build-signal-r2 (+ manual & tuning PDFs, cdn.shopify.com)
4. https://bps-space.github.io/electronics/imu-selection.html
5. https://hackaday.com/2022/08/05/bps-space-succesfully-lands-a-model-rocket/ (secondary)
6. https://dare.tudelft.nl/projects/act/
7. https://leedsrocketry.co.uk/wp-content/uploads/2024/08/Development-of-a-Canard-Controlled-Active-Stability-System-for-a-Sounding-Rocket.pdf
8. https://leedsrocketry.co.uk/wp-content/uploads/2024/08/Canard-Control-Development.pdf
9. https://www.ucrocketry.org/rockets
10. https://scholarcommons.scu.edu/mech_senior/81/
11. https://seniordesign.me.wisc.edu/2023/10/24/badger-blast-off/
12. https://digitalcommons.liberty.edu/honors/1601/
13. https://engjournal.bmstu.ru/articles/2031/eng/2031.pdf
14. https://cyberleninka.ru/article/n/razrabotka-sistemy-upravleniya-aktivnyh-ruley-v-modelyah-raket
15. https://cyberleninka.ru/article/n/razrabotka-bortovogo-izmeritelnogo-kompleksa-modeli-rakety
16. https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/conditions-for-rocket-stability/
17. https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/rocket-weather-cocking/
18. https://www.nar.org/ModelRocketSafetyCode
19. https://www.apogeerockets.com/education/downloads/Newsletter671.pdf ; …/Newsletter625.pdf ; …/Newsletter301 ; https://www.apogeerockets.com/downloads/Technical_Publications/Tech_Pub_21.pdf ; https://www.apogeerockets.com/Peak-of-Flight/Newsletter594
20. https://docs.rocketpy.org/en/v0.13.1/notebooks/dispersion_analysis/dispersion_analysis.html
21. arXiv:2205.06908 (+ https://www.caltech.edu/about/news/rapid-adaptation-of-deep-learning-teaches-drones-to-survive-any-weather)
22. arXiv:1901.06553
23. arXiv:2401.15508
24. arXiv:2407.15083
25. https://www.esa-gnc.eu/papers-open/sopot2023/222.pdf ; https://ntrs.nasa.gov/citations/20140007340 ; https://www.nasa.gov/wp-content/uploads/2015/05/sls.pdf
26. https://www.madcowrocketry.com/featherweight-blue-raven/ (reseller; manufacturer page blocked)
27. arXiv:2601.15486
