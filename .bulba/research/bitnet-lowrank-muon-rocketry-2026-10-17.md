# Ternary weights / low-rank compression / Muon — evidence check for a ВИШ rocket-AI document

Date: 2026-10-17. Purpose: verify with PRIMARY sources (papers, repo READMEs, vendor docs — not summaries) three technology choices claimed for an on-board NN stabilization system of a two-stage solid rocket (~4 km): **(A) BitNet-style 1.58-bit ternary quantization, (B) low-rank/spectral compression, (C) Muon optimizer.**

Method: 8 search rounds (English + Russian), every number below read from the source itself on 2026-10-17 (HTML/abstract/repo README/docs fetched directly). Dates are the source's own. Unverifiable items are marked and excluded from the "safe" list.

---

## 0. TL;DR verdict

- **(A) BitNet family — well-evidenced, but the evidence is LLM/small-LM scale.** Ternary weights (‒1,0,+1) at 1.58 bits/weight match fp16 LLMs at ≥2B (measured), save 3.55x GPU memory / 2.71x latency at 3B vs the paper's LLaMA baseline, and CPU inference engines show 1.37–6.17x speedups / 55.4–82.2% energy cuts on x86/ARM **CPUs**. The only **small-model** evidence (100K–48M params — the scale of a control net) is two 2024 papers: BitNet b1.58 "Reloaded" (state-of-the-art small LMs when hidden sizes are doubled; matches/surpasses small vision models) and "When are 1.58 bits enough?" (MLPs/GNNs/encoders on par or better than 16/32-bit).
- **(B) Low-rank — proven at LLM scale only.** ASVD: training-free, rank −50–75% (= 10–20% net compression) at ~1% MMLU loss; 50% KV-cache cut. SVD-LLM: better than ASVD/SVD at 20–40% compression. SVD itself has a published bare-metal Cortex-M4F implementation (Kalman filtering, Electronics 2021) — but **no public measurement of SVD-LLM/ASVD-class low-rank compression of a tiny control net on an MCU exists**. Claim narrowly.
- **(C) Muon — strong training-time evidence; irrelevant to on-board inference.** 1.35x nanoGPT-speedrun record (124M); 1.5B to GPT-2-XL HellaSwag in 10 vs 13.3 8×H100-hours; ~2x compute efficiency vs AdamW in LLM scaling laws; used for Kimi K2 (1T total, 15.5T tokens, zero loss spikes). Available as `torch.optim.Muon` in current PyTorch docs. **No evidence below ~124M-parameter scale; nothing about MCU.** Do not claim it speeds up the rocket's on-board network.

---

## 1. BitNet family

### 1.1 BitNet b1.58 (arXiv:2402.17764, v1 27 Feb 2024 — Microsoft Research)

Verified content (paper HTML):
- Every parameter ternary {−1, 0, 1}; "matches the full-precision (FP16/BF16) Transformer LLM with the same model size and training tokens" in perplexity and end tasks.
- vs **their** LLaMA LLM baseline, 3B: **2.71× lower latency, 3.55× less GPU memory** ("being 2.71 times faster and using 3.55 times less GPU memory"). 3.9B: 2.4× / 3.32×, and "performs significantly better than LLaMA LLM 3B".
- 70B: **max batch size 16 → 176 (11.0×), throughput 333 → 2977 tokens/s (8.9×)** (Table 3).
- Energy: "BitNet b1.58 saves **71.4 times** arithmetic operations energy consumption for matrix multiplication on **7nm chips**" — this is an energy-model estimate, not a measurement (paper cites its energy model).
- Feature-filtering: zeros give free sparsity.

### 1.2 BitNet b1.58 2B4T technical report (arXiv:2504.12285, v2 25 Apr 2025)

What was measured / stated:
- First open-source, **natively 1-bit** 2B LLM; trained on **4T tokens**; weights on HF + GPU/CPU (bitnet.cpp) inference released.
- Table 1 (bitnet b1.58 2B4T column vs five full-precision 1–2B peers — LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, MiniCPM 2B): **memory 0.4 GB (non-emb)** vs 1.4–4.8 GB; **latency 29 ms (CPU, TPOT)** vs 41–124 ms; **energy 0.028 J (estimated)** vs 0.186–0.649 J.
- Benchmark average: **54.19** (BitNet) vs 44.90 / 43.74 / 55.23 / 48.70 / 42.05 (peers, same order) — i.e. on par with the best peer (Qwen2.5-1.5B), above the rest. MT-bench 5.85.
- Claim in the abstract: "performance on par with leading open-weight, full-precision LLMs of similar size" while reducing memory/energy/decoding latency.

### 1.3 BitNet a4.8 (arXiv:2411.04965, 7 Nov 2024) — activation quantization

- **4-bit activations** for inputs to attention and FFN; intermediate states **sparsified + 8-bit** quantized (hybrid scheme vs outlier channels).
- "Comparable to BitNet b1.58 with equivalent training costs", + faster inference via INT4/FP4 kernels.
- **Activates only 55% of parameters; supports 3-bit KV cache** (measured/строено in their tables; sparse 4-bit KV/QKV variants show negligible accuracy loss).

### 1.4 The small-scale gap is covered by two specific papers

- **"BitNet b1.58 Reloaded" (arXiv:2407.09527, 16 Jul 2024):** 1.58-bit QAT for **100K–48M-parameter** language and vision models; "state-of-the-art performance for small language models when doubling hidden layer sizes", "reaches or even surpasses state-of-the-art performance for small vision models of identical size"; concludes 1.58-bit QAT is "a viable and promising approach also for training smaller deep learning networks". → closest published scale to a rocket control net.
- **"When are 1.58 bits enough?" (arXiv:2411.05882, Nov 2024):** ternary training for **MLPs** and GNNs and encoder/encoder-decoder transformers: "on par with or sometimes even better than the standard 32/16-bit models" (regularization effect reported in T5-like encoders). Caveat from the same paper's probes: tiny networks (X-OR) needed more parameters / higher LR.

### 1.5 Comparison table

| Candidate | Source / date | Key facts & numbers | Credibility | Effort to adopt/verify |
|---|---|---|---|---|
| BitNet b1.58 | arXiv:2402.17764, 2024-02-27 | ternary {−1,0,+1}; 3B: 2.71× faster / 3.55× less GPU mem vs their LLaMA; 70B: 8.9× throughput; 71.4× matmul-energy claim @7nm (model-based) | Peer-facing MSR paper; numbers are their benchmark | High to reproduce; cite as-is |
| BitNet b1.58 2B4T | arXiv:2504.12285, 2025-04 | 4T-token native 1-bit 2B model; 0.4 GB non-emb / 29 ms TPOT / 0.028 J est. vs 1.4–4.8 GB / 41–124 ms / 0.19–0.65 J; avg benchmark 54.19 (best peer 55.23) | Official tech report + released weights | Open weights + bitnet.cpp; easy to quote |
| BitNet a4.8 | arXiv:2411.04965, 2024-11-07 | 4-bit activations, sparsify+8-bit intermediate, 55% params activated, 3-bit KV cache | MSR paper | Quote; not needed for MCU story |
| Reloaded (small models) | arXiv:2407.09527, 2024-07-16 | 100K–48M params: SOTA small LMs (doubling width), matches/passes small vision | Independent (Univ. of Southern Denmark) | THE citation for "works at our scale" |
| When are 1.58 bits enough | arXiv:2411.05882, 2024-11 | MLPs/GNNs/encoders ternary ≥ 16/32-bit quality; tiny nets need more params | Independent | Supports MLP-control claim |

---

## 2. CPU/edge inference engines for ternary

### 2.1 bitnet.cpp (github.com/microsoft/BitNet + arXiv:2410.16144, Oct 2024)

- README (accessed 2026-10-17): "speedups of **1.37x to 5.07x** on ARM CPUs … reduces energy consumption by **55.4% to 70.0%**"; "up to **6.17x** speedup on x86 CPUs"; energy "up to **82.2%** on x86".
- Tech report arXiv:2410.16144: x86 2.37–6.17x, ARM 1.37–5.07x; energy −71.9…−82.2% x86, −55.4…−70.0% ARM; hardware = **Apple M2 Ultra** (Mac Studio, 64 GB) and **Intel Core i7-13700H** (Surface Laptop Studio 2); "can run a **100B** BitNet b1.58 model on a single CPU … **5-7 tokens/s** (human reading speed)".
- System paper arXiv:2502.11880 (Feb 2025): sub-**2-bit-per-weight**, "up to **6.25x** speed over full-precision baselines and up to **2.32x** over low-bit baselines"; lossless I2_S/TL kernels.
- README also lists BitNet embedding models (0.6B): 1.42–2.28x prefill speedup vs F16 (8 threads).

### 2.2 T-MAC (arXiv:2407.00088; EuroSys 2025; repo microsoft/T-MAC)

- Lookup-table (LUT) mpGEMM — no dequantization: "up to **4x** increase in throughput and **70%** reduction in energy consumption compared to llama.cpp".
- Kernel speedup up to **6.6×**, average **3.6×** vs SOTA llama.cpp; e2e **2.8×** for Llama-2-7B-2bit.
- BitNet-b1.58-3B: **30 tokens/s (1 core) / 71 tokens/s (8 cores) on M2-Ultra; 11 tokens/s on Raspberry Pi 5**.
- Repo README (accessed 2026-10-17): 3B BitNet on **Surface Laptop 7 (Snapdragon X Elite): 20 tok/s 1 core, 48 tok/s 4 cores, "4~5x speedup" vs llama.cpp**; "even faster than the NPU in token generation on Snapdragon X Elite"; energy 60–70% lower.

**Hardware scoping is the key limit: these are desktop/laptop-class CPUs — no MCU (ESP32/STM32/RP) is tested in either paper.**

---

## 3. Real small-NN / transformer deployments on MCUs (with numbers)

### 3.1 ESP32-S3 (512 KB SRAM, dual Xtensa LX7 @240 MHz — Espressif product page)

| Demo | Model | Measured | Source (accessed 2026-10-17) |
|---|---|---|---|
| DaveBben/esp32-llm | stories260K (260K params, ~1 MB RAM) | **19.13 tok/s** (dual-core + ESP-DSP SIMD; 240 MHz CPU, 80 MHz PSRAM; ESP32-S3FH4R2, 2 MB PSRAM) | repo README (2024) |
| doryiii/esp32-llm | stories260K f32 / 3.3M INT8 | **~30 tok/s** (f32) / **~12 tok/s** (INT8; "memory bandwidth limit 13.1 tok/s"); needs 2 MB / 8 MB PSRAM | repo README |
| Circuit-Digest/ESP32-Tiny-LLM | stories260K, fp32, 64-token seq, 1.05 MB bin | **22.0 tok/s** S3 DevKit (8 MB OPI PSRAM) · 21.0 XIAO S3 · 11.0 ESP32-CAM (QSPI) · **2.32 tok/s** standard ESP32, no PSRAM (flash streaming) | repo README (repo created 2026-09-23) |
| slvDev/esp32-ai | **28.9M** params (25M as flash lookup table; 4-bit), 512 KB SRAM + 8 MB PSRAM + 16 MB flash | **9.88 tok/s** end-to-end; **94.9 ms/token** compute; ~450 B flash reads/token | repo README (2026) |
| imFARSI/NanoMind-S3 | stories15M (15.2M dense LLaMA2; 6 layers d=288) | INT8 14.74 MB → ~1.49 tok/s; **INT4 7.49 MB → ~2.54–2.96 tok/s (~2×)**; 3.54 MB KV-cache | repo README |
| vitorengers/esp32-trm-bitnet | TRM-Att transformer, **BitNet 1.58 ternary weights + INT8 activations**, 2 layers h=512 | **~4–6 minutes per Sudoku puzzle** (81 positions, 16 steps) — existence proof of ternary transform on S3, NOT real-time | repo README (repo created 2026-03-02) |

- **ESP-NN (Espressif, official):** person-detection VWW (INT8) on ESP32-S3: **54 ms** with ESP-NN vs 2300 ms without; MobileNetV3-Small 729 ms; YOLO11n (192×192) 1253 ms (single core) [espressif/esp-nn README].
- ESP-DL docs: ESP32-S3 "up to **7.5 giga MAC/s** at 240 MHz" (quantized models necessary).
- **RP2040/RP2350: no public transformer demo with measured tok/s found.** Closest public facts: RP2350 = 520 KB SRAM, Cortex-M33 @150 MHz (datasheet/Wikipedia) — i.e. a coherent small-control-net target, but the "transformer on RP" demo does not exist publicly as far as this search reached. Treat as empty cell, not as evidence.

### 3.2 STM32 / CMSIS-NN / TFLite Micro

- **CMSIS-NN paper (arXiv:1801.06601, 23 Jan 2018):** kernels give **4.6× runtime/throughput and 4.9× energy efficiency** vs baseline on Cortex-M; CIFAR-10 CNN on Cortex-M7 @216 MHz: **~99.1 ms/image, 79.9% (8-bit), ~133 KB footprint, ~249 MOps**.
- **Arm blog (2021-10-27):** TFLite-Micro + CMSIS-NN image recognition demo on Cortex-M7 board: **3.1× speedup** vs reference kernels (GCC), 2.9× (armclang).
- **ST blog (2022-11-09):** MLPerf Tiny on STM32; X-CUBE-AI 7.3 generated code **82% faster** inference (image classification) than TFLM on STM32H7A3.
- No public "transformer on STM32 with tok/s" demo found; nearest are CNN benchmarks above + NPU-class parts.

### 3.3 MLPerf Tiny (context benchmark)

- Suite (arXiv:2106.07597, 2021): 4 tasks — KWS (DS-CNN, 52.5 KB), VWW (MobileNetV1, 325 KB), image classification (ResNet, 96 KB), anomaly detection (FC-AE, 270 KB); measures accuracy + latency + energy.
- **v1.4 results (mlcommons.org, 2026-07):** some submissions "below **0.30 ms** latencies" for KWS/VWW/IC; **ST STM32U3 NPU: −76.0% inference time and −23.3% power** vs the same Cortex-M33 without NPU; **STM32H7P preview (Neural-ART NPU): up to −96.0% inference time** vs CM7-only; Syntiant NDP120 runs streaming wake-word at a **3.3% duty cycle**.

### 3.4 Comparison table

| Candidate | Source / date | Key facts & numbers | Credibility | Effort to adopt/verify |
|---|---|---|---|---|
| llama2.c on ESP32-S3 | DaveBben README 2024; doryiii; Circuit-Digest 2026 | 19.13–30 tok/s (260K), 12 tok/s (3.3M INT8); 2.32 tok/s no-PSRAM | Popular repos; per-run numbers, no independent replication | Cheap: board + ESP-IDF; re-measure themselves |
| esp32-ai (28.9M) | slvDev README 2026 | 9.88 tok/s; 94.9 ms/token; 25M params in flash | Single-author repo, detailed, reproducible | Medium (model download 14.9 MB, board ~$8) |
| NanoMind-S3 | imFARSI README 2026 | INT4 7.49 MB ~2.54–2.96 tok/s vs INT8 1.49 | Single-author, documents method | Medium |
| ESP-NN (official) | Espressif README, accessed 2026-10-17 | VWW 54 ms; MobileNetV3 729 ms; YOLO11n 1253 ms | Vendor benchmark | Easy to quote |
| CMSIS-NN | arXiv:1801.06601 2018; Arm/ST blogs | 4.6×/4.9×; 99.1 ms CIFAR CNN; X-CUBE-AI +82% vs TFLM | Arm/ST official | Easy to quote |
| MLPerf Tiny | arXiv:2106.07597; mlcommons.org 2026-07 | 0.30 ms-class latencies; STM32 NPU −76…−96% | Industry benchmark org | Easy to quote |

---

## 4. Ternary / quantized NNs in embedded control, robotics, drones (not LLMs)

- **Neuroflight (arXiv:1901.06553, 2019):** first open-source NN flight-controller firmware; NN replaces attitude control + motor mixing; **executes at over 2 kHz on an ARM Cortex-M7**; GymFC RL training environment.
- **Neuromorphic attitude estimation & control (arXiv:2411.13945, Nov 2024):** spiking NN end-to-end attitude control on a Crazyflie; **runs at 500 Hz on a Teensy MCU**; attitude tracking error **3.0° vs 2.7°** for the regular flight stack; 15% spiking activity.
- **PULP-Dronet family (arXiv:2407.12675, Jul 2024; arXiv:2403.11661, Mar 2024):** CNN vision navigation on GAP8 SoC: baseline **19 fps / 320 KB / ~41 MMAC**; distilled **Tiny-PULP-Dronet v3 = 2.9k params, 1.1M MAC, up to 139 fps, 168× smaller, 100% flight success**; **0.7 mJ/inference, ~100 mW**.
- **FPGA NN thrust controller for UAVs (arXiv:2403.18703, Mar 2024):** DNN low-level motor control on a Crazyflie (STM32F405 MCU + Lighthouse FPGA expansion deck); flight-tested trajectories.
- **NASA NF-15B IFCS (NTRS 20080034509, AIAA 2008):** direct adaptive **neural-network flight control system flight-tested**; "adaptive system provided better closed-loop behavior … improved vehicle stability margins" under a simulated destabilizing failure; caveat in the abstract: "continued maneuvering caused continued adaptation change".
- **Ternary hardware for tiny inference:** CUTIE complete-unrolled ternary inference engine (ETH, TinyML EMEA 2021): peak **3.1 POp/s/W @7nm (0.6 @22nm), 0.65 V**, fully unrolled, 82,944 MAC/cycle class design — evidence ternary mapping to hardware pays off, but it is custom silicon, not ESP32-class.
- **MCU toolchain reality (arXiv:2508.15008, Aug 2025, QNN survey):** ternary = 2-bit representation; "**most conventional embedded platforms lack support for native sub-byte arithmetic**, restricting practical deployment of 1- and 2-bit networks"; bit-packing overhead "can negate the theoretical gains"; NPUs such as MAX78000/MAX78002 have native low-bit support. Reports (citing prior work) "up to 19× speedup and 8× memory reduction" for binarized nets.
- **Vendor ternary support:** TI `tinyml-tensorlab` (README accessed 2026-10-17; releases 2025–2026) advertises "**Automated Ternary-weight / 8-bit-activation QAT**" and "Ternary, 4-bit Quantization support in tinyml-modelmaker", with NPU offload (TINPU). Ternary MCU workflows exist — in TI's toolchain, not for ESP32/STM32.
- **Honest gap:** no published demo of a *ternary* network controlling a drone/rocket in real time was found. The ternary evidence is: accelators (CUTIE), tooling (TI), transformer demos (ESP32 TRM-BitNet, minutes/puzzle), and quantization in control NNs (PULP-Dronet is int8, not ternary).

---

## 5. Low-rank / spectral compression & Muon

### 5.1 Low-rank compression of networks

| Candidate | Source / date | Key facts & numbers | Notes |
|---|---|---|---|
| ASVD | arXiv:2312.05821 (Dec 2023; v2 2024) | Training-free activation-aware SVD; rank −50–75% (= net 10–20% compression) with ~1% MMLU loss; further 50% KV-cache reduction without performance drop | LLaMA/LLaMA-2, LLM scale; requires activation calibration data |
| SVD-LLM | arXiv:2403.07378 (Mar 2024) | Truncation-aware data whitening + sequential low-rank update; LLaMA-2-7B WikiText-2 ppl: original 5.68 → SVD-LLM **7.73** @20% compression (ASVD 11.14) → **9.27** @40%; memory 13.5 → 10.2 → 7.76 GB | Post-training; strong at high ratios |
| SVD on Cortex-M4F | Electronics 2021, 10(1), 34 (Alessandrini et al.; code MAlessandrini-Univpm/svd-embedded) | Bare-metal SVD library implemented on ARM Cortex-M4F (STM32F4 discovery board), applied to **Kalman filtering of an IMU** | Proof SVD *runs* on MCU; it is a runtime math library, **not** NN compression |
| MCU compression pipeline (contrast) | arXiv:2205.10369 (2022) | DNN compression for Cortex-M via **pruning + weight quantization** (not SVD); AlexNet/ResNet down to 1–2 MB flash | Shows what methods MCU practice actually uses |

**Gap (verification boundary):** no public measurement was found of SVD-LLM/ASVD-class low-rank compression applied to a tiny embedded control network (nor of its latency cost per inference on an MCU). The claim "low-rank compression is proven for our case" cannot be supported; "low-rank is a proven LLM compression technique and SVD is implementable on Cortex-M4" can.

### 5.2 Muon

| Candidate | Source / date | Key facts & numbers |
|---|---|---|
| Muon (Keller Jordan blog) | kellerjordan.github.io/posts/muon/, **8 Dec 2024** | nanoGPT speedrun record improved by **1.35×**; **1.5B** transformer to GPT-2-XL HellaSwag in **10 vs 13.3 8×H100-hours (AdamW)**; CIFAR-10 record 3.3 → 2.6 A100-s; NS iteration coefficients (3.4445, −4.7750, 2.0315), runs in bf16; scope: **2D hidden-layer params only**, AdamW for embeddings/heads/biases |
| Muon repo | github.com/KellerJordan/Muon (README accessed 2026-10-17) | "1.35×" GPT-2-small record; "Used by the **Kimi.ai** frontier lab for scaled LLM training"; ships `MuonWithAuxAdam` |
| Muon is Scalable (Moonshot) | arXiv:2502.16982 (Feb 2025) | Weight decay + per-parameter update scale → out-of-the-box large-scale use; **~2× computational efficiency vs AdamW** (scaling laws); **Moonlight 3B/16B MoE, 5.7T tokens**; distributed implementation open-sourced |
| Kimi K2 | arXiv:2507.20534 (Jul 2025) | **MuonClip** (= Muon + QK-clip); **1T total / 32B active** MoE; pre-trained on **15.5T tokens with zero loss spikes** |
| PyTorch | docs.pytorch.org `torch.optim.Muon` (accessed 2026-10-17) | Muon in the official optimizer API; default NS coeffs (3.4445, −4.775, 2.0315), ns_steps=5; LR-adjust "original" (Keller) vs "match_rms_adamw" (Moonshot, 0.2·max(A,B)); docs note 2D hidden params only, AdamW for the rest |
| nanoGPT speedrun | KellerJordan/modded-nanogpt README (accessed 2026-10-17) | Muon listed among the techniques behind records; introduced 2024-10-04 with the record going 24.9 → 22.3 min; current record "under 40 seconds on 8×H100" vs llm.c's 45 min (multi-technique, not Muon alone) |

**Scope statement to use:** Muon is a *training* optimizer; it changes nothing about the on-board network's size, memory or inference latency. Smallest credible measured scale = 124M (GPT-2 speedrun) / CIFAR-10 CNN; **no evidence at ≤1M parameters and none on MCU**.

---

## 6. Comparison table (master, top candidates)

| # | Candidate | Source / date | Key facts or metrics | Credibility | Effort to adopt/verify |
|---|---|---|---|---|---|
| 1 | BitNet b1.58 | arXiv:2402.17764, 2024-02 | 3B: 2.71× faster / 3.55× less GPU mem; 71.4× matmul energy @7nm (model) | MSR, heavily cited | Quote-level |
| 2 | BitNet 2B4T | arXiv:2504.12285, 2025-04 | 0.4 GB / 29 ms / 0.028 J vs peers; avg 54.19 | Official + weights | Quote-level |
| 3 | Reloaded (small models) | arXiv:2407.09527, 2024-07 | 100K–48M params: matches/passes FP at doubled width | Independent | THE scale citation |
| 4 | bitnet.cpp | arXiv:2410.16144 + README | 1.37–6.17×, −55…−82% energy; M2 Ultra / i7-13700H | MS official | Quote (hardware=CPU) |
| 5 | T-MAC | arXiv:2407.00088 + README | ≤4× / −70% energy; 3B BitNet 30/71 tok/s M2U; 11 tok/s RPi5 | EuroSys 2025, MS | Quote (hardware=CPU) |
| 6 | ESP32-S3 stories260K | DaveBben 2024 / doryiii / Circuit-Digest | 19–30 tok/s; 2.32 tok/s no-PSRAM; 2–8 MB PSRAM | Popular repos | Re-measurable in a day |
| 7 | ESP32-S3 28.9M LLM | slvDev README 2026 | 9.88 tok/s; 94.9 ms/tok; 14.9 MB 4-bit | Single-author, detailed | Board ~$8 |
| 8 | Ternary transformer on S3 | vitorengers README 2026 | BitNet-1.58 weights, INT8 acts; minutes/puzzle | Hobby demo | Existence proof only |
| 9 | ESP-NN / CMSIS-NN | Espressif; arXiv:1801.06601 2018 | VWW 54 ms; 4.6×/4.9× CMSIS-NN | Vendor/Arm | Quote-level |
| 10 | MLPerf Tiny v1.4 | mlcommons.org 2026-07 | 0.3 ms class; ST NPU −76…−96% | Benchmark org | Quote-level |
| 11 | Neuroflight | arXiv:1901.06553, 2019 | NN flight control >2 kHz on Cortex-M7 | Peer paper | Quote-level |
| 12 | SNN control (Crazyflie) | arXiv:2411.13945, 2024-11 | 500 Hz on Teensy; 3.0° vs 2.7° | Peer paper | Quote-level |
| 13 | PULP-Dronet / Tiny v3 | arXiv:2407.12675, 2024-07 | 2.9k params, 139 fps, 0.7 mJ/inference, 100% success | Peer paper + flights | Quote-level |
| 14 | NASA NF-15B IFCS | NTRS 20080034509, 2008 | NN adaptive flight control flight-tested; better stability margins | NASA | Quote-level |
| 15 | CUTIE ternary | ETH TinyML EMEA 2021 | 3.1 POp/s/W @7nm | Academic | Context only |
| 16 | ASVD | arXiv:2312.05821 | −50…−75% rank = −10…20% net @ ~1% MMLU; −50% KV | Peer paper | Quote as LLM-scale |
| 17 | SVD-LLM | arXiv:2403.07378 | ppl 5.68 → 7.73 (20%) / 9.27 (40%) | Peer paper | Quote as LLM-scale |
| 18 | SVD on Cortex-M4F | Electronics 2021, 10(1), 34 | Bare-metal SVD for IMU Kalman filter | Peer paper + code | Feasibility only |
| 19 | Muon blog/repo | Keller blog 2024-12-08; repo | 1.35×; 10 vs 13.3 8×H100-h (1.5B); Kimi.ai uses it | Author + adoption | Quote-level |
| 20 | Muon is Scalable | arXiv:2502.16982, 2025-02 | ~2× compute efficiency; Moonlight 3B/16B, 5.7T tokens | Moonshot | Quote-level |
| 21 | Kimi K2 / MuonClip | arXiv:2507.20534, 2025-07 | 1T MoE, 15.5T tokens, zero loss spikes | Moonshot | Quote-level |
| 22 | PyTorch Muon | docs.pytorch.org, accessed 2026-10-17 | `torch.optim.Muon` official | PyTorch | Easy |

### Rejected / not usable as evidence (and why)

- Fork numbers (`jake-g/esp32-llm` ~17 tok/s) — derivative of DaveBben; use the upstream 19.13 tok/s instead.
- Secondary press (CNX Software, Hackster, Elektroda, The Register) about esp32-ai / NanoMind — not used as sources; repo READMEs are the primary.
- Syntiant NDP120 "1.80 ms / 49.59 µJ" (EE Times, 2022) — secondary; not opened primary; excluded from safe list; MLCommons' own v1.4 blog is used instead.
- FPGA binarized accelerators (LP-BNN: 21.5 µs AlexNet etc.) — FPGA, not MCU; excluded from MCU claims.
- PT-BitNet (ScienceDirect/Elsevier 2025, post-training ternary to 70B) — found via search only, not opened; not quoted with numbers.
- Russian-language sources: Russian searches (`нейросеть на микроконтроллере`, `тернарные нейросети`, `стабилизация ракеты нейросеть`) returned Habr overviews (e.g. habr.com/ru/articles/1002110, 2026-02-21 — ESP32C6 + AI platform; habr.com/ru/companies/ncloudtech/articles/878096 — local-LLM overview) and no Russian-language primary measurements. Not usable as evidence for these three technologies.

---

## 7. Safe to claim in a competition document (VERIFIED, each with source)

1. Ternary (1.58-bit, {−1,0,+1}) weights are a real, published technique with measured parity to fp16 at LLM scale: 3B BitNet b1.58 — 2.71× lower latency, 3.55× less GPU memory vs the paper's LLaMA 3B baseline [arXiv:2402.17764]; a 2B native 1-bit model trained on 4T tokens shows 0.4 GB (non-emb) / 29 ms TPOT / 0.028 J estimated vs 1.4–4.8 GB / 41–124 ms / 0.19–0.65 J for same-size fp peers [arXiv:2504.12285].
2. Ternary QAT is **published as working for small models 100K–48M params** (matching or beating fp at doubled width) [arXiv:2407.09527], and for MLPs/GNNs/encoders on par or better than 16/32-bit [arXiv:2411.05882].
3. 4-bit activations and 3-bit KV caches for 1-bit LLMs exist as a published scheme [arXiv:2411.04965] (55% of params activated).
4. Measured CPU (x86/ARM) gains for ternary inference engines: 1.37–6.17× speedup and 55.4–82.2% energy reduction [arXiv:2410.16144 / microsoft/BitNet README]; T-MAC ≤4× throughput & −70% energy vs llama.cpp, 3B BitNet at 30–71 tok/s on M2-Ultra and 11 tok/s on Raspberry Pi 5 [arXiv:2407.00088].
5. Small transformers really run on ESP32-S3 MCUs with measured numbers: 19.13–30 tok/s for a 260K-param llama2.c model (2 MB PSRAM class) [DaveBben/doryiii READMEs]; 9.88 tok/s for a 28.9M-param model with 25M params in flash [slvDev/esp32-ai README]; a ternary-weight transformer demo exists on ESP32-S3 [vitorengers/esp32-trm-bitnet README, minutes/puzzle].
6. Vendor evidence that MCU-class NN inference is fast enough for control: ESP-NN person detection 54 ms on ESP32-S3 [espressif/esp-nn README]; CMSIS-NN 4.6×/4.9× improvement and a CIFAR CNN at ~99 ms on Cortex-M7 [arXiv:1801.06601]; ST's Cortex-M33/M7 and NPU numbers in MLPerf Tiny v1.4 (−76% time STM32U3 NPU, −96% STM32H7P NPU preview vs CPU-only) [mlcommons.org 2026-07].
7. Neural controllers in flight hardware are established: NN flight control >2 kHz on Cortex-M7 [arXiv:1901.06553]; spiking control at 500 Hz [arXiv:2411.13945]; a 2.9k-param CNN vision controller at 139 fps / 0.7 mJ per inference [arXiv:2407.12675]; NASA flight-tested adaptive NN flight control [NTRS 20080034509].
8. Low-rank compression is proven for LLM post-training: ASVD rank −50–75% (net 10–20%) at ~1% accuracy loss, 50% KV cut [arXiv:2312.05821]; SVD-LLM better at 20–40% ratio [arXiv:2403.07378]. SVD itself has a bare-metal Cortex-M4F implementation (used for IMU Kalman filtering) [Electronics 2021, 10(1), 34].
9. Muon is proven as a training optimizer: 1.35× nanoGPT-speedrun record, 1.5B in 10 vs 13.3 8×H100-hours [Keller blog 2024-12-08, repo]; ~2× compute efficiency vs AdamW in LLM scaling laws and Moonlight 3B/16B MoE 5.7T tokens [arXiv:2502.16982]; production: Kimi K2 1T-param MoE, 15.5T tokens, zero loss spikes with MuonClip [arXiv:2507.20534]; officially in PyTorch [docs.pytorch.org, accessed 2026-10-17].

## 8. Do NOT claim (unsupported / would be overclaiming)

1. **"1.58-bit gives ~70× energy savings on our MCU"** — the 71.4× figure is an arithmetic-op energy *model* for matmul on 7nm chips [arXiv:2402.17764], not a measurement, and not on any MCU. ESP32/STM32 have no ternary MAC hardware; the QNN survey states most embedded platforms "lack support for native sub-byte arithmetic" [arXiv:2508.15008].
2. **"bitnet.cpp / T-MAC speedups apply to microcontrollers"** — both are desktop/laptop CPU results (x86, Apple M2, Snapdragon X Elite, RPi5) [arXiv:2410.16144, arXiv:2407.00088]; no MCU is tested.
3. **"Ternary LLMs run on the ESP32-S3 in real time"** — the only ternary transformer demo found runs ~4–6 min/puzzle [vitorengers repo]; the token demos (19–30 tok/s) are small fp32/int8 story models.
4. **"Muon will speed up / shrink our on-board network"** — Muon affects training only; no evidence below ~124M params; nothing on MCU. Any sentence linking Muon to on-board inference is false by construction.
5. **"Muon gives 2× anything for our model"** — 2× is a scaling-law comparison vs AdamW on LLM pretraining [arXiv:2502.16982]; 1.35× is a specific 124M-param speedrun [Keller blog]. Neither transfers to a <1M-param net without re-measurement.
6. **"SVD-LLM/ASVD compress embedded networks by 10–40%"** — those ratios are LLM post-training measurements requiring calibration data; no MCU deployment, no latency measurements exist publicly. Bare-metal SVD on Cortex-M4F is demonstrated only as a math library for a Kalman filter [Electronics 2021].
7. **Any "our network will do X tok/s / Y ms" number extrapolated from the demos** — every demo above ran a different model; the numbers are evidence of feasibility, not transferable performance. Measure on your own board.
8. **Russian-language sources for these techniques as primary evidence** — none found; Habr articles found are secondary overviews. Use the English primaries.
9. **Syntiant NDP120, LP-BNN, CUTIE numbers as if they applied to a commodity MCU** — custom accelerators/FPGA/custom silicon respectively.
10. **"BitNet b1.58 replaces all layers"** — the released recipe keeps some components in higher precision patterns (training keeps fp shadow weights; Muon likewise uses AdamW for non-2D params); the practical MCU recipe demonstrated is ternary weights + INT8 activations [vitorengers repo; TI tooling].

---

## 9. Open questions / not verified here

- Exact energy measurement of ternary weights vs int8 on ESP32-S3/STM32 — no public source found.
- Whether BitNet-style QAT of a control net (<1M params) reaches production control quality — untested publicly; the closest are the 100K–48M SLM/vision results [arXiv:2407.09527].
- On-MCU cost of SVD decomposition itself (all sources measure offline compression, not embedded factorization).
- UT-BitNet/PT-BitNet-class post-training ternarization (found via search only, not opened) — possible future direction, not evidence.
- Arduino/STM32 "TFLite Micro ternary" hardware path — no public benchmark found; TI's toolchain is the only vendor-verified ternary QAT path found.

Confidence in the VERIFIED lists: **high** (all numbers read from primary sources; two LLM-scale papers (2B4T, b1.58) were read in HTML; MCU numbers from the maintainers' own READMEs — small-project numbers are single-source and should be re-measured by the team before quoting as their own).
