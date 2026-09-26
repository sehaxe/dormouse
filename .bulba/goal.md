# Goal: autonomous development — speed war while the official baseline trains

MODE: AWAY

STATUS 2026-09-26 (research-grounded, research/2026-09-26-small-lm-dynamics.md):
- lr sweep RUNNING (3e-4 / 1e-3 / 3e-3, no-engram probes, 750 steps each,
  eval@250/500/750): verdict 7 says 1e-4 was ~10x too low for Muon+.
- Grounding: eval unigram entropy = 5.17 BPB; our probes score ~8.0 = worse
  than counting letters. Trigram anchor ~3.3. Any run must beat 5.17 first.
- Engram stays OFF until a core proves learning (verdicts 3-5: lookup memory
  must be a minority contributor; no paper trains it on its own eval stream).
- Chinchilla re-judge point: 150 MB seen ≈ 29k steps — generalization
  verdicts before that are premature.
Queue after the sweep: winning lr -> longer probe (5-10k steps) -> measure-
then-gate the Engram (V2) -> fused Rung 1 / gradcheck SO (daytime).
