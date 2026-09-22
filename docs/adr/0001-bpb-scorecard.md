# BPB at fixed budget is the score

We need one comparable number for "capability per GB". We score bits per byte on the eval tail at a fixed step budget and a fixed memory envelope (16 GB VRAM plus declared host RAM), with step time as a 1.2x guardrail rather than a formula term. Params-per-GB was rejected because capacity is not capability: a 1.5B-row lookup table is not a 1.5B model. A composite score was rejected because its weights are arbitrary and every A/B argument would become an argument about weights.
