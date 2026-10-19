# MODE: AWAY — overnight 2026-10-18

## TASK (owner's words)
"Привести всё в идеал, провести сверх-оптимизацию тренировки и инференса, сделать идеальную модель и пост. Можешь качать данные. После — запустить тренинг полноценной модели: претрейн + SFT + всё остальное. Чтобы к утру была первая модель, которая реально умная и с которой можно поговорить."

## DONE WHEN
1. Battery verdicts recorded (muon 4 arms incl. sf_muon, act_quant, sigreg) with both verdict rules.
2. A b1 pretrain run launched and COMPLETED on the optimized recipe (or resumed), checkpoint on disk.
3. SFT run on that checkpoint completed; checkpoint on disk.
4. `dormouse serve` answering on port 3783 with the SFT checkpoint; transcripts saved to ~/logs/chat_transcript_v10.txt (standard questions + mine.qa pairs).
5. NIGHT_REPORT.md in port/ + memory updated + goal.md closed with STATUS: DONE.

## PLAN (execution order)
1. Battery finishes (~04:00): verifier's final report (4 muon arms + act_quant + sigreg).
2. Resume-compatibility test: try `--set opt_graph=true` on the b1 resume → if refused, check `--init-from` for train → decision rule: (1) resume+newkeys works → resume with arms; (2) init-from works → fresh+init-from; (3) else resume-plain (model quality wins; arms validated at medium).
3. Launch pretrain (nohup, systemd-run MemoryMax=40G), b12 s512 probe first (fallback b8), eval every 1000, ckpt every 5000, --timers. Budget: end by ~09:00 for SFT window.
4. Watch pretrain (checkpoint/eval rows); on crash: diagnose, fallback form, relaunch.
5. SFT on the best checkpoint (~1h, sft6 corpus + mine.qa, 3000-4000 steps).
6. Serve + transcripts; morning report.

## SIDE LANES (CPU, parallel to GPU)
- KV-decode port (fused_recurrent_kda state-carry) — dispatched to an agent.
- Data: code/math tails download = queued for tomorrow (corpus has 19.7 GB; seen 0.3 GB — throughput is the bottleneck, not volume).

## CONSTRAINTS
- One heavy GPU thing at a time; preflight before every launch.
- Do NOT touch the desktop GUI.
- Code frozen during the pretrain (checkpoints must be one build); CPU-only agents may write files NOT read by the running trainer.
