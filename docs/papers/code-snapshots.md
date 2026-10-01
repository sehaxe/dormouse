# External code snapshots

Papers are in this directory as PDFs ([provenance.tsv](provenance.tsv) names the
arXiv version and its sha256). The other kind of original is **someone's code**:
where a formula was checked against a reference implementation rather than
against prose. Those are not vendored — the code stays where its authors put it,
and this file records *which revision* we read, so a claim can name it
(AGENTS.md §1.4).

Snapshot = URL + commit + date. No code is copied into this repo.

| what we read it for | repository | commit read | license | read on |
|---|---|---|---|---|
| the Newton–Schulz coefficient triple, and its printed table | `NoahAmsel/PolarExpress` | `71cc37943d99` | MIT | 2026-09-30 |
| GDN-2 reference ops (`fla/ops/gdn2`), KDA's `kda.py` decay init | `fla-org/flash-linear-attention` | `9f38d24980c4` | MIT | 2026-09-29 |
| the official chunked KDA forward (`tests/torch_ref.py`) | `MoonshotAI/FlashKDA` | `7afb9f454f16` | MIT | 2026-09-29 |
| GatedDeltaNet-2 erase/write gates (NVlabs) | `NVlabs/GatedDeltaNet-2` | `a5552fe3c67e` | NOASSERTION | 2026-09-29 |
| Muon+ `ColRow` normalization and its param routing | `K1seki221/MuonPlus` | `8a9ace123afe` | not declared | 2026-09-29 |
| DSpark draft head, acceptance head, DeepSpec repo | `deepseek-ai/DeepSpec` | `005e03b81cec` | MIT | 2026-09-29 |
| the official Engram oracle (`engram_demo_v1.py`) | `deepseek-ai/Engram` | `fb7f84a21f91` | Apache-2.0 | 2026-09-29 |
| Attention Residuals reference implementation | `MoonshotAI/Attention-Residuals` | `85e22310fe5e` | not declared | 2026-09-29 |
| modded-nanogpt Track 3 optimization record | `KellerJordan/modded-nanogpt` | `4ea6b937337a` | MIT | 2026-09-29 |

**Two of these are not free to reuse, and neither is vendored:**
`NVlabs/GatedDeltaNet-2` declares no license (`NOASSERTION` — read, do not copy)
and `K1seki221/MuonPlus` and `MoonshotAI/Attention-Residuals` declare none
either. What we implemented came from the papers; these rows say what the
formulas were *checked against*.

`71cc37943d99` is still PolarExpress's HEAD as of 2026-10-01, which is why
AGENTS.md can cite it by short sha.