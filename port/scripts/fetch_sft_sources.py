"""Fetch the SFT shortlist into data_local/sft_raw/ (raw, unfiltered).

Chosen by docs/records/research/2026-11-14-sft-corpora.md: Russian-first, human
multi-turn where it exists, licence-clean, and small enough that a 60M model can
see each example several times in one evening.

Nothing here is converted; the builder reads these files and emits the corpus.
Re-running is cheap: `snapshot_download` skips files already present.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1] / "data_local" / "sft_raw"

#: (repo, allow_patterns) — the pattern keeps a 476 MB set from becoming 2 GB of
#: parquet shards we will subsample anyway.
SETS = [
    ("OpenAssistant/oasst2", None),
    ("IlyaGusev/ru_turbo_saiga", None),
    ("lksy/ru_instruct_gpt4", None),
    ("d0rj/alpaca-cleaned-ru", None),
    ("Den4ikAI/russian_instructions_2", None),
    ("Den4ikAI/russian_dialogues_2", None),
]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for repo, patterns in SETS:
        name = repo.split("/")[-1]
        dest = ROOT / name
        print(f"== {repo}", flush=True)
        try:
            path = snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=dest,
                allow_patterns=patterns,
                max_workers=4,
            )
        except Exception as exc:  # noqa: BLE001 — a failed download is data, not a crash
            print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(repo)
            continue
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        print(f"   ok {size / 1e6:.1f} MB -> {dest}", flush=True)

    if failed:
        print(f"\n{len(failed)} set(s) failed: {', '.join(failed)}", flush=True)
        return 1
    print("\nall sets present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
