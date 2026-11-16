#!/bin/sh
# One unattended run of the corpus pipeline: tokenize every pool, then reclaim
# the extracted text, assemble the mix, and verify it.
#
#   RAYON_NUM_THREADS=4 ./run_corpus.sh [tokenizer.json]
#
# Resumable: every step reads `pre/state.json` and continues. The log is
# `pre/build.log`; `set -e` means a FAILED verify stops before anything else.
set -eu
cd "$(dirname "$0")/.."
PRE=../data_local/pre
TOK="${1:-$PRE/tokenizer32k.json}"

echo "=== $(date -Iseconds) build ($TOK)"
.venv/bin/python -u scripts/corpus.py build --tokenizer "$TOK" --batch 1024

echo "=== $(date -Iseconds) stats"
.venv/bin/python -u scripts/corpus.py stats

echo "=== $(date -Iseconds) reclaim: the token streams are the deliverable, the"
echo "    extracted text is the intermediate (13 GB, needed for mix.bin's space)"
for f in "$PRE"/text/*.bin; do
    case "$f" in
        *_eval.bin) echo "  keep $f (the previous pass's eval carve, not ours)" ;;
        *) rm -f "$f" && echo "  deleted $f" ;;
    esac
done

echo "=== $(date -Iseconds) assemble"
.venv/bin/python -u scripts/corpus.py assemble

echo "=== $(date -Iseconds) verify"
.venv/bin/python -u scripts/corpus.py verify

echo "=== $(date -Iseconds) done"
df -h /home | tail -1
