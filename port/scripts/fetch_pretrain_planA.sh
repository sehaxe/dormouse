#!/usr/bin/env bash
# Plan A of docs/records/research/2026-11-14-pretrain-corpora.md: 10.02 GB, 19 files.
#
# Every line names the byte size the Hub reports, and a mismatch is a hard error:
# a truncated parquet looks exactly like a shorter document, so the only way to
# catch it is the size the API promised.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
RAW=data_local/raw
mkdir -p "$RAW"
BASE=https://huggingface.co/datasets

get() { # get <dir> <expected-bytes> <resolve-path>
  local dir=$1 want=$2 url=$3
  mkdir -p "$RAW/$dir"
  local name
  name=$(echo "$url" | sed -E 's#^.*refs%2Fconvert%2Fparquet/##; s#/(partial-)?train/#_#; s#\.parquet$##')
  local f="$RAW/$dir/$name.parquet"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" = "$want" ]; then echo "ok   $f"; return 0; fi
  curl -fsSL --retry 5 --retry-delay 3 -o "$f" "$BASE/$url" || { echo "FAIL $f"; return 1; }
  local got; got=$(stat -c%s "$f")
  [ "$got" = "$want" ] || { echo "SIZE MISMATCH $f: $got != $want"; return 1; }
  echo "ok   $f"
}

get fwedu   2153444469 "HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/sample-100BT/train/0000.parquet"
get fwedu   2152716615 "HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/sample-100BT/train/0001.parquet"
get ru      1629162947 "epfml/FineWeb2-HQ/resolve/refs%2Fconvert%2Fparquet/rus_Cyrl/train/0000.parquet"
get wiki_ru  581024230 "wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ru/train/0000.parquet"
get code    172370083  "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-mit/partial-train/0000.parquet"
get code    171433052  "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-mit/partial-train/0001.parquet"
get code    165742365  "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-apache-2.0/partial-train/0000.parquet"
get code    165124740  "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-apache-2.0/partial-train/0001.parquet"
get math    286267316  "HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/finemath-4plus/train/0000.parquet"
get math    288474354  "HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/finemath-4plus/train/0001.parquet"
get synth   248133641  "HuggingFaceTB/cosmopedia/resolve/refs%2Fconvert%2Fparquet/auto_math_text/train/0000.parquet"
get synth   253620171  "HuggingFaceTB/cosmopedia/resolve/refs%2Fconvert%2Fparquet/stanford/train/0000.parquet"
get science 285532203  "allenai/peS2o/resolve/refs%2Fconvert%2Fparquet/v1/partial-train/0000.parquet"
get wiki_en 420296449  "wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.en/train/0000.parquet"

echo "=== total"
du -sh "$RAW"
