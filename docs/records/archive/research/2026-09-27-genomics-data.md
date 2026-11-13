# 2026-09-27 — The genetics arm: real data, honest anchors

The plan in `2026-09-27-universal-domains.md` is now a corpus on disk with
measured baselines. Nothing in `crates/` changed; the only new artefact is this
report and the data itself.

Everything below was measured on this box. Where I could not verify something, it
says so instead of assuming it.

## 1. What was fetched

14 genomes from NCBI RefSeq, spanning all 7 domains, plus one superseded file
kept on disk. **Every `.gz` was verified against NCBI's own
`md5checksums.txt` (15/15 OK)** and **every organism name was read off the
file's own FASTA header** — not off a filename, and not off a hand-written
accession list. That distinction cost three wrong labels (see §2).

Raw dir: `/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/genomics_raw/`
(`<label>__<accession>_<assembly>_genomic.fna.gz`, plus the decompressed `.fna`
next to it, as the trainer reads plain bytes).

| organism (from header) | lineage | RefSeq assembly | gz MB | fna MB | records | BioProject |
|---|---|---|---|---|---|---|
| Homo sapiens | Chordata/Mammalia | GCF_000001405.40 GRCh38.p14 | 972.9 | 3339.7 | 705 | PRJNA31257 |
| Mus musculus | Chordata/Mammalia | GCF_000001635.27 GRCm39 | 834.3 | 2762.3 | 61 | PRJNA20689 |
| Danio rerio | Actinopterygii | GCF_000002035.6 GRCz11 | 528.4 | 1700.4 | 1923 | PRJNA11776 |
| Drosophila melanogaster | Arthropoda/Insecta | GCF_000001215.4 Rel 6 + ISO1 + MT | 44.2 | 145.7 | 1870 | PRJNA13812 |
| Arabidopsis thaliana | Plantae/Brassicaceae | GCF_000001735.4 TAIR10.1 | 37.5 | 121.2 | 7 | PRJNA10719 |
| Caenorhabditis elegans | Nematoda | GCF_000002985.6 WBcel235 | 31.7 | 101.5 | 7 | PRJNA13758 |
| Saccharomyces cerevisiae S288C | Fungi | GCF_000146045.2 R64 | 3.8 | 12.3 | 17 | PRJNA43747 |
| Escherichia coli K-12 MG1655 | Pseudomonadota | GCF_000005845.2 ASM584v2 | 1.4 | 4.7 | 1 | PRJNA225 |
| Mycobacterium tuberculosis H37Rv | Actinobacteriota | GCF_000195955.2 ASM19595v2 | 1.3 | 4.5 | 1 | PRJNA224 |
| Lachnospira eligens | Bacillota/Clostridia | GCF_000146185.1 ASM14618v1 | 0.8 | 2.9 | 3 | PRJNA29073 |
| Thermus thermophilus HB8 | Deinococcota | GCF_000091545.1 ASM9154v1 | 0.6 | 2.1 | 3 | PRJNA13202 |
| Methanocaldococcus jannaschii | Archaea/Euryarchaeota | GCF_000091665.1 ASM9166v1 | 0.5 | 1.8 | 3 | PRJNA102 |
| *Plasmodium falciparum* 3D7 — **held out** | Protista/Apicomplexa | GCF_000002765.6 | 6.7 | 23.6 | 14 | PRJNA13173 |
| *Staphylococcus aureus* N315 — **held out** | Bacillota/Firmicutes | GCF_000009645.1 ASM964v1 | 0.8 | 2.9 | 2 | PRJNA264 |

Fetched raw: **2.46 GB gzipped, 8.23 GB decompressed** (14 genomes; a 15th,
superseded, file is on disk but unused). The trainable slice after budgeting is
1.529 GB and the held-out slices add 144.5 MB, so **1.674 GB of usable data**,
inside the 1-3 GB target.

URLs are all of the form
`https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/<ddd>/<ddd>/<ddd>/<accession>_<assembly>/<accession>_<assembly>_genomic.fna.gz`
and are listed per file in `genomics_raw/provenance.json` (with md5 and
BioProject), machine-readable by `shards/manifest.tsv`.

**Licence — what I actually verified.** NCBI's
[Policies and Disclaimers](https://www.ncbi.nlm.nih.gov/home/about/policies/),
section *Molecular Data Usage* (fetched 2026-09-27), says verbatim: *"NCBI
itself places no restrictions on the use or distribution of the data contained
therein. Nor do we accept data when the submitter has requested restrictions on
reuse or redistribution."* The same page adds the caveat that some submitters
(or their country of origin) may claim patent or copyright over part of what
they submitted, which NCBI cannot adjudicate. The *Copyright Status of Webpages*
section puts US-government-created information in the public domain and asks only
for attribution to NLM. So: treat as public domain, cite NCBI/RefSeq, and note
that RefSeq *annotations* are NCBI's own US-government work while the underlying
reads come from submitters.

## 2. Three of the inherited labels were wrong — check headers, not filenames

The accession list I inherited (`/tmp/opencode/genome_urls.txt` and the first
`fetch_genomes.sh`) was wrong in ways that all *looked* fine:

| label as inherited | accession | what the file actually is |
|---|---|---|
| `mouse` | GCF/000/001/215 | *Drosophila melanogaster* Release 6 (mouse is GCF_000001635) |
| `yeast` | GCF/000/146/185 | *Lachnospira eligens*, a Clostridia bacterium (S288C is GCF_000146045) |
| `synechococcus` | GCF/000/009/645 | *Staphylococcus aureus* N315 |
| `methanocaldococcus` | GCF/000/091/545 | *Thermus thermophilus* HB8 |
| `arabidopsis` | GCF/000/002/035 | *Danio rerio* GRCz11 (arabidopsis is GCF_000001735) |

The files were mislabelled, not corrupt — every md5 matched. Two more traps from
the same session, both now handled in `/tmp/opencode/fetch_genomes3.sh`:

* **`head -1` on the accession listing picks the OLDEST assembly.** For human
  that is `GCF_000001405.10_NCBI34`, a 2005 build. The accession *version* lives
  inside field 2 of the name (`000001405.40`), so `sort -t_ -k3` sorts on the
  assembly name and silently returns NCBI34. Fixed by resolving the accession
  from an authority (NCBI Datasets `dataset_report`, or esearch/esummary) and
  matching the FTP listing on the **exact accession prefix**, which removes the
  version question entirely.
* **The directory name and the file name differ.** The dir is
  `${accession}_${asm}` but the file inside can be `${accession}_${asm_name}_…`
  (`GCF_000146045.2_R64/` holds `GCF_000146045.2_R64-1-1_genomic.fna.gz`), and
  `*_cds_from_genomic.fna.gz` also ends in `_genomic.fna.gz`. Resolve the
  directory, then list *its* contents, and reject anything whose header is not
  the genome.

A parallel `curl` pool also lost log lines, so a wrong-organism download can
pass unnoticed — the per-file header check is the only thing that caught
*Staphylococcus aureus*. One `nohup`'d job was killed when the tool call that
launched it hit its timeout; `setsid` it.

The old *Drosophila* Release 3 (`GCF_000334755.1`) is on disk as
`drosophila_superseded__*` and is **not** in the corpus.

## 3. Shard layout and the split

Built by `/tmp/opencode/build_shards.py`; the human-readable record with the full
provenance table is `genomics_raw/shards/SPLIT.md`, the machine-readable one is
`shards/manifest.tsv` / `manifest.json` / `verify.json`.

```
genomics_raw/shards/
  train/corpus_000.bin .. corpus_015.bin   1.529 GB, 4573 records, 16 FNV shards
  train_corpus.txt                          sharder input, records separated by \n\n
  eval_species/plasmodium__*.fna            23.6 MB   whole genome, nowhere in train
  eval_species/staphylococcus__*.fna         2.9 MB   whole genome, nowhere in train
  eval_chrom/human_held_out_records.fna     55.8 MB   every record filed as chr22
  eval_chrom/mouse_held_out_records.fna     62.2 MB   every record filed as chr19
  manifest.tsv  manifest.json  verify.json  SPLIT.md
```

`train/corpus_*.bin` came from the repo's own `shard` binary
(`FNV(doc) % 16`), so **each shard is a uniform random sample of the whole
slice**: shard 000 alone contains drosophila 119, zebrafish 115, human 41,
mouse 5, yeast 1 records. That is what makes the mixture survive the trainer's
file-order shuffle. Shards sum to the corpus byte-for-byte (1 529 180 211 B,
zero loss).

Two held-out axes, both excluded **structurally at build time** (ADR-0010 — not
by convention, not upstream of the thing that trains):

1. **Unseen species** — *P. falciparum* (~80% AT) and *S. aureus* N315 (low-GC
   Gram-positive). Whole genomes, absent from train. Both sit outside the
   training composition, so the number is not a near-trivial repeat.
2. **Unseen chromosome of a seen species** — every record that *announces
   itself* as human chr22 or mouse chr19, including the GRCh38 unlocalized and
   alt-locus scaffolds filed under chr22. Those scaffolds **are** chr22
   sequence, so keeping them in train would quietly leak the eval chromosome;
   this is the leak the exclusion exists to prevent.

Verified mechanically: 44 held-out record ids, **0** present in train
(`verify.json → leaked_ids: []`).

**Byte budget.** The three giants are capped by keeping the first `cap/size` of
*every* record, cut at a line boundary: human 0.1497, mouse 0.1448, zebrafish
0.1470. The other nine are whole. Train composition by bytes: human 32.2%,
mouse 25.6%, zebrafish 16.3%, drosophila 9.5%, arabidopsis 7.9%, worm 6.6%,
yeast 0.8%, bacteria+archaea 0.9% — **vertebrate-dominated by construction**,
while by record count zebrafish+drosophila scaffolds are 83%.

Two known simplifications, both deliberate:

* For a capped genome the kept part is the **first** ~15% of each record, so
  chromosome tails (telomeres, centromeres, satellite) are absent from train.
  The obvious alternative — uniform Bernoulli over records — is unbiased in
  expectation but has sd ≈ 50% of the budget on heavy-tailed chromosome sizes: it
  actually produced **177 MB against a 500 MB cap and 842 MB against a 400 MB
  one**, and can drop a whole chromosome. If the tail ever matters, subsample
  *within* records at fixed block stride instead.
* Sequence lines here are **80 columns (81 bytes)**, not the 60 I assumed;
  cutting on a 61-byte stride glued headers onto sequence lines and made records
  unparseable. The cut is now `rfind(b"\n")` — width-agnostic.

## 4. Anchors

Run with the repo's own tool (`dormouse-data --bin anchors`, already built at
`target/release/anchors`; I did not rebuild it — no crates changed). Full logs:
`/home/sehaxe/logs/genomics_anchors.log` (68 runs, published FASTA) and
`/home/sehaxe/logs/genomics_anchors_trimmed.log` (32 runs, gap-trimmed);
drivers `/tmp/opencode/run_anchors.sh` and `run_anchors_trim.sh`. The tool's own
split keeps the trailing 25% of each read as held-out and fits the n-gram model
on the first 75% only — which turns out to matter a lot, see §4.3.

### 4.1 The whole slice and both held-out axes

`--bytes 2000000000` was requested; the train slice is 1.529 GB, so that read
consumed all of it (1.147 GB fit / 0.382 GB held out by the tool's own trailing
25% split). "hdr" = FASTA headers left in, "seq" = `--skip-header`.

| slice | read | unigram | 5-gram+backoff | 8-gram+backoff | 8-gram unseen ctx |
|---|---|---|---|---|---|
| train slice, order 5, hdr | 1.529 GB | 3.189 | **2.030** | | 0.0% |
| train slice, order 5, seq | 1.529 GB | 3.185 | **2.030** | | 0.0% |
| train slice, order 8, hdr | 1.529 GB | 3.189 | | **1.991** | 0.0% |
| train slice, order 8, seq | 1.529 GB | 3.185 | | **1.991** | 0.0% |
| eval_species, order 5, hdr | 0.026 GB | 2.970 | 2.064 | | 0.0% |
| eval_species, order 8, seq | 0.026 GB | 2.969 | | **2.057** | 1.4% |
| eval_chrom, order 5, hdr | 0.118 GB | 3.205 | 2.228 | | 0.0% |
| eval_chrom, order 8, seq | 0.118 GB | 3.205 | | **2.202** | 0.6% |

Four things this says:

1. **Headers are worth nothing here** — ≤0.004 BPB either way, because they are
   0.03% of the bytes. So the earlier "2.034 vs 2.040 with/without headers" was
   noise, and a model cannot win by modelling FASTA headers.
2. **The 8-gram floor is 1.991, i.e. the 4-letter uniform floor.** ln 4 = 2.0
   bits/byte, and an 8-gram counter with backoff gets 1.991. Going 5 → 8 buys
   0.039 BPB. A counter has captured essentially *nothing* beyond base
   composition: every bit of headroom on this data is long-range, which is
   exactly the claim the plan makes, now with a number on it.
3. **The generalization gap is the real bar.** Unseen chromosome of a seen
   species = **2.202**, i.e. **+0.211 BPB** over the in-slice 1.991. Unseen
   species = 2.057, only +0.066. Counter-intuitive but consistent: a *new
   species* is mostly new base composition plus new 3-mers (the unigram itself
   drops to 2.970), while a *held-out chromosome* is a sequence the model has
   seen 15% of, so the n-gram model transfers and the residual 0.2 BPB is
   genuinely local-but-not-repeated structure. **2.202 is the number a genomics
   run has to beat.**
4. 0.0% unseen contexts everywhere except eval_species (1.4%) and eval_chrom
   (0.6%): over a 4-letter alphabet almost every 7-gram has been seen, so
   "unseen context" backoff is not what separates these numbers.

### 4.2 Per organism — the bar is not one number

Read = first 150 MB of each genome (whole file for the smaller ones). "trim" =
the same bytes with every gap character removed.

| genome | read | unigram | 5g | 8g | 5g trim | 8g trim | 8g trim unseen ctx |
|---|---|---|---|---|---|---|---|
| Thermus thermophilus | 2 MB | 1.965 | 1.841 | 1.788 | 1.873 | **1.818** | 2.8% |
| Methanocaldococcus jannaschii | 2 MB | 1.967 | 1.904 | 1.884 | 1.961 | 1.919 | 7.7% |
| *S. aureus* N315 (held out) | 3 MB | 1.993 | 1.970 | 1.960 | 1.970 | 1.960 | 1.7% |
| M. tuberculosis H37Rv | 4 MB | 2.000 | 1.950 | 1.937 | 1.950 | 1.937 | 0.9% |
| Lachnospira eligens | 3 MB | 2.027 | 1.982 | 1.971 | 1.982 | 1.971 | 1.6% |
| E. coli K-12 MG1655 | 5 MB | 2.071 | 2.018 | 2.007 | 2.018 | 2.007 | 0.8% |
| *S. cerevisiae* S288C | 12 MB | 2.551 | 2.119 | 2.116 | 2.119 | 2.116 | 3.0% |
| *P. falciparum* 3D7 (held out) | 24 MB | 2.685 | 1.911 | 1.890 | 1.911 | 1.890 | 2.0% |
| Arabidopsis thaliana | 121 MB | 2.752 | 2.148 | 2.132 | 2.178 | 2.163 | 0.9% |
| C. elegans | 102 MB | 2.898 | 2.176 | 2.157 | 2.176 | 2.157 | 0.7% |
| Drosophila melanogaster | 146 MB | 3.024 | 2.151 | 2.120 | 2.229 | 2.196 | 0.7% |
| Mus musculus | 150 MB | 3.026 | 2.193 | 2.130 | 2.243 | 2.183 | 0.8% |
| Danio rerio | 150 MB | 3.027 | 2.209 | 2.162 | 2.257 | 2.212 | 0.9% |
| Homo sapiens | 150 MB | 5.724 | 1.385 | 1.362 | 2.246 | **2.177** | 0.5% |

The floor spans **0.4 BPB across lineages** (1.818 *Thermus* → 2.212 zebrafish),
so a single "genomics BPB" is meaningless. The eukaryote cluster sits at
2.12-2.22, the bacteria/archaea at 1.82-2.01. A mixed-domain eval number is
dominated by which genomes happen to be in it — which is exactly why the two
held-out axes are kept separate above.

### 4.3 The assembly-gap trap (this one is worth the whole report)

The human row is a measurement artifact and I only caught it by reproducing
5.724 BPB by hand:

* GRCh38.p14's primary assembly really does start with a multi-Mb `N` pad, and
  chr1's first 150 MB straddles a large assembly gap.
* The anchors tool's held-out split is the **trailing 25% of the byte stream**.
  For human that window is **48.2% `N`** while the train 75% is **0.3% `N`** —
  the unigram model assigns `N` a tiny probability and the unigram reads
  **5.724 BPB**. Zero bytes were unseen (I checked), so the tool is correct and
  the *sample* is pathological.
* The 5-gram/8-gram numbers are flattered the same way, because a run of `N` is
  trivially predictable: human reads **1.362 BPB**, below the 4-letter uniform
  floor, which is impossible for real sequence.
* Removing every gap character from the same 150 MB: unigram 3.073, 5-gram
  2.246, **8-gram 2.177** — in line with mouse (2.183) and zebrafish (2.212).

**7.7% of the train corpus and 12.6% of eval_chrom are gap `N`.** So the
headline 1.991 is a slightly optimistic floor: some of that 7.7% is free for a
counter. Gap-trimming the whole slice gives 2.167 at order 8 (on a 150 MB read
fit on 112 MB, so it is also a 10× smaller fit — the two effects are not
separated in that number; the per-organism table above is the clean comparison,
same bytes, N removed).

Recommendation: **gap-trim the genomics arm** (`anchors_trimmed/` in the raw dir
has the trimmed copies, produced by `/tmp/opencode/trim_gaps.py`). 7.7% of a 7%
mixture is half a percent of the whole run spent on predicting `N`, and it
corrupts every anchor measured on published RefSeq FASTA. The corpus in
`shards/train` is left as published FASTA (faithful to the source) and flagged
here; trimming it is a one-line change in `build_shards.py` if you want it.


## 5. What fraction of the corpus should be genomics

The trainer drains one file at a time in a per-epoch shuffled order
(`ByteStream::refill`, 8 MB chunks, next file at EOF), so **the mixture weight
is exactly bytes-per-epoch**. There is no sampler, no weight table, nothing to
configure: bytes are the weight.

| base corpus | size | genomics 1.529 GB = | to hit 5% | to hit 15% |
|---|---|---|---|---|
| `pretrain/real_sharded` (current text run) | 20.36 GB | **7.0%** | 1.07 GB | 3.59 GB |
| `pretrain/mix` (the 373 GB multi-domain mix) | 372.97 GB | 0.41% | 19.6 GB | 65.8 GB |

**Recommendation: 7% against the current 20.36 GB text corpus — keep it as is.**
It is the middle of the requested 5-15% band, and the slice is already
vertebrate-heavy, so pushing to 15% would spend the budget on more of the same
three genomes rather than on breadth.

The number that matters for the *other* corpus: 1.5 GB is **not** a 5-15% arm of
the 373 GB `mix/` — it is 0.4%. A 5% genomics arm there needs ~20 GB and a 15%
arm ~66 GB, which is the scale of the RefSeq vertebrate+invertebrate panels
(NCBI ships tens of thousands of assemblies under `genomes/all/GCF/`). That is a
breadth play, not a download problem: add more `*_genomic.fna.gz` to the same
`genomics_raw/`, rerun `build_shards.py`, and the mixture re-weights itself with
no code change. Until then, do not put genomics into the 373 GB mix and call it
a mixture — it will be invisible.

## 6. The exact command

No code change is required to train on the mixture. The loader recurses
subdirectories, so a directory holding both corpora is the whole mechanism:

```bash
G=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data
mkdir -p $G/mix_text_genomics
ln -sfn $G/pretrain/real_sharded            $G/mix_text_genomics/text
ln -sfn $G/genomics_raw/shards/train        $G/mix_text_genomics/genomics
```

```bash
free -g | awk 'NR==2{print "avail GB:",$7}'          # doctrine: >= 25, production >= 40
pgrep -ax train || true                              # one heavy thing at a time
systemd-run --user --scope -p MemoryMax=40G \
  ./target/release/train \
    --data  $G/mix_text_genomics \
    --eval  $G/genomics_raw/shards/eval_chrom \
    --preset small --batch 10 --seq-len 512 \
    --ckpt-name text_genomics --ckpt-dir checkpoints \
    --eval-every 500 --guard \
    --log /home/sehaxe/logs/train_text_genomics.log
```

Read the shards for a mixture check before spending GPU time — a dry count of
what the loader will see:

```bash
ls $G/mix_text_genomics/text/*.bin $G/mix_text_genomics/genomics/*.bin | wc -l   # 80 files
du -sb $G/mix_text_genomics/text $G/mix_text_genomics/genomics                     # 20.36 GB / 1.53 GB
```

**Two traps in this layout:**

* **Never point `--data` at `genomics_raw/`.** `.gz` is in the loader's
  `bin_exts`, so the 2.5 GB of *compressed* files would be read as training bytes
  — training on gzip noise. Only `--data .../shards/train` (plain `.bin`) is
  safe. This is why the shards live in a subdirectory rather than beside the
  raw downloads.
* **`--eval` takes ONE directory.** The trainer cannot watch text and genomics
  in the same run. The command above watches the genomics held-out slice (the
  new signal); for text continuity use
  `--eval $G/pretrain/real_eval_v2` instead, or alternate. Making both work at
  once is the one code change I would actually ask for, and it is mechanical:
  a second `--eval2 <dir>` flag on `train.rs` (mirroring `--eval`) and a second
  `ByteStream` next to `eval_data` at `crates/dormouse-train/src/lib.rs:623`,
  scored on the existing eval path. ~10 lines, no model-side change. I did not
  write it, per the brief.

## 7. Honest limits of this slice

* 12 genomes is breadth-of-lineage, not breadth-of-life: one strain per species,
  no individual human variation, no structural variants, no RNA-seq, no
  multi-allelic FASTA. The byte stream is one reference per genome.
* Only ACGT + ASCII headers are represented. Real genomics pipelines also need
  bisulfite/alignment formats, and the 80-column wrap is itself a learnable
  regularity the model will exploit (a line ends every 81 bytes).
* The train slice is vertebrate-dominated (§3), the big three are truncated
  per record (§3), and **7.7% of its bytes are assembly-gap `N` (§4.3)** — all
  three are deliberate, all three are stated, none of them is hidden in a
  number.
* A byte-level LM on genomes is not a genomics model: there is no reverse
  complement augmentation, no k-mer framing, and the 4-letter floor (ln 4 =
  2.0 bits/byte) is *below* the unigram anchor measured here, which is the whole
  reason the headroom is all long-range.
* Not verified: whether these exact numbers reproduce on the GPU trainer. The
  anchors are static n-gram baselines computed on CPU; no training run was
  started (another agent owns the GPU).
