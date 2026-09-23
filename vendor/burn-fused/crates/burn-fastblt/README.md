# burn-fastblt - Byte-Level BLT + FastBLT for Burn

[![CI](https://github.com/sehaxe/burn-fastblt/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-fastblt/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-fastblt)](https://crates.io/crates/burn-fastblt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Byte-level language modeling toolkit. No tokenizer. Universal language coverage.

> Papers: [BLT](https://arxiv.org/abs/2412.09871) (Meta, Dec 2024),
> [FastBLT](https://arxiv.org/abs/2605.08044) (Meta, May 2026).

## Install

```bash
cargo add burn-fastblt
```

## API

| Export | Paper | What |
|--------|-------|------|
| `byte_patch` | BLT | Entropy-based dynamic byte grouping |
| `bltd_loss` | FastBLT | Block diffusion auxiliary loss |
| `self_spec` | FastBLT | Self-speculation draft generator |

## License

MIT. See [LICENSE](LICENSE).
