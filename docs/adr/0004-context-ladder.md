# Staged context ladder to 1M with a long gate

Target context is 1M bytes, reached by stages (512 to 4k to 32k to 256k to 1M) after bulk pretraining at short context. Training 1M from scratch wastes compute on empty attention blocks. Quality A/Bs stay at s512 where they are cheap and predictive of large-scale winners (DataDecide, arXiv:2504.11393). A separate long gate measures BPB at distance (512k-1M positions vs 1-4k) after each rung. The eval tail grows from 30 MB to ~500 MB, 0.7% of the corpus, because 30 MB is only 30 windows at 1M context.

Consequence: MSA is judged by the long gate, not by short-context A/Bs (sparse-attention wins live at 32K+, arXiv:2502.11089).
