# Sources

CASA uses MobileNetV2 feature extraction, HFEA encoder fusion and the training recipe adapted from [WDMF-Net](https://github.com/LYT-Works/WDMF-Net), reference revision `453c5493b2aab88c908ced9135b677285059ec70`. The upstream reference is retained in `others/WDMF-Net`.

The processing modules are `ChangeAwareCompressedQKV` and `HyperbolicCompressedQKV`. Their design uses full-resolution queries with compressed key/value context, following [SAT](https://arxiv.org/abs/2604.07994), and hyperbolic representations and clipping informed by [Deep Change Monitoring](https://arxiv.org/abs/2503.00643).

Temporal skips use MultiScaleTAR and decoding uses DCR. Source archives for each experiment round are stored with its logs.
