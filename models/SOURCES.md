# Sources

CASA-Pyramid uses native MobileNetV2 feature extraction adapted from [WDMF-Net](https://github.com/LYT-Works/WDMF-Net), reference revision `453c5493b2aab88c908ced9135b677285059ec70`. The reference implementation remains in `others/WDMF-Net`. HFEA is absent from this architecture. The backbone ends at the last used 320-channel feature; the pretrained 1280-channel tail and classifier are not loaded.

TAR performs temporal fusion and channel projection. DCR provides foldable decoding. `PyramidContext` implements a shared four-scale prototype pool after TAR, with complete spatial queries and compressed keys/values. The asymmetric query/context principle follows [SAT](https://arxiv.org/abs/2604.07994). Supervised change/background prototype assignments, the joint pyramid pool and matched Euclidean/hyperbolic variants form this project's experimental design.

Poincare geometry and clipping refer to [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112), [Hyperbolic Attention Networks](https://arxiv.org/abs/1805.09786) and [Deep Change Monitoring](https://arxiv.org/abs/2503.00643). These references motivate the geometry; they do not establish this model's detection accuracy. Training source and configuration archives accompany each experiment round.
