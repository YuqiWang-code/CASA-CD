# WDMF-Net reference

Upstream: https://github.com/LYT-Works/WDMF-Net, revision `453c5493b2aab88c908ced9135b677285059ec70`, downloaded 2026-10-21.

Paper: *A Lightweight Wavelet-Aligned Difference and Mask-Guided Fusion Network for Change Detection*, TGRS 2026. Local PDF: `docs/参考文献/baseline/WDMF-Net(TGRS2026).pdf`.

The source models, training, augmentation, dataset and metric code are retained for reference. Bundled result directories, figures and nested Git metadata are omitted. Upstream source content is retained, with line endings managed by this project. No LICENSE file was present in the downloaded repository.

The active CASA-WDMF model uses the MobileNetV2/HFEA encoder, TAR skips, DCR decoder, and four WDMF-style output heads. New processing modules and the local data-path adaptation are under `models/`; the training protocol is under `train_scripts/CASA-WDMF/Run1/`.
