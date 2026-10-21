# Deep Change Monitoring: Feature Clipping reference

- Paper: *Deep Change Monitoring: A Hyperbolic Representative Learning Framework and a Dataset for Long-term Fine-grained Tree Change Detection*, CVPR 2025, local copy at `docs/参考文献/流形空间spd黎曼几何/Deep Change Monitoring（CVPR2025）.pdf`, §4.1, Eq. (7).
- Upstream code: https://github.com/liyantett/Tree-Changes-Detection-with-Siamese-Hyperbolic-network, commit `b39ff80e1baf16dc9f3b44c070f4e48cad6f89ef` (downloaded 2026-10-21).
- License: no LICENSE file in the downloaded repository; retained for research reference, not incorporated into CASA-CD code.

## Retained source content

| File | Relevant code |
|---|---|
| `Train.py` | `--clip_r` defaults to 2.3 and is passed to the model. |
| `hyptorch/nn.py` | `ToPoincare.forward`: `x * min(1, clip_r/(norm(x)+1e-5))` before `expmap0`. |
| `hyptorch/pmath.py` | `expmap0`, ball projection and Poincaré operations used by `ToPoincare`. |

Removed the nested Git checkout, duplicate archive, dataset/training utilities, unused networks, caches, and unrelated hyperbolic helpers. The retained source content was not edited; Git may normalize line endings. These files may not run standalone.

## Applicability to Run9

The paper's Feature Clipping is an **upper** norm bound, intended to stop embeddings approaching the Poincaré boundary and vanishing gradients. Run9 LEVIR stage3 instead showed radius *variation* collapsing (`r_cv=0.004133`, `rank_corr(HD,1-cos)=0.999206` at epoch 154), while `1-r²` stayed at least 0.390. An upper bound cannot force variation to reappear and, at the current HSAM scale of about 1, the source default limit 2.3 would normally be inactive. Therefore Run10 uses the existing `local_dispersion` radius mode, repairs its numerical edge cases, and checks the degeneration gate every epoch. This is a separate mechanism and has no guaranteed accuracy or convergence benefit.