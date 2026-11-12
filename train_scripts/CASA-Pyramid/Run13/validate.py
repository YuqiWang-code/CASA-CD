"""Exercise the ablation switches and queue paths before GPU training."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from config import DATA, DATASETS, MATRIX, RUNTIME, TRAINABLE_PARAMS, VARIANTS, experiments, model_options
from pipeline import pending_runs

sys.path.insert(0, str(RUNTIME/'models'))
from model.casa_net import CASANet


def main():
    torch.set_num_threads(4)
    assert len(experiments()) == len(set(experiments())) == 6
    assert set(VARIANTS) == {'TAR_PLAIN', 'DCR_PLAIN'}
    for variant, (module, heads, detail, tar, decoder) in VARIANTS.items():
        assert module == 'none' and heads == 1 and not detail
        assert (tar, decoder) in (('plain', 'reparam'), ('reparam', 'plain'))
        options = model_options(variant)
        assert options[options.index('--tar-variant')+1] == tar
        assert options[options.index('--decoder-variant')+1] == decoder
    with TemporaryDirectory() as directory:
        complete, stopped, unstarted = experiments()[:3]
        path = Path(directory)/complete
        path.mkdir(parents=True)
        (path/'train_log.txt').write_text('=== TEST RESULTS ===\nF1=0.9\n=== END TEST RESULTS ===\n')
        with patch('pipeline.ROOT', Path(directory)):
            pending, finished = pending_runs({'stopped': [stopped]})
            assert finished == [complete] and pending[0] == stopped and unstarted in pending
    for dataset in DATASETS:
        directory = DATA/dataset/'list'
        train, val, test = [(directory/f'{split}.txt').read_text().split() for split in ('train', 'val', 'test')]
        assert all(len(x) == len(set(x)) for x in (train, val, test))
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))

    image = torch.randn(1, 6, 256, 256)
    torch.manual_seed(2333)
    control = CASANet().eval()
    with torch.no_grad():
        reference = control.predict(image[:, :3], image[:, 3:])
    for variant, (module, heads, detail, tar, decoder) in VARIANTS.items():
        torch.manual_seed(2333)
        model = CASANet(module=module, context_heads=heads, detail_head=detail,
                        tar_variant=tar, decoder_variant=decoder).eval()
        assert sum(p.numel() for p in model.parameters()) == TRAINABLE_PARAMS[variant]
        assert model.tar_variant == tar and model.decoder_variant == decoder
        torch.manual_seed(2333)
        folded = CASANet(module=module, context_heads=heads, detail_head=detail,
                         tar_variant=tar, decoder_variant=decoder).eval()
        folded.switch_to_deploy()
        assert sum(p.numel() for p in folded.parameters()) == 1935810
        with torch.no_grad():
            outputs = model(image[:, :3], image[:, 3:])
            deployed = folded.predict(image[:, :3], image[:, 3:])
        assert len(outputs) == 4 and outputs[0].shape == (1, 1, 256, 256)
        assert torch.allclose(outputs[0], deployed, atol=2e-4)
        assert all(torch.isfinite(out).all() for out in outputs)
        # The ablated arm must actually differ from the control, not be a silent no-op.
        assert not torch.allclose(outputs[0], reference, atol=1e-6), variant
        logits, aux = model(image[:, :3], image[:, 3:], return_aux=True)
        logits[0].mean().backward()
        names = ([n for n, _ in model.named_parameters() if n.startswith('tar.stage1.temporal')]
                 if tar == 'plain' else [n for n, _ in model.named_parameters() if n.startswith('decoder.fuse1')])
        grads = dict(model.named_parameters())
        assert all(grads[n].grad is not None and grads[n].grad.norm() > 0 for n in names), variant
    for dataset, arms in MATRIX.items():
        assert arms == tuple(VARIANTS)
    print('Run13 matrix, ablation switches, folding and gradient paths PASS', flush=True)


if __name__ == '__main__':
    main()
