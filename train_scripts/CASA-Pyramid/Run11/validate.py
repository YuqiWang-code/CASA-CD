"""Validate simplified pooling against actual archived weights and the loss contract."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import torch
from config import ROOT, RUNTIME, PROTOCOL

sys.path.insert(0, str(RUNTIME / 'models'))
from model.layers.hyperbolic_context import HyperbolicContext
from model.layers.class_pooling import class_assignments


def main():
    out = ROOT / 'validation'
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ROOT.parent / 'Run10/source_code.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('models/'):
                path = out / 'run10' / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    subprocess.run([sys.executable, str(Path(__file__).with_name('old_fixture.py')), str(out)], check=True)
    fixture = torch.load(out / 'equivalence.pth', weights_only=True)
    removed = fixture['state'].pop('raw_compression_temperature')
    c = HyperbolicContext().eval()
    c.load_state_dict(fixture['state'], strict=True)
    torch.set_num_threads(4)
    with torch.no_grad():
        actual, _ = c(fixture['features'])
    errors = [float((a-b).abs().max()) for a,b in zip(actual, fixture['expected'])]
    assert max(errors) < 2e-6, errors
    for v in (0., 1.):
        w, support = class_assignments(torch.full((2,1,8,8), v), 2)
        assert torch.isfinite(w).all() and torch.isfinite(support).all()
        assert (w[:, :, int(v)] == 0).all()
    tests = ['test_interior_loss.py', 'test_context_geometry.py', 'test_direct_geometry.py',
             'test_centered_readout.py', 'test_hyperbolic_context.py', 'test_CASA.py',
             'test_CASA_resume.py', 'test_data_pipeline.py']
    for name in tests:
        subprocess.run([sys.executable, str(RUNTIME / 'models' / name)], check=True)
    report = dict(protocol=PROTOCOL, passed=True, equivalence_max_diff=errors, tests=tests,
                  source='Run10 actual archived source, WHU frozen context, posterior disabled')
    (ROOT / 'validation_result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
