"""The factorial and GPU reservation respect paired losses and physical GPU1."""
from unittest.mock import patch
from config import VARIANTS, DATASETS, experiments
from pipeline import choose_run

assert len(experiments()) == len(set(experiments())) == 24
for d in DATASETS:
    assert {(s['module'], s['interior_positive_weight']) for s in VARIANTS.values()} == {
        (m, w) for m in ('none','hyperbolic','matched') for w in (1.,1.25)}
name = 'H_SIMPLE/SYSU-CD-256'
with patch('pipeline.subprocess.check_output', return_value='15000'):
    assert choose_run([name], {}, {1:22000}, {'H_SIMPLE':11000}) == (name,1,11000)
    assert choose_run([name], {1:dict(gpu=1,reserved_mib=12000)}, {1:22000}, {'H_SIMPLE':11000}) is None
with patch('pipeline.subprocess.check_output', return_value='9000'):
    assert choose_run([name], {}, {1:22000}, {'H_SIMPLE':11000}) is None
print('24 paired cells, GPU1 placement and measured/reserved memory checks PASS')
