"""Build the archived Run10 posterior-off context reference."""
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'run10/models'))
from model.layers import hyperbolic_context as impl
torch.set_num_threads(4)
torch.manual_seed(2333)
c = impl.HyperbolicContext().eval()
state = torch.load('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run10/H_POSTERIOR/WHU-CD-256/best_model.pth', map_location='cpu', weights_only=True)
c.load_state_dict({k[len('context.'):]: v for k,v in state.items() if k.startswith('context.')})
compress = impl.compress_posterior
impl.compress_posterior = lambda x,g,q,k,grid,t,d: compress(x,g,q,k,grid,t,lambda a,b:a.new_zeros((*a.shape[:-1],b.shape[-2])))
features = [torch.randn(2,64,n,n) for n in (64,32,16,8)]
with torch.no_grad():
    expected, _ = c(features)
torch.save(dict(features=features, expected=expected, state=c.state_dict()), root / 'equivalence.pth')
