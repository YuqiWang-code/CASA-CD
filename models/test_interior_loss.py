"""Interior weighting changes only the first BCE and its positive interior gradient."""
import torch
import torch.nn.functional as F
from train_CASA import foreground_interior_bce, loss_function, mask_boundary


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    y = torch.zeros(1, 1, 16, 16, dtype=torch.double)
    y[:, :, 3:13, 3:13] = 1
    z = torch.randn_like(y, requires_grad=True)
    raw = F.binary_cross_entropy_with_logits(z, y)
    assert torch.equal(raw, foreground_interior_bce(z, y, 1.))
    inner = y * (mask_boundary(y, 2) == 0)
    weight = 1 + .25 * inner
    got = foreground_interior_bce(z, y, 1.25)
    expected = (weight * F.binary_cross_entropy_with_logits(z, y, reduction='none')).sum() / weight.sum()
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    a, = torch.autograd.grad(raw, z, retain_graph=True)
    b, = torch.autograd.grad(got, z)
    torch.testing.assert_close(b, a * weight * y.numel() / weight.sum())
    for target in [torch.zeros_like(y), (torch.arange(16)[None, None, :, None] == 8).expand_as(y).double()]:
        torch.testing.assert_close(foreground_interior_bce(z, target, 1.25),
                                   F.binary_cross_entropy_with_logits(z, target), atol=0, rtol=0)
    outputs = [torch.randn_like(y, requires_grad=True) for _ in range(4)]
    base = loss_function(outputs, y)
    weighted = loss_function(outputs, y, 1.25)
    expected_delta = foreground_interior_bce(outputs[0], y, 1.25) - foreground_interior_bce(outputs[0], y)
    torch.testing.assert_close(weighted - base, expected_delta)
    ga = torch.autograd.grad(base, outputs, retain_graph=True)
    gb = torch.autograd.grad(weighted, outputs)
    for i in (1, 2, 3):
        torch.testing.assert_close(ga[i], gb[i], atol=0, rtol=0)
    print('Interior BCE normalization, target band, gradients and first-head-only scope PASS')


if __name__ == '__main__':
    main()
