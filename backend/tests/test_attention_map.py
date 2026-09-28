"""Check that visualizer uses normalized softmax attention, not patch cosine."""

import torch
import numpy as np

from ml.visualize_last_attention import cls_attention


def test_last_block_attention_shape_and_mass():
    attention = torch.nn.Module()
    attention.num_heads = 2
    attention.scale = 0.5
    attention.qkv = torch.nn.Linear(4, 12, bias=False)
    torch.nn.init.zeros_(attention.qkv.weight)
    scores = cls_attention(torch.zeros(1, 5, 4), attention)
    assert scores.shape == (2, 2)
    assert abs(float(scores.sum()) - 0.8) < 1e-6  # CLS keeps 0.2 mass.


def test_cls_row_matches_full_nonuniform_attention_matrix():
    torch.manual_seed(20260927)
    attention = torch.nn.Module()
    attention.num_heads = 2
    attention.scale = 0.5
    attention.qkv = torch.nn.Linear(4, 12, bias=True)
    tokens = torch.randn(1, 5, 4)
    q, k, _ = attention.qkv(tokens).reshape(1, 5, 3, 2, 2).permute(2, 0, 3, 1, 4).unbind(0)
    full = torch.softmax((q @ k.transpose(-2, -1)) * attention.scale, dim=-1)
    expected = full[:, :, 0, 1:].mean(1).detach().numpy().reshape(2, 2)
    np.testing.assert_allclose(cls_attention(tokens, attention), expected, atol=1e-7)
