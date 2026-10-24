import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertSelfAttention


class EpanechnikovAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        shape = (self.num_attention_heads,)
        bandwidth = torch.full(shape, math.sqrt(self.attention_head_size))
        self.bandwidth_raw = nn.Parameter(torch.log(torch.expm1(bandwidth)))
        self.kernel_affine_scale_raw = nn.Parameter(torch.log(torch.expm1(torch.ones(shape))))
        self.kernel_affine_bias = nn.Parameter(torch.zeros(shape))

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        past_key_value=None, output_attentions=False,
    ):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise ValueError("Only BERT encoder self-attention is supported")
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        distance = (
            query.square().sum(-1, keepdim=True)
            + key.square().sum(-1, keepdim=True).transpose(-2, -1)
            - 2 * (query @ key.transpose(-2, -1))
        ).clamp_min(0)
        bandwidth = (F.softplus(self.bandwidth_raw) + 1e-6).to(query)
        weights = F.relu(1 - 0.5 * distance / bandwidth.square().view(1, -1, 1, 1))
        scale = F.softplus(self.kernel_affine_scale_raw) + 1e-6
        weights = F.relu(scale.view(1, -1, 1, 1) * weights + self.kernel_affine_bias.view(1, -1, 1, 1))
        valid = 1
        if attention_mask is not None:
            valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask >= 0
            if valid.ndim == 2:
                valid = valid[:, None, None, :]
            weights = weights * valid
        empty = weights.sum(-1, keepdim=True) <= 0
        diagonal = torch.eye(query.shape[2], device=query.device, dtype=query.dtype)[None, None]
        weights = torch.where(empty, diagonal * valid, weights)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        context = (weights @ value).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.shape[:2], self.all_head_size)
        return (context, weights) if output_attentions else (context,)


class KernelBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        config.attention_kernel = "epanechnikov"
        config.renorm_kernel = False
        config.kernel_bandwidth_init = "sqrt_head_dim"
        config.kernel_self_fallback = True
        config.kernel_affine_enabled = True
        config.kernel_affine_trainable = True
        config.kernel_affine_scale_init = 1.0
        config.kernel_affine_bias_init = 0.0
        for layer in self.bert.encoder.layer:
            layer.attention.self = EpanechnikovAttention(config)


def freeze_parameters(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(part in name for part in (
            ".attention.self.query.", ".attention.self.key.", ".attention.self.value.",
            ".attention.output.dense.", "bandwidth_raw", "kernel_affine_scale_raw", "kernel_affine_bias",
        ))
