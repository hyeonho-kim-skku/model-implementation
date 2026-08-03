"""Reusable parameter storage and validation for layer-wise visual prompts."""

from numbers import Integral

import torch
import torch.nn as nn


def normalize_prompt_tokens_per_layer(value, num_layers):
    """Return a validated tuple of per-layer token counts.

    ``value`` may be a comma-separated string or an iterable of integers.  The
    caller is responsible for deciding whether a missing value should fall back
    to a scalar prompt count.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError(
                "prompt_tokens_per_layer must be a comma-separated list of integers."
            )
        try:
            counts = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(
                "prompt_tokens_per_layer must contain only integers."
            ) from error
    else:
        try:
            raw_counts = tuple(value)
        except TypeError as error:
            raise ValueError(
                "prompt_tokens_per_layer must be a list of integers or a comma-separated string."
            ) from error
        if any(isinstance(count, bool) or not isinstance(count, Integral) for count in raw_counts):
            raise ValueError("prompt_tokens_per_layer must contain only integers.")
        counts = tuple(int(count) for count in raw_counts)

    if len(counts) != int(num_layers):
        raise ValueError(
            "prompt_tokens_per_layer length must match the number of transformer "
            f"blocks ({num_layers}), got {len(counts)}."
        )
    if any(count < 0 for count in counts):
        raise ValueError("prompt_tokens_per_layer values must be non-negative.")
    return counts


class LayerwisePromptTokens(nn.Module):
    """Own independently-sized prompt parameters for transformer layers."""

    def __init__(self, token_counts, embedding_dim, init_std=0.02):
        super().__init__()
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be greater than 0.")
        if float(init_std) <= 0:
            raise ValueError("init_std must be greater than 0.")

        self.token_counts = tuple(int(count) for count in token_counts)
        if any(count < 0 for count in self.token_counts):
            raise ValueError("token_counts values must be non-negative.")
        self.embedding_dim = int(embedding_dim)
        self.init_std = float(init_std)
        self.prompts = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1, count, self.embedding_dim))
                for count in self.token_counts
            ]
        )
        for prompt in self.prompts:
            if prompt.numel() > 0:
                nn.init.trunc_normal_(prompt, std=self.init_std)

    @property
    def total_tokens(self):
        return sum(self.token_counts)

    def prompt_for_layer(self, layer_index):
        return self.prompts[layer_index]
