"""Faithful Isomorphic Pruning integration, isolated from project methods.

This package wraps Torch-Pruning's upstream implementation of the ECCV 2024
Isomorphic Pruning algorithm.  It intentionally does not reuse the project's
gate-Taylor collectors or joint MLP/head ranking policy.
"""

from pruning.isomorphic.pruner import prune_model_isomorphic

__all__ = ["prune_model_isomorphic"]
