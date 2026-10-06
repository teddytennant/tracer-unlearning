"""Seeded checks for spectral reshape, confidence, and concept recovery."""

import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tracer.search import confidence, identify, spectral_reshape


def test_spectral_reshape_quadratic_forms():
    M = np.diag(np.array([4.0, 1.0, 0.001], dtype=np.float64))
    out = np.asarray(spectral_reshape(M, 1.0, 1e-2), dtype=np.float64)
    assert out.dtype == np.float64
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    q1 = float(e1 @ out @ e1)
    q2 = float(e2 @ out @ e2)
    q3 = float(e3 @ out @ e3)
    # Eq. (1) replaces kept eigenvalues with (lambda/lambda_max)**rho.
    assert q1 == pytest.approx(1.0, rel=1e-6, abs=1e-8)
    assert q2 == pytest.approx(0.25, rel=1e-6, abs=1e-8)
    assert q3 == pytest.approx(0.0, abs=1e-8)


def test_constant_confidence_is_zero():
    scores = np.ones(12, dtype=np.float64)
    assert float(confidence(scores, kappa=0.25)) == 0.0


def test_recovery_two_targets():
    # B=3 and alpha=0.5 are used for this short horizon.
    seed = 0
    np.random.seed(seed)
    _key = jax.random.PRNGKey(seed)
    n_cand = 16
    X = np.eye(n_cand, dtype=np.float64)
    groups = [0, 1]
    encodings = {0: X, 1: X.copy()}
    targets = [0, 1]
    superset = 2
    token_sets = [{i} for i in range(n_cand)]
    token_sets[superset] = {0, 99}
    imprints = []
    for g in groups:
        for t in targets:
            v = X[t]
            imprints.append((np.outer(v, v).astype(np.float64), g))
    grid = [
        (beta, gamma, rho)
        for beta in (0.0, 1.0)
        for gamma in (0.0, 1.0)
        for rho in (0.5, 1.0)
    ]
    idx, n_hat = identify(
        imprints,
        encodings,
        token_sets,
        groups,
        grid,
        J=6,
        B=3,
        alpha=0.5,
        kappa=0.25,
    )
    assert n_hat == 2
    assert idx == targets
    assert superset not in idx


def test_no_imprints_above_threshold():
    seed = 1
    np.random.seed(seed)
    X = np.eye(4, dtype=np.float64)
    encodings = {0: X}
    token_sets = [{i} for i in range(4)]
    groups = [0]
    grid = [(0.0, 0.0, 1.0), (1.0, 1.0, 0.5)]
    tiny = (1e-12 * np.eye(4, dtype=np.float64), 0)
    idx, n_hat = identify([], encodings, token_sets, groups, grid)
    assert idx == []
    assert n_hat == 0
    idx, n_hat = identify([tiny], encodings, token_sets, groups, grid, J=6, B=3, alpha=0.5)
    assert idx == []
    assert n_hat == 0
