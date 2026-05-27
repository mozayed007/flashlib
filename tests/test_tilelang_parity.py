"""TileLang backend parity tests.

Verifies TileLang kernel outputs match the Triton (and torch reference)
outputs within published precision tolerances.  These tests are CUDA-only
and skip cleanly on CPU CI.

Structure follows ``tests/test_backend_parity.py`` — one test function
per primitive, grouped by phase.

Run on Modal H100::

    modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py

Or locally with CUDA::

    pytest tests/test_tilelang_parity.py -v
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for TileLang parity tests", allow_module_level=True)

try:
    from flashlib._tilelang import tilelang_available
    if not tilelang_available():
        pytest.skip("tilelang not installed", allow_module_level=True)
except Exception:
    pytest.skip("tilelang not importable", allow_module_level=True)

DEVICE = "cuda"
SEED = 42


def _seeded(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _torch_kmeans_assign(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Reference assignment: argmin_j ||x_i - c_j||^2."""
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    c_sq = (c * c).sum(dim=-1).unsqueeze(-2)
    cross = torch.matmul(x.float(), c.float().transpose(-1, -2))
    dist = x_sq + c_sq - 2.0 * cross
    return dist.argmin(dim=-1).to(torch.int32)


# ---------------------------------------------------------------------------
# Phase 1: KMeans
# ---------------------------------------------------------------------------


class TestKMeansTileLang:
    """Phase 1 — KMeans assign + full Lloyd loop parity."""

    # Phase 1 kernel not yet implemented — tilelang_assign_euclid raises NotImplementedError
    @pytest.mark.skip(reason="Phase 1 kernel not yet implemented")
    def test_assign_matches_torch_reference(self):
        """TileLang assign should agree with the torch reference."""
        _seeded()
        B, N, D, K = 1, 4096, 128, 64
        x = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)
        c = torch.randn(B, K, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.kmeans.tilelang.assign import tilelang_assign_euclid
        tl_ids = tilelang_assign_euclid(x, c).to(torch.int32)
        ref_ids = _torch_kmeans_assign(x, c)

        mismatch_rate = (tl_ids != ref_ids).float().mean().item()
        assert mismatch_rate < 5e-3, (
            f"TileLang assign disagrees with torch ref: {mismatch_rate:.4f}"
        )

    @pytest.mark.skip(reason="Phase 1 kernel not yet implemented")
    def test_assign_matches_triton(self):
        """TileLang assign should agree with Triton assign."""
        _seeded()
        B, N, D, K = 1, 4096, 128, 64
        x = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)
        c = torch.randn(B, K, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.kmeans.tilelang.assign import tilelang_assign_euclid
        from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton

        tl_ids = tilelang_assign_euclid(x, c).to(torch.int32)
        triton_ids = euclid_assign_triton(x, c).to(torch.int32)

        mismatch_rate = (tl_ids != triton_ids).float().mean().item()
        assert mismatch_rate < 5e-3, (
            f"TileLang vs Triton assign mismatch: {mismatch_rate:.4f}"
        )

    @pytest.mark.skip(reason="Phase 1 — impl.py tilelang branch not yet wired in (falls through to triton)")
    def test_full_lloyd_loop(self):
        """TileLang-backed flash_kmeans should produce valid clusters."""
        _seeded()
        B, N, D, K = 1, 2048, 64, 16
        x = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.kmeans.impl import flash_kmeans
        cluster_ids, centroids, n_iter = flash_kmeans(
            x, K, max_iters=10, backend="tilelang",
        )

        assert cluster_ids.shape == (B, N)
        assert centroids.shape == (B, K, D)
        assert 1 <= n_iter <= 10
        assert cluster_ids.min() >= 0 and cluster_ids.max() < K


# ---------------------------------------------------------------------------
# Phase 2: KNN
# ---------------------------------------------------------------------------


class TestKNNTileLang:
    """Phase 2 — KNN top-K parity."""

    @pytest.mark.skip(reason="Phase 2 — impl.py tilelang branch not yet wired in (falls through to triton)")
    def test_topk_matches_triton(self):
        _seeded()
        B, N, D, k = 1, 1024, 64, 5
        x = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)
        c = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.knn.impl import flash_knn_dispatch
        vals_tl, idxs_tl = flash_knn_dispatch(x, c, k, backend="tilelang")
        vals_tr, idxs_tr = flash_knn_dispatch(x, c, k, backend="triton")

        match_rate = (idxs_tl == idxs_tr).float().mean().item()
        assert match_rate > 0.9, f"KNN index match rate: {match_rate:.4f}"


# ---------------------------------------------------------------------------
# Phase 3: PCA + Truncated SVD
# ---------------------------------------------------------------------------


class TestPCATileLang:
    """Phase 3 — PCA parity."""

    def test_pca_matches_triton(self):
        _seeded()
        N, D, K = 1024, 64, 8
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.pca.impl import flash_pca
        (evals_tl, evecs_tl) = flash_pca(X, K, backend="tilelang")
        (evals_tr, evecs_tr) = flash_pca(X, K, backend="triton")

        # Eigenvalues should be close
        torch.testing.assert_close(evals_tl, evals_tr, rtol=1e-2, atol=1e-2)


class TestTruncatedSVDTileLang:
    """Phase 3 — Truncated SVD parity."""

    def test_svd_matches_triton(self):
        _seeded()
        N, D, K = 1024, 64, 8
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.truncated_svd.impl import flash_truncated_svd
        S_tl, Vh_tl = flash_truncated_svd(X, K, backend="tilelang")
        S_tr, Vh_tr = flash_truncated_svd(X, K, backend="triton")

        torch.testing.assert_close(S_tl, S_tr, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Phase 4: GEMM variants
# ---------------------------------------------------------------------------


class TestCovGEMMTileLang:
    """Phase 4 — cov_gemm parity."""

    def test_cov_gemm_matches_triton(self):
        _seeded()
        N, D = 4096, 128
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.linalg.cov_gemm.tilelang import cov_gemm as tilelang_cov_gemm
        from flashlib.linalg.cov_gemm.triton import cov_gemm as triton_cov_gemm

        out_tl = tilelang_cov_gemm(X)
        out_tr = triton_cov_gemm(X)

        torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)


class TestGramGEMMTileLang:
    """Phase 4 — gram_gemm parity."""

    def test_gram_gemm_matches_triton(self):
        _seeded()
        N, D = 256, 4096
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.linalg.gram_gemm.tilelang import gram_gemm as tilelang_gram_gemm
        from flashlib.linalg.gram_gemm.triton import gram_gemm as triton_gram_gemm

        out_tl = tilelang_gram_gemm(X)
        out_tr = triton_gram_gemm(X)

        torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)


class TestAbGEMMTileLang:
    """Phase 4 — ab_gemm parity."""

    def test_ab_gemm_matches_triton(self):
        _seeded()
        N, D1, D2 = 4096, 128, 64
        A = torch.randn(N, D1, device=DEVICE, dtype=torch.float32)
        B = torch.randn(N, D2, device=DEVICE, dtype=torch.float32)

        from flashlib.linalg.ab_gemm.tilelang import ab_gemm as tilelang_ab_gemm
        from flashlib.linalg.ab_gemm.triton import ab_gemm as triton_ab_gemm

        out_tl = tilelang_ab_gemm(A, B)
        out_tr = triton_ab_gemm(A, B)

        torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)


# ---------------------------------------------------------------------------
# Phase 5: Remaining primitives
# ---------------------------------------------------------------------------


class TestStandardScalerTileLang:
    """Phase 5 — StandardScaler parity."""

    def test_fit_transform_matches_triton(self):
        _seeded()
        N, D = 2048, 128
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.standard_scaler.impl import (
            flash_standard_scaler_fit_transform,
        )
        Y_tl, (mean_tl, std_tl) = flash_standard_scaler_fit_transform(
            X, backend="tilelang",
        )
        Y_tr, (mean_tr, std_tr) = flash_standard_scaler_fit_transform(
            X, backend="triton",
        )

        torch.testing.assert_close(Y_tl, Y_tr, rtol=1e-3, atol=1e-3)


class TestMultinomialNBTileLang:
    """Phase 5 — MultinomialNB stub (delegates to Triton)."""

    def test_tilelang_delegates_to_triton(self):
        _seeded()
        N_train, D, N_test = 512, 64, 128
        n_classes = 3
        X_train = torch.abs(torch.randn(N_train, D, device=DEVICE))
        y_train = torch.randint(0, n_classes, (N_train,), device=DEVICE)
        X_test = torch.abs(torch.randn(N_test, D, device=DEVICE))

        from flashlib.primitives.multinomial_nb.impl import flash_multinomial_nb
        labels_tl = flash_multinomial_nb(
            X_train, y_train, X_test, n_classes, backend="tilelang",
        )
        labels_tr = flash_multinomial_nb(
            X_train, y_train, X_test, n_classes, backend="triton",
        )

        match_rate = (labels_tl == labels_tr).float().mean().item()
        assert match_rate > 0.9


class TestLinearRegressionTileLang:
    """Phase 5 — LinearRegression parity."""

    def test_matches_triton(self):
        _seeded()
        N, D = 1024, 32
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        w_true = torch.randn(D, device=DEVICE)
        y = X @ w_true + 0.01 * torch.randn(N, device=DEVICE)

        from flashlib.primitives.linear_regression.impl import flash_linear_regression
        w_tl, b_tl = flash_linear_regression(X, y, backend="tilelang")
        w_tr, b_tr = flash_linear_regression(X, y, backend="triton")

        torch.testing.assert_close(w_tl, w_tr, rtol=1e-2, atol=1e-1)


class TestRidgeTileLang:
    """Phase 5 — Ridge parity."""

    def test_matches_triton(self):
        _seeded()
        N, D = 1024, 32
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        w_true = torch.randn(D, device=DEVICE)
        y = X @ w_true + 0.01 * torch.randn(N, device=DEVICE)

        from flashlib.primitives.ridge.impl import flash_ridge_regression
        w_tl, b_tl = flash_ridge_regression(X, y, alpha=1.0, backend="tilelang")
        w_tr, b_tr = flash_ridge_regression(X, y, alpha=1.0, backend="triton")

        torch.testing.assert_close(w_tl, w_tr, rtol=1e-2, atol=1e-1)


class TestLogisticRegressionTileLang:
    """Phase 5 — LogisticRegression parity."""

    def test_matches_triton(self):
        _seeded()
        N, D = 1024, 32
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        y = (X[:, 0] > 0).to(torch.float32)

        from flashlib.primitives.logistic_regression.impl import (
            flash_logistic_regression,
        )
        w_tl, b_tl = flash_logistic_regression(X, y, backend="tilelang")
        w_tr, b_tr = flash_logistic_regression(X, y, backend="triton")

        # Weights should be correlated (same sign, similar magnitude)
        cos_sim = torch.nn.functional.cosine_similarity(
            w_tl.unsqueeze(0), w_tr.unsqueeze(0),
        ).item()
        assert cos_sim > 0.9


class TestDBSCANTileLang:
    """Phase 5 — DBSCAN parity."""

    def test_labels_match_triton(self):
        _seeded()
        N, D = 512, 16
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.dbscan.impl import flash_dbscan
        labels_tl = flash_dbscan(X, eps=2.0, min_samples=5, backend="tilelang")
        labels_tr = flash_dbscan(X, eps=2.0, min_samples=5, backend="triton")

        # Exact label match is hard (ordering), check n_clusters match
        n_tl = len(set(labels_tl.tolist()) - {-1})
        n_tr = len(set(labels_tr.tolist()) - {-1})
        assert abs(n_tl - n_tr) <= 1


class TestSpectralClusteringTileLang:
    """Phase 5 — SpectralClustering parity."""

    def test_labels_match_triton(self):
        _seeded()
        N, D, K = 256, 16, 3
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.spectral_clustering.impl import (
            flash_spectral_clustering,
        )
        labels_tl = flash_spectral_clustering(
            X, K, n_neighbors=10, seed=42, backend="tilelang",
        )
        labels_tr = flash_spectral_clustering(
            X, K, n_neighbors=10, seed=42, backend="triton",
        )

        # Check same number of clusters found
        n_tl = len(set(labels_tl.tolist()))
        n_tr = len(set(labels_tr.tolist()))
        assert n_tl == n_tr


# ---------------------------------------------------------------------------
# Phase 5 stubs (hdbscan, umap, tsne, random_forest)
# These should delegate to Triton without error when backend="tilelang".
# ---------------------------------------------------------------------------


class TestStubsDelegateToTriton:
    """Stubs should silently fall through to Triton."""

    def test_hdbscan_tilelang_stub(self):
        _seeded()
        N, D = 256, 16
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.hdbscan.impl import flash_hdbscan
        labels = flash_hdbscan(
            X, min_cluster_size=10, min_samples=3, k=8, backend="tilelang",
        )
        assert labels.shape == (N,)

    def test_umap_tilelang_stub(self):
        _seeded()
        N, D = 256, 16
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.umap.impl import flash_umap
        emb = flash_umap(
            X, n_neighbors=10, n_components=2, n_epochs=10, backend="tilelang",
        )
        assert emb.shape == (N, 2)

    def test_tsne_tilelang_stub(self):
        _seeded()
        N, D = 128, 16
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        from flashlib.primitives.tsne.impl import flash_tsne
        emb = flash_tsne(X, n_iter=50, backend="tilelang")
        assert emb.shape == (N, 2)

    def test_random_forest_tilelang_stub(self):
        _seeded()
        N, D = 256, 16
        X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        y = torch.randint(0, 2, (N,), device=DEVICE)

        # flash_random_forest is FlashRandomForestClassifier (no backend arg)
        # Use tilelang_random_forest which wraps it as a module-level function
        from flashlib.primitives.random_forest.tilelang.random_forest import tilelang_random_forest
        rf = tilelang_random_forest(n_estimators=4, max_depth=3)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert preds.shape == (N,)
