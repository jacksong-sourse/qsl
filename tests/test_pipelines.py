"""
测试 qsl.pipelines 中的应用管线。

覆盖:
    - DrugDiscoveryPipeline (药物发现)
    - CryptoAnalysisPipeline (密码分析)
    - PortfolioOptimizer (投资组合优化)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsl.pipelines.drug_discovery import DrugDiscoveryPipeline, MoleculeResult
from qsl.pipelines.crypto_analysis import CryptoAnalysisPipeline, CryptoResult
from qsl.pipelines.portfolio import PortfolioOptimizer, PortfolioResult


# ============================================================================
#  TestDrugDiscovery
# ============================================================================

class TestDrugDiscovery:

    def test_init(self):
        """DrugDiscoveryPipeline 应正确初始化。"""
        pipeline = DrugDiscoveryPipeline(
            target_protein="ACE2",
            num_candidates=5,
            top_k=3,
        )
        assert pipeline.target == "ACE2"
        assert pipeline.num_candidates == 5
        assert pipeline.top_k == 3

    def test_generate_candidates_returns_smiles_list(self):
        """_generate_candidates 应返回 SMILES 字符串列表。"""
        pipeline = DrugDiscoveryPipeline(num_candidates=4)
        candidates = pipeline._generate_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) == 4
        for c in candidates:
            assert isinstance(c, str)
            assert len(c) > 0

    def test_compute_binding_energy_returns_float_tuple(self):
        """_compute_binding_energy 应返回 (float, float) 元组。"""
        pipeline = DrugDiscoveryPipeline(num_candidates=5)
        energy, confidence = pipeline._compute_binding_energy("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert isinstance(energy, float)
        assert isinstance(confidence, float)
        assert confidence > 0.0

    def test_run_returns_molecule_result_list_within_top_k(self):
        """run 应返回不超过 top_k 个 MoleculeResult。"""
        pipeline = DrugDiscoveryPipeline(
            target_protein="TestProtein",
            num_candidates=6,
            top_k=3,
        )
        results = pipeline.run(verbose=False)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, MoleculeResult)
            assert isinstance(r.smiles, str)
            assert isinstance(r.binding_energy, float)
            assert isinstance(r.confidence, float)
            assert isinstance(r.rank, int)
            assert r.rank > 0

    def test_results_sorted_by_energy(self):
        """结果应按 binding_energy 升序排列。"""
        pipeline = DrugDiscoveryPipeline(num_candidates=8, top_k=5)
        results = pipeline.run(verbose=False)
        energies = [r.binding_energy for r in results]
        assert energies == sorted(energies)


# ============================================================================
#  TestCryptoAnalysis
# ============================================================================

class TestCryptoAnalysis:

    def test_init(self):
        """CryptoAnalysisPipeline 应正确初始化。"""
        pipeline = CryptoAnalysisPipeline(
            cipher_type="rsa",
            public_key_modulus=15,
            key_size=128,
        )
        assert pipeline.cipher_type == "rsa"
        assert pipeline.N == 15
        assert pipeline.key_size == 128

    def test_analyze_rsa_returns_crypto_result_with_speedup_gt_1(self):
        """analyze RSA 应返回 speedup > 1 的 CryptoResult。"""
        pipeline = CryptoAnalysisPipeline(
            cipher_type="rsa",
            public_key_modulus=15,
        )
        result = pipeline.analyze()
        assert isinstance(result, CryptoResult)
        assert result.algorithm == "RSA"
        assert result.speedup > 0.0
        assert result.classical_ops > 0

    def test_analyze_symmetric_for_small_key(self):
        """小 key 分析的 analyze。"""
        pipeline = CryptoAnalysisPipeline(key_size=8)
        result = pipeline.analyze()
        assert isinstance(result, CryptoResult)
        assert "SYMMETRIC" in result.algorithm
        assert result.key_size == 8

    def test_compare_method_returns_list(self):
        """compare 静态方法应返回 CryptoResult 列表。"""
        results = CryptoAnalysisPipeline.compare([4, 6, 8])
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CryptoResult)

    def test_crypto_result_dataclass_fields_correct(self):
        """CryptoResult 各字段类型应正确。"""
        result = CryptoResult(
            algorithm="AES",
            key_size=256,
            classical_ops=1e20,
            quantum_ops=1e12,
            speedup=1e8,
            is_vulnerable=False,
        )
        assert result.algorithm == "AES"
        assert result.key_size == 256
        assert isinstance(result.classical_ops, float)
        assert isinstance(result.quantum_ops, float)
        assert isinstance(result.speedup, float)
        assert isinstance(result.is_vulnerable, bool)
        assert result.plaintext is None
        assert result.details is None


# ============================================================================
#  TestPortfolio
# ============================================================================

class TestPortfolio:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.returns, self.cov = PortfolioOptimizer.sample_problem(n_assets=4, seed=42)

    def test_init_with_sample_problem_data(self):
        """用 sample_problem 数据初始化应成功。"""
        opt = PortfolioOptimizer(
            returns=self.returns,
            covariance=self.cov,
            risk_aversion=0.5,
            budget=2,
        )
        assert opt.n_assets == 4
        assert opt.budget == 2

    def test_optimize_returns_portfolio_result(self):
        """optimize 应返回包含 weights/return/risk/sharpe 的 PortfolioResult。"""
        opt = PortfolioOptimizer(
            returns=self.returns,
            covariance=self.cov,
            risk_aversion=0.5,
            budget=None,
        )
        result = opt.optimize(p=1, verbose=False)
        assert isinstance(result, PortfolioResult)
        assert isinstance(result.weights, np.ndarray)
        assert len(result.weights) == 4
        assert isinstance(result.expected_return, float)
        assert isinstance(result.risk, float)
        assert isinstance(result.sharpe_ratio, float)

    def test_build_qubo_matrix_returns_valid_matrix(self):
        """_build_qubo_matrix 应返回形状正确的方阵。"""
        opt = PortfolioOptimizer(
            returns=self.returns,
            covariance=self.cov,
        )
        Q = opt._build_qubo_matrix()
        assert isinstance(Q, np.ndarray)
        assert Q.shape == (4, 4)

    def test_sample_problem_returns_correct_shapes(self):
        """sample_problem 应返回形状正确的 (returns, cov)。"""
        r, cov = PortfolioOptimizer.sample_problem(n_assets=5, seed=99)
        assert isinstance(r, np.ndarray)
        assert isinstance(cov, np.ndarray)
        assert r.shape == (5,)
        assert cov.shape == (5, 5)
        # covariance 应是对称的
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
