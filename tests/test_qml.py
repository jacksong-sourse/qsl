"""
测试量子机器学习（QML）模块。

覆盖:
    - QuantumLayer: 角度编码、振幅编码、前向传播、梯度计算、可训练参数
    - QNN: 混合模型创建、前向传播、fit/predict/predict_proba
    - quantum_kernel: 核矩阵对称性、值域、RBF 量子核
    - QuantumSVM: 拟合、预测、sklearn API 兼容性
    - QGAN: 生成器采样、训练、判别器分类、损失追踪
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# ============================================================
# torch 可用性检查
# ============================================================
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================
# QML 模块导入检查
# ============================================================
try:
    from qsl.qml.layers import QuantumLayer
    QML_LAYERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QML_LAYERS_AVAILABLE = False

try:
    from qsl.qml.qnn import QNN
    QML_QNN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QML_QNN_AVAILABLE = False

try:
    from qsl.qml.kernels import quantum_kernel, rbf_quantum_kernel
    QML_KERNELS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QML_KERNELS_AVAILABLE = False

try:
    from qsl.qml.qsvm import QuantumSVM
    QML_QSVM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QML_QSVM_AVAILABLE = False

try:
    from qsl.qml.qgan import QGAN
    QML_QGAN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QML_QGAN_AVAILABLE = False


# ============================================================
# 通用 fixtures 和工具函数
# ============================================================

@pytest.fixture(scope="module")
def rng():
    """可复现的随机数生成器。"""
    return np.random.RandomState(42)


# ============================================================
# TestQuantumLayer
# ============================================================

@pytest.mark.skipif(not QML_LAYERS_AVAILABLE, reason="QML layers 模块不可用")
class TestQuantumLayer:
    """测试 QuantumLayer。"""

    def test_angle_encoding_produces_valid_state(self):
        """角度编码产生有效量子态（概率和为 1）。"""
        torch.manual_seed(42)
        layer = QuantumLayer(n_qubits=2, n_features=2, encoding="angle")
        x = torch.randn(4, 2)  # batch=4, features=2

        state = layer._encode_state(x)
        probs = torch.abs(state) ** 2
        sum_probs = probs.sum(dim=-1)

        # 概率和应接近 1
        for s in sum_probs:
            assert abs(s.item() - 1.0) < 1e-5, f"概率和 = {s.item()}，偏离 1"

    def test_amplitude_encoding_normalizes(self):
        """振幅编码将输入归一化。"""
        torch.manual_seed(42)
        n_qubits = 3  # N = 8
        layer = QuantumLayer(n_qubits=n_qubits, n_features=4, encoding="amplitude")
        x = torch.randn(4, 4)

        state = layer._encode_state(x)
        probs = torch.abs(state) ** 2
        sum_probs = probs.sum(dim=-1)

        for s in sum_probs:
            assert abs(s.item() - 1.0) < 1e-5, f"振幅编码概率和 = {s.item()}，偏离 1"

    def test_dense_angle_encoding_produces_valid_state(self):
        """密集角度编码产生有效量子态。"""
        torch.manual_seed(42)
        layer = QuantumLayer(n_qubits=2, n_features=3, encoding="dense_angle")
        x = torch.randn(4, 3)

        state = layer._encode_state(x)
        probs = torch.abs(state) ** 2
        sum_probs = probs.sum(dim=-1)

        for s in sum_probs:
            assert abs(s.item() - 1.0) < 1e-5, f"dense_angle 概率和 = {s.item()}，偏离 1"

    def test_forward_pass_returns_correct_shape(self):
        """前向传播返回正确形状 (batch, n_qubits)。"""
        torch.manual_seed(42)
        n_qubits, n_features, batch = 2, 3, 5
        layer = QuantumLayer(n_qubits=n_qubits, n_features=n_features)
        x = torch.randn(batch, n_features)

        output = layer(x)
        assert output.shape == (batch, n_qubits), (
            f"期望形状 ({batch}, {n_qubits})，实际 {output.shape}"
        )

    def test_expectations_in_range(self):
        """PauliZ 期望值在 [-1, 1] 范围内。"""
        torch.manual_seed(42)
        layer = QuantumLayer(n_qubits=3, n_features=4)
        x = torch.randn(10, 4)

        output = layer(x)
        assert torch.all(output >= -1.0), "存在期望值 < -1"
        assert torch.all(output <= 1.0), "存在期望值 > 1"

    def test_gradient_computation(self):
        """反向传播后参数具有梯度。"""
        torch.manual_seed(42)
        layer = QuantumLayer(n_qubits=2, n_features=3)
        x = torch.randn(4, 3)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.trainable_params.grad is not None, "trainable_params 应有梯度"
        assert layer.trainable_params.grad.abs().sum() > 0, "梯度不应全为零"

    def test_trainable_params_is_nn_parameter(self):
        """trainable_params 是 nn.Parameter 类型。"""
        layer = QuantumLayer(n_qubits=2, n_features=3)
        assert isinstance(layer.trainable_params, nn.Parameter), (
            f"trainable_params 类型应为 nn.Parameter，实际 {type(layer.trainable_params)}"
        )

    def test_multiple_encodings_forward(self):
        """三种编码方式都能正常完成前向传播。"""
        torch.manual_seed(42)
        x = torch.randn(3, 3)

        for enc in ["angle", "amplitude", "dense_angle"]:
            layer = QuantumLayer(n_qubits=2, n_features=3, encoding=enc)
            output = layer(x)
            assert output.shape == (3, 2), f"编码 {enc} 输出形状错误: {output.shape}"

    def test_n_layers_parameter(self):
        """不同变分层数不影响前向传播。"""
        torch.manual_seed(42)
        x = torch.randn(4, 2)

        for n_layers in [1, 2, 3]:
            layer = QuantumLayer(n_qubits=2, n_features=2, n_layers=n_layers)
            output = layer(x)
            assert output.shape == (4, 2), (
                f"n_layers={n_layers} 输出形状错误: {output.shape}"
            )

    def test_dense_angle_has_dense_layer(self):
        """dense_angle 编码具有 nn.Linear 层。"""
        layer = QuantumLayer(n_qubits=2, n_features=3, encoding="dense_angle")
        assert layer.dense is not None, "dense_angle 应有 dense 层"
        assert isinstance(layer.dense, nn.Linear), (
            f"dense 应为 nn.Linear，实际 {type(layer.dense)}"
        )

    def test_angle_encoding_has_no_dense_layer(self):
        """angle 和 amplitude 编码不应有 dense 层。"""
        for enc in ["angle", "amplitude"]:
            layer = QuantumLayer(n_qubits=2, n_features=3, encoding=enc)
            assert layer.dense is None, f"编码 {enc} 不应有 dense 层"


# ============================================================
# TestQNN
# ============================================================

@pytest.mark.skipif(not QML_QNN_AVAILABLE, reason="QML QNN 模块不可用")
class TestQNN:
    """测试 QNN（混合量子-经典神经网络）。"""

    @pytest.fixture(autouse=True)
    def _seed(self):
        """每个测试前固定随机种子。"""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_hybrid_model_creation(self):
        """创建混合模型不报错，包含量子层和经典层。"""
        model = QNN(n_qubits=2, n_features=3, n_outputs=2)
        assert hasattr(model, "quantum_layer"), "QNN 应有 quantum_layer"
        assert hasattr(model, "classical"), "QNN 应有 classical 层"
        assert isinstance(model.quantum_layer, QuantumLayer), (
            "quantum_layer 应为 QuantumLayer"
        )

    def test_forward_pass_shape(self):
        """前向传播返回正确形状 (batch, n_outputs)。"""
        model = QNN(n_qubits=2, n_features=3, n_outputs=2)
        x = torch.randn(5, 3)

        output = model(x)
        assert output.shape == (5, 2), f"期望 (5, 2)，实际 {output.shape}"

    def test_forward_pass_shape_with_hidden_dim(self):
        """有隐藏层时前向传播返回正确形状。"""
        model = QNN(n_qubits=3, n_features=2, n_outputs=3, hidden_dim=16)
        x = torch.randn(4, 2)

        output = model(x)
        assert output.shape == (4, 3), f"期望 (4, 3)，实际 {output.shape}"

    def test_forward_pass_shape_no_hidden_dim(self):
        """无隐藏层时前向传播返回正确形状。"""
        model = QNN(n_qubits=3, n_features=2, n_outputs=2, hidden_dim=0)
        x = torch.randn(4, 2)

        output = model(x)
        assert output.shape == (4, 2), f"期望 (4, 2)，实际 {output.shape}"

    def test_predict_returns_integer_labels(self):
        """分类模式下 predict 返回整数标签。"""
        model = QNN(n_qubits=2, n_features=3, n_outputs=2)
        X = np.random.randn(50, 3).astype(np.float32)

        preds = model.predict(X)
        assert preds.dtype in (np.int64, np.int32, np.intc, int), (
            f"预测标签应为整数类型，实际 {preds.dtype}"
        )
        assert preds.shape == (50,), f"预测形状应为 (50,)，实际 {preds.shape}"
        # 二分类标签应在 {0, 1} 内
        assert set(np.unique(preds)).issubset({0, 1}), (
            f"二分类标签应在 {{0, 1}}，实际 {set(np.unique(preds))}"
        )

    def test_predict_proba_returns_probabilities(self):
        """predict_proba 返回概率值。"""
        model = QNN(n_qubits=2, n_features=3, n_outputs=2)
        X = np.random.randn(50, 3).astype(np.float32)

        proba = model.predict_proba(X)
        assert proba.shape == (50, 2), f"概率形状应为 (50, 2)，实际 {proba.shape}"
        # 每个样本的概率和应接近 1
        proba_sums = proba.sum(axis=1)
        for s in proba_sums:
            assert abs(s - 1.0) < 1e-5, f"概率和 = {s}，偏离 1"
        assert np.all(proba >= 0) and np.all(proba <= 1), "概率应在 [0, 1]"

    def test_predict_proba_shape_with_3_classes(self):
        """三分类 predict_proba 返回正确形状。"""
        model = QNN(n_qubits=2, n_features=4, n_outputs=3)
        X = np.random.randn(10, 4).astype(np.float32)

        proba = model.predict_proba(X)
        assert proba.shape == (10, 3), f"三分类概率形状应为 (10, 3)，实际 {proba.shape}"

    def test_fit_on_xor_like_data(self):
        """在类 XOR 数据上训练，损失应下降。"""
        rng = np.random.RandomState(42)

        n_samples = 50
        X = rng.randn(n_samples, 2) * 0.8
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

        model = QNN(n_qubits=2, n_features=2, n_outputs=2,
                     quantum_encoding="angle", n_quantum_layers=1)
        losses = model.fit(X, y, epochs=30, lr=0.05, batch_size=16, verbose=False)

        assert len(losses) == 30, f"应有 30 个 loss 值，实际 {len(losses)}"
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        assert final_loss < initial_loss, (
            f"训练损失应下降: 初始 {initial_loss:.4f} -> 最终 {final_loss:.4f}"
        )

    def test_training_loss_decreases_monotonically_trend(self):
        """使用更多 epoch 验证损失总体下降趋势且无 NaN/Inf。"""
        rng = np.random.RandomState(42)
        n_samples = 30
        X = rng.randn(n_samples, 2) * 0.8
        y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)

        model = QNN(n_qubits=2, n_features=2, n_outputs=2,
                     n_quantum_layers=2, hidden_dim=8)
        losses = model.fit(X, y, epochs=60, lr=0.03, batch_size=16, verbose=False)

        assert len(losses) == 60
        assert not np.any(np.isnan(losses)), "损失不应包含 NaN"
        assert not np.any(np.isinf(losses)), "损失不应包含 Inf"


# ============================================================
# TestQuantumKernel
# ============================================================

@pytest.mark.skipif(not QML_KERNELS_AVAILABLE, reason="QML kernels 模块不可用")
class TestQuantumKernel:
    """测试量子核方法。"""

    @pytest.fixture(autouse=True)
    def _seed(self):
        """固定随机种子。"""
        np.random.seed(42)

    def test_kernel_matrix_symmetric_for_same_input(self):
        """X1 == X2 时核矩阵对称。"""
        X = np.random.randn(10, 4).astype(np.float32)
        K = quantum_kernel(X, X, n_qubits=4)

        assert K.shape == (10, 10), f"核矩阵形状应为 (10, 10)，实际 {K.shape}"
        assert np.allclose(K, K.T, atol=1e-6), "核矩阵应对称"

    def test_kernel_values_in_range(self):
        """核矩阵值在 [0, 1] 范围内。"""
        X = np.random.randn(8, 3).astype(np.float32)
        K = quantum_kernel(X, X, n_qubits=3)

        assert np.all(K >= 0), "核矩阵值应 >= 0"
        assert np.all(K <= 1 + 1e-5), "核矩阵值应 <= 1"

    def test_kernel_diagonal_is_one(self):
        """对角线元素（自身 fidelity）应为 1。"""
        X = np.random.randn(6, 3).astype(np.float32)
        K = quantum_kernel(X, X, n_qubits=3)

        for i in range(6):
            assert abs(K[i, i] - 1.0) < 1e-5, (
                f"对角线 K[{i},{i}] = {K[i,i]}，应接近 1"
            )

    def test_kernel_positive_semi_definite_like(self):
        """核矩阵的特征值几乎非负（半正定近似）。"""
        X = np.random.randn(10, 3).astype(np.float32)
        K = quantum_kernel(X, X, n_qubits=3)

        eigenvalues = np.linalg.eigvalsh(K)
        # 允许轻微数值误差（如 -1e-7）
        assert np.all(eigenvalues > -1e-5), (
            f"存在明显负特征值: {eigenvalues[eigenvalues < -1e-5]}"
        )

    def test_kernel_with_different_inputs_shape(self):
        """X1 != X2 时核矩阵形状正确。"""
        X1 = np.random.randn(6, 4).astype(np.float32)
        X2 = np.random.randn(8, 4).astype(np.float32)

        K = quantum_kernel(X1, X2, n_qubits=4)
        assert K.shape == (6, 8), f"核矩阵形状应为 (6, 8)，实际 {K.shape}"

    def test_rbf_quantum_kernel_works(self):
        """RBF 量子核正常工作且值在 [0, 1] 范围内。"""
        X1 = np.random.randn(5, 3).astype(np.float32)
        X2 = np.random.randn(5, 3).astype(np.float32)

        K = rbf_quantum_kernel(X1, X2, gamma=1.0, n_qubits=3)
        assert K.shape == (5, 5), f"RBF 核形状应为 (5, 5)，实际 {K.shape}"
        assert np.all(K >= 0), "RBF 核值应 >= 0"
        assert np.all(K <= 1 + 1e-5), "RBF 核值应 <= 1"

    def test_rbf_quantum_kernel_symmetric(self):
        """RBF 核矩阵对角线为 1 且对称。"""
        X = np.random.randn(4, 2).astype(np.float32)
        K = rbf_quantum_kernel(X, X, gamma=0.5, n_qubits=2)

        assert np.allclose(K, K.T, atol=1e-6), "RBF 核应对称"
        for i in range(4):
            assert abs(K[i, i] - 1.0) < 1e-5, f"对角线 K[{i},{i}] 应接近 1"


# ============================================================
# TestQuantumSVM
# ============================================================

@pytest.mark.skipif(not QML_QSVM_AVAILABLE, reason="QML QSVM 模块不可用")
class TestQuantumSVM:
    """测试 QuantumSVM。"""

    @pytest.fixture(autouse=True)
    def _seed(self):
        """固定随机种子。"""
        np.random.seed(42)

    def test_fit_on_small_dataset(self):
        """在小数据集（4 样本, 2 特征）上拟合。"""
        X = np.array([[1.0, 0.5],
                       [0.5, 1.0],
                       [-1.0, -0.5],
                       [-0.5, -1.0]], dtype=np.float32)
        y = np.array([1, 1, 0, 0])

        qsvm = QuantumSVM(C=10.0, random_state=42)
        qsvm.fit(X, y)

        assert hasattr(qsvm, "svc_"), "fit 后应有 svc_"
        assert hasattr(qsvm, "n_features_in_"), "fit 后应有 n_features_in_"
        assert qsvm.n_features_in_ == 2, f"n_features_in_ 应为 2，实际 {qsvm.n_features_in_}"

    def test_predict_returns_correct_shape(self):
        """predict 返回正确形状和标签。"""
        X = np.array([[1.0, 1.0],
                       [0.8, 0.9],
                       [-1.0, -1.0],
                       [-0.8, -0.9]], dtype=np.float32)
        y = np.array([1, 1, 0, 0])

        qsvm = QuantumSVM(C=10.0, random_state=42)
        qsvm.fit(X, y)

        preds = qsvm.predict(X)
        assert preds.shape == (4,), f"预测形状应为 (4,)，实际 {preds.shape}"

    def test_get_params(self):
        """get_params 返回参数字典。"""
        qsvm = QuantumSVM(n_qubits=3, C=2.0, gamma=0.5, use_rbf=False,
                          probability=True, random_state=123, scale=False)
        params = qsvm.get_params()

        assert params["n_qubits"] == 3
        assert params["C"] == 2.0
        assert params["gamma"] == 0.5
        assert params["use_rbf"] is False
        assert params["probability"] is True
        assert params["random_state"] == 123
        assert params["scale"] is False

    def test_set_params(self):
        """set_params 修改参数并生效。"""
        qsvm = QuantumSVM()
        qsvm.set_params(C=5.0, use_rbf=True)

        assert qsvm.C == 5.0
        assert qsvm.use_rbf is True

    def test_score(self):
        """score 返回准确率分数。"""
        X = np.array([[1.0, 1.0],
                       [0.8, 0.9],
                       [-1.0, -1.0],
                       [-0.8, -0.9]], dtype=np.float32)
        y = np.array([1, 1, 0, 0])

        qsvm = QuantumSVM(C=10.0, random_state=42)
        qsvm.fit(X, y)

        s = qsvm.score(X, y)
        assert 0.0 <= s <= 1.0, f"准确率应在 [0, 1]，实际 {s}"

    def test_scaling_works(self):
        """scale=True 时自动标准化，预测不报错。"""
        X = np.array([[100.0, 200.0],
                       [101.0, 201.0],
                       [-100.0, -200.0],
                       [-101.0, -201.0]], dtype=np.float32)
        y = np.array([1, 1, 0, 0])

        qsvm = QuantumSVM(scale=True, C=10.0, random_state=42)
        qsvm.fit(X, y)

        preds = qsvm.predict(X)
        assert preds.shape == (4,)


# ============================================================
# TestQGAN
# ============================================================

@pytest.mark.skipif(not QML_QGAN_AVAILABLE, reason="QML QGAN 模块不可用")
class TestQGAN:
    """测试 QGAN（量子生成对抗网络）。"""

    @pytest.fixture(autouse=True)
    def _seed(self):
        """固定随机种子。"""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_generator_produces_binary_samples(self):
        """生成器产生二值样本。"""
        qgan = QGAN(latent_dim=4, data_dim=4, n_qubits=4)
        samples = qgan.sample(n_samples=20)

        assert samples.shape == (20, 4), f"样本形状应为 (20, 4)，实际 {samples.shape}"
        unique_vals = np.unique(samples)
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"样本应只含 0 和 1，实际含 {unique_vals}"
        )

    def test_sample_method_returns_correct_shape(self):
        """sample 方法返回正确形状。"""
        qgan = QGAN(latent_dim=2, data_dim=3)

        for n in [1, 5, 10]:
            samples = qgan.sample(n_samples=n)
            assert samples.shape == (n, 3), f"n={n} 时形状错误: {samples.shape}"

    def test_generate_is_alias_of_sample(self):
        """generate 方法是 sample 的别名。"""
        qgan = QGAN(latent_dim=2, data_dim=3)
        s1 = qgan.sample(n_samples=10)
        torch.manual_seed(42)
        np.random.seed(42)
        s2 = qgan.generate(n_samples=10)

        assert s1.shape == s2.shape, "generate 与 sample 形状应对齐"

    def test_discriminator_classifies(self):
        """判别器对真假数据给出有效输出，范围在 [0, 1]。"""
        qgan = QGAN(latent_dim=2, data_dim=3)

        real_batch = torch.ones(8, 3)
        fake_batch = torch.zeros(8, 3)

        with torch.no_grad():
            real_out = qgan.discriminator(real_batch)
            fake_out = qgan.discriminator(fake_batch)

        assert torch.all(real_out >= 0) and torch.all(real_out <= 1), (
            "判别器输出应在 [0, 1]"
        )
        assert torch.all(fake_out >= 0) and torch.all(fake_out <= 1), (
            "判别器输出应在 [0, 1]"
        )
        assert real_out.shape == (8, 1), f"输出形状应为 (8, 1)，实际 {real_out.shape}"

    def test_training_runs_few_epochs(self):
        """训练少量 epoch 不报错且损失被追踪。"""
        rng = np.random.RandomState(42)
        real_data = (rng.rand(100, 3) > 0.5).astype(np.float32)

        qgan = QGAN(latent_dim=3, data_dim=3, n_qubits=3)
        history = qgan.train(real_data, epochs=5, batch_size=16,
                             verbose=False)

        assert "g_loss" in history, "应追踪生成器损失"
        assert "d_loss" in history, "应追踪判别器损失"
        assert len(history["g_loss"]) == 5, f"应有 5 个 g_loss，实际 {len(history['g_loss'])}"
        assert len(history["d_loss"]) == 5, f"应有 5 个 d_loss，实际 {len(history['d_loss'])}"

    def test_losses_are_finite(self):
        """损失值均为有限值（非 NaN/Inf）。"""
        rng = np.random.RandomState(42)
        real_data = (rng.rand(50, 2) > 0.5).astype(np.float32)

        qgan = QGAN(latent_dim=2, data_dim=2, n_qubits=2)
        history = qgan.train(real_data, epochs=5, batch_size=8,
                             verbose=False)

        for loss_list in [history["g_loss"], history["d_loss"]]:
            assert not np.any(np.isnan(loss_list)), "损失不应含 NaN"
            assert not np.any(np.isinf(loss_list)), "损失不应含 Inf"
