<div align="center">

# QSL — Quantum Search Language

**A Qiskit-competitive full-stack quantum computing framework · Chinese & English · AI-powered**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/qsl-quantum?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/qsl-quantum/)
[![License](https://img.shields.io/badge/License-MIT-4CC61E)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-731%2B-00C851)](https://github.com/jacksong-sourse/qsl/actions)
[![Minimal Deps](https://img.shields.io/badge/min%20deps-numpy%20only-important)](#-installation)

[Features](#-features) ·
[Installation](#-installation) ·
[Quick Start](#-quick-start) ·
[Circuit API](#-circuit-api) ·
[Algorithms](#-quantum-algorithms) ·
[AI Scientist](#-ai-quantum-scientist) ·
[Visualization](#-visualization) ·
[CLI](#-command-line) ·
[Qiskit Interop](#-qiskit-interoperability)

</div>

[中文 README](./README.md)

---

## ✨ Features

QSL (Quantum Search Language) is a full-stack quantum computing framework for research and education. The core dependency is **NumPy only** — install and you are ready to go.

- **🔧 Circuit layer (Qiskit-competitive)**
  - `QuantumCircuit` class: gate append/insert/remove, `inverse()`, `compose()`, `decompose()`, `transpile()`
  - Symbolic `Parameter` + `bind_parameters()` / `assign_parameters()` — the foundation for VQE/QAOA/QML
  - Universal controlled operations `gate.control(n)`, gate power `gate.power(k)`, gate inverse `gate.inverse()`
  - Complete gate library: Pauli/H/S/T/SX families, RX/RY/RZ/RXX/RYY/RZZ, CRX/CRY/CRZ/CU/CP, CH/CS/CSdg/CT/CTdg, iSWAP/ECR/DCX/CSWAP/MCMT
- **📚 Standard circuit library**: Bell, GHZ, W state, QFT, QPE, Grover diffusion, quantum teleportation, random circuits, quantum walks
- **💻 High-performance simulation**
  - Vectorized state-vector simulator (up to 26–28 qubits)
  - Density-matrix path with built-in noise model: depolarizing, amplitude damping, phase damping, readout error
  - Optional CuPy GPU acceleration
  - Binary-search sampling, direct Pauli-string expectation values (sampling-free)
- **🔁 Ecosystem interoperability**: OpenQASM 2.0 import/export, QASM 3.0 export; `to_qiskit()` / `from_qiskit()` / `to_cirq()` two-way conversion
- **📐 Visualization**: publication-ready matplotlib circuit diagrams, Bloch sphere, state city plot, Q-sphere, amplitude bar charts, histograms (with correct-answer highlighting for Grover demos)
- **🧮 Core algorithms**: QFT, Shor's factorization, Grover (BBHT unknown-M search, Boolean-circuit oracles), QAOA (Max-Cut and other combinatorial problems), VQE (variational quantum eigensolver)
- **🤖 AI Quantum Scientist**
  - `LLMProvider` abstraction: OpenAI / DeepSeek / Kimi / Qwen / Ollama — configure once, switch globally; defaults to DeepSeek/Kimi for users in China
  - Natural-language problem → automatic algorithm selection, parameter extraction, circuit compilation, execution, verification, explanation (in Chinese or English)
  - Auto-verifier: Shor result cross-multiplication, SAT solution substitution, QAOA vs. classical baseline, Grover solution check; automatic re-planning on failure
  - 10 built-in Chinese demo templates (no API key needed): factorization, 3-SAT, Sudoku, Max-Cut, TSP, graph coloring, Grover search, GHZ preparation, QRNG, BB84
- **⚙️ Engineering**: 731+ pytest tests, GitHub Actions CI (Python 3.9–3.12 matrix), ruff linting, wheel build verification

---

## 📦 Installation

**Minimal install (NumPy only, import in under 5 seconds):**

```bash
pip install qsl-quantum
```

**Optional extras by use case:**

```bash
pip install "qsl-quantum[viz]"          # matplotlib visualization
pip install "qsl-quantum[algorithms]"   # scipy (needed for QAOA/VQE/Shor)
pip install "qsl-quantum[qml]"          # torch + scikit-learn (quantum machine learning)
pip install "qsl-quantum[cross]"        # qiskit + cirq (converters / cross-validation)
pip install "qsl-quantum[ai]"           # openai + langchain (AI scientist)
pip install "qsl-quantum[full]"         # all optional dependencies
pip install "qsl-quantum[dev]"          # development & testing tools
```

**Verify the install:**

```bash
python -c "import qsl; print(qsl.__version__)"
# 0.6.1
python -m qsl --version
```

---

## 🚀 Quick Start

### 1. Your first quantum circuit: the Bell state

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)       # Hadamard superposition
qc.cx(0, 1)   # CNOT entanglement

# Execute and inspect the result
res = qc.execute(shots=1024)
print(res.counts)           # → {0: ~512, 3: ~512}
print(res.statevector())    # → [0.707+0j 0 0 0.707+0j]
res.state.pretty_print()    # → 0.7071|00⟩ + 0.7071|11⟩
```

### 2. Grover search (solving SAT)

```python
from qsl import solve_sat

# Solve 3-SAT: (x0 ∨ ¬x1) ∧ (x1 ∨ x2) ∧ (¬x0 ∨ ¬x2)
# CNF format: each clause is a list of literals (1-indexed variables, negative = negation)
result = solve_sat(
    cnf_clauses=[[1, -2], [2, 3], [-1, -3]],
    n_qubits=3,
    shots=10
)
print(result.get_solutions())   # → list of satisfying assignments
```

### 3. Shor's factoring algorithm

```python
from qsl.algorithms import ShorSolver

factors = ShorSolver(15).factor()   # → [3, 5]
print(f"15 = {factors[0]} × {factors[1]}")
```

### 4. Parameterized circuits (the foundation of QAOA/VQE)

```python
from qsl import QuantumCircuit, Parameter
import numpy as np

theta = Parameter("θ")
qc = QuantumCircuit(2)
qc.h(0); qc.h(1)
qc.rzz(theta, 0, 1)     # Note: angle first, qubits second (consistent with Qiskit)
qc.rx(0.3, 0); qc.rx(0.3, 1)

bound = qc.assign({"θ": np.pi/2})   # bind parameters
counts = bound.measure_all(shots=1000)
```

### 5. AI Quantum Scientist (natural language, Chinese-first but works in English)

```python
from qsl import QuantumAgent, create_provider, set_default_provider

# Configure LLM (optional; without a key the rule-based fallback kicks in)
# Auto-detects DEEPSEEK_API_KEY / MOONSHOT_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY
provider = create_provider()
if provider is not None:
    set_default_provider(provider)

agent = QuantumAgent(verbose=True)
report = agent.run("Factorize 15 into prime factors")
# → auto-selects Shor, builds the circuit, executes, verifies (3×5=15), returns a structured report
print(report.to_markdown())
```

Without any LLM key, a rule-based router with Chinese parameter extraction handles common tasks out of the box.

---

## 🧩 Circuit API

### Building circuits

```python
from qsl import QuantumCircuit, Parameter

qc = QuantumCircuit(3, name="demo", global_phase=0.0)

# Single-qubit gates
qc.h(0); qc.x(1); qc.y(2); qc.z(0)
qc.s(0); qc.t(1); qc.sx(2)          # S / T / √X
qc.sdg(0); qc.tdg(1); qc.sxdg(2)    # conjugate transpose

# Parameterized gates (angle / control / target — aligned with Qiskit)
qc.rx(0.5, 0); qc.ry(0.5, 1); qc.rz(0.5, 2)
qc.p(0.3, 0); qc.u(1.0, 0.2, 0.3, 0)    # θ, φ, λ

# Two-qubit gates
qc.cx(0, 1); qc.cy(0, 1); qc.cz(0, 1)
qc.ch(0, 1); qc.cs(0, 1); qc.ct(0, 1)
qc.crx(0.7, 0, 1); qc.cry(0.7, 0, 1); qc.crz(0.7, 0, 1)
qc.cp(0.5, 0, 1); qc.cu(0.3, 0.4, 0.5, 0, 1)   # θ, φ, λ, c, t [, γ]
qc.swap(0, 1); qc.iswap(0, 1); qc.ecr(0, 1); qc.dcx(0, 1)
qc.rxx(0.5, 0, 1); qc.ryy(0.5, 0, 1); qc.rzz(0.5, 0, 1)

# Three-qubit / multi-qubit
qc.ccx(0, 1, 2)             # Toffoli
qc.cswap(0, 1, 2)           # Fredkin
qc.mcx([0,1], 2)            # multi-controlled X
qc.mcz([0,1,2])             # multi-controlled Z
qc.barrier()
```

### Circuit transformations

```python
qc_inv  = qc.inverse()           # circuit inverse
qc2     = qc.compose(qc_inv)     # circuit composition
qc_dec  = qc.decompose()         # decompose into basis gate set
qc_t    = qc.transpile(optimization_level=2)  # compile & optimize
qc_rev  = qc.reverse_bits()      # reverse bit ordering

# Gate-level transforms
from qsl.quantum_gates import H
cc_h = H.control(2)              # doubly-controlled H gate
h2   = H.power(0.5)              # √H
hdag = H.inverse()               # H†
```

### Execution and measurement

```python
# State-vector simulation
res = qc.execute(shots=1024, seed=42)
counts = res.counts                  # dict[int, int]
sv     = res.statevector()           # complex state vector
probs  = res.probabilities_dict()    # {bitstring: prob}

# Convenience
counts = qc.measure_all(shots=1000)

# Density-matrix + noise simulation
from qsl import NoiseModel
noise = NoiseModel(
    depolarizing=0.01,       # 1% depolarizing
    amplitude_damping=0.005, # T1 amplitude damping
    phase_damping=0.01,      # T2 phase damping
    readout_error=0.02       # 2% readout bit-flip
)
res_noisy = qc.execute_density(shots=1024, noise=noise)

# Analytic expectation values (sampling-free)
ev_z  = qc.expectation("IZ")                # ⟨Z1⟩ (position 0 = Pauli I, position 1 = Z; left = low qubit)
ev_zz = qc.expectation("ZZ")                # ⟨Z0 Z1⟩
ev_xx_zz = qc.expectation([(0.5, "ZZ"), (-0.3, "XX")])
```

### Import / Export

```python
from qsl import (QuantumCircuit,
    dumps_qasm2, loads_qasm2, dumps_qasm3,
    to_qiskit, from_qiskit, to_cirq)

# QASM interoperability
qasm_str = dumps_qasm2(qc)
qc2 = loads_qasm2(qasm_str)
print(dumps_qasm3(qc))

# Qiskit / Cirq two-way conversion
qk = to_qiskit(qc)
qc_back = from_qiskit(qk)
cq = to_cirq(qc)
```

### Standard circuit library

```python
from qsl.circuit import library

qc_bell   = library.bell_state("phi+")     # |Φ+⟩
qc_ghz    = library.ghz_state(4)           # 4-qubit GHZ
qc_w      = library.w_state(4)             # 4-qubit W state
qc_qft    = library.qft(4)                 # 4-qubit QFT
qc_iqft   = library.qft(4, inverse=True)   # inverse QFT
qc_qpe    = library.qpe(U_gate, n_counting=4)  # QPE (pass the unitary whose eigenphase you want)
qc_diff   = library.grover_diffusion(4)    # Grover diffusion operator
qc_tp     = library.teleportation()        # quantum teleportation
qc_rand   = library.random_circuit(5, depth=10, seed=0)
qc_walk   = library.quantum_walk_cycle(8)  # quantum walk on an 8-node cycle
```

---

## 🧮 Quantum Algorithms

### QFT — Quantum Fourier Transform

```python
from qsl.circuit import library

qc = library.qft(4)     # 4-qubit QFT circuit (a QuantumCircuit object)
print(qc.draw())        # ASCII circuit diagram
res = qc.execute()
```

### QAOA — Quantum Approximate Optimization (Max-Cut example)

```python
from qsl.algorithms import QAOA
import numpy as np

# Adjacency matrix for a 4-node cycle Max-Cut
adj = np.array([
    [0,1,0,1],
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0],
], dtype=float)
cost = QAOA.maxcut_cost_matrix(adj)

qaoa = QAOA(n_qubits=4, cost_matrix=cost, p=2)
qaoa.optimize(maxiter=200)
print("Best cut value:", qaoa.optimal_energy)
print("Best bitstring:", qaoa.optimal_bitstring_str)  # e.g. "0101"
```

### VQE — Variational Quantum Eigensolver

```python
from qsl.algorithms import VQE

# Find ground-state energy of H = 0.5·Z1 − 0.2·X0·X1
# (Pauli string length must equal n_qubits)
vqe = VQE(
    n_qubits=2,
    hamiltonian_pauli_terms=[(0.5, "IZ"), (-0.2, "XX")],
    n_layers=2,
)
vqe.optimize(maxiter=200)
print("Ground-state energy:", vqe.ground_energy)
```

---

## 🤖 AI Quantum Scientist

QSL ships with a Chinese-first AI quantum scientist that drives quantum computing via natural language. Without any LLM key it falls back to a rule-based router (with Chinese parameter extraction); with a key it calls an LLM for complex tasks.

```python
from qsl import QuantumAgent, create_provider, set_default_provider

# Configure an LLM (optional — without a key, rule-based fallback is used)
# Auto-detects env vars: DEEPSEEK_API_KEY / MOONSHOT_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY
provider = create_provider()
if provider is not None:
    set_default_provider(provider)

agent = QuantumAgent(verbose=True)
report = agent.run("Factorize 15 into prime factors")
# → auto-selects Shor, builds the circuit, executes, verifies (3×5=15), returns a structured report
print(report.to_markdown())
```

**10 built-in Chinese demos (no API key required):**

```bash
python -m qsl --list-demos       # list demos
python -m qsl --ai-demo 1        # run demo #1
```

From code:

```python
from qsl import run_demo, list_demos
for d in list_demos():
    print(d['id'], d['name'], d['desc'])
report = run_demo(1, verbose=True)
print(report.to_markdown())
```

**Configuring an LLM (DeepSeek / Kimi recommended for users in China):**

```bash
# Pick one
export DEEPSEEK_API_KEY="sk-..."
export MOONSHOT_API_KEY="sk-..."      # Kimi
export OPENAI_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."     # Qwen (DashScope)

# Optional: explicitly select provider/model
export QSL_LLM=deepseek               # deepseek / kimi / openai / qwen / ollama
export QSL_LLM_MODEL=deepseek-chat
```

---

## 📊 Visualization

Requires `pip install "qsl-quantum[viz]"`.

```python
import matplotlib.pyplot as plt
from qsl import QuantumCircuit
from qsl import plot_histogram, plot_bloch_sphere, plot_state_city

qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)
res = qc.execute(shots=4096)

# 1. Publication-quality circuit diagram (matplotlib)
fig, ax = qc.draw(output="mpl", style="iqp")

# 2. Measurement histogram (supports marking correct solutions — essential for Grover demos)
plot_histogram(res.counts, title="Bell-state measurement")

# 3. State visualization
# plot_bloch_sphere(state)        # single-qubit Bloch sphere
# plot_state_city(density_matrix) # 3D city plot of a density matrix
# plot_amplitudes(sv)             # amplitude bar chart
# plot_qsphere(sv)                # Q-sphere
plt.show()
```

---

## 💻 Command Line

```bash
python -m qsl                      # interactive start
python -m qsl --version            # version
python -m qsl --help               # help
python -m qsl --demo               # list and run Grover demos
python -m qsl --demo 1             # directly run Grover demo #1
python -m qsl --solve 3 "x0|~x1" "x1|x2" "~x0|~x2"   # CLI SAT solving
python -m qsl --file test.qsl      # run a .qsl DSL file
python -m qsl --list-demos         # list the 10 Chinese AI demos
python -m qsl --ai-demo 1          # run a Chinese AI demo
```

---

## 🔁 Qiskit Interoperability

QSL's gate parameter ordering and global-phase conventions match Qiskit's (angles first, qubits second), enabling bit-exact numerical comparison.

```python
from qsl import QuantumCircuit, to_qiskit, from_qiskit
import numpy as np

# qsl → qiskit
qc = QuantumCircuit(2); qc.h(0); qc.crx(0.5, 0, 1)
qk = to_qiskit(qc)

# qiskit → qsl
from qiskit.circuit.library import QFT
qc_back = from_qiskit(QFT(4))
```

Cross-validation tests cover every standard gate with per-amplitude error < 1e-10.

---

## 📁 Project Structure

```
qsl/
├── circuit/        # Circuit object model (QuantumCircuit/Gate/Parameter/QASM/converters/viz/library)
├── core/           # State-vector/density-matrix simulators, Grover, oracles, Boolean parser
├── algorithms/     # QFT, Shor, QAOA, VQE
├── qml/            # QuantumLayer, QNN, quantum kernels, QSVM, QGAN
├── backends/       # Local simulator, IBM/AWS Braket hardware backends
├── compiler/       # DSL parser, compiler, circuit optimizer, layout mapping, error mitigation
├── viz/            # matplotlib visualization (circuit / Bloch sphere / city plot / histograms)
├── ai/             # LLMProvider, NLP translator, agent, auto-verifier, Chinese demos, explainer
├── meta/           # Algorithm search, AI compiler, theorem conjecturing
├── network/        # Distributed nodes, quantum blockchain (demonstration)
├── pipelines/      # Drug discovery / cryptanalysis / portfolio application examples
└── utils/          # Exceptions and parameter validation
```

---

## 🛠️ Development

```bash
git clone https://github.com/jacksong-sourse/qsl.git
cd qsl
pip install -e ".[dev,viz,algorithms]"

pytest                                # run all 731+ tests
pytest --cov=qsl --cov-report=term    # coverage
ruff check qsl                        # lint
```

---

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) (follows [Keep a Changelog](https://keepachangelog.com/)).

- **v0.6.1** (2026-07-19): parametric-gate ordering fix, CLI enhancements, Qiskit-compatibility APIs, Dirac-notation printing, BBHT restart, removed obsolete setup.py
- **v0.6.0** (2026-07-19): circuit layer, QASM, converters, visualization, noise simulation, LLMProvider, auto-verifier, Chinese demos

---

## 📄 License

MIT License © 2026 Song Ziming

---

## 🙋 FAQ

**Q: What are the minimal dependencies?**
A: NumPy only. `pip install qsl-quantum` gives you the simulator, Grover, Shor (the large-integer parts of Shor/QAOA/VQE need SciPy via `[algorithms]`), and more.

**Q: How is this related to Qiskit?**
A: QSL is an independently implemented quantum computing framework with a Qiskit-like API to minimize migration cost; it interoperates with Qiskit via `to_qiskit()`/`from_qiskit()`, so you can mix the two.

**Q: What is the simulation qubit limit?**
A: The vectorized state-vector path can simulate up to 26–28 qubits on a typical laptop (memory-bound); the density-matrix path is for noisy simulation on smaller qubit counts.

**Q: Can I use the AI features without an OpenAI key?**
A: Yes. Without any key, a Chinese rule-based router handles intent recognition and parameter extraction for common tasks (factorization, search, optimization, GHZ preparation, etc.). DeepSeek and Kimi work directly from mainland China.
