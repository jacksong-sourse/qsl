"""
Drug Discovery Pipeline - Quantum-accelerated molecular binding energy calculation.

=====================================================================
*** WARNING: DEMONSTRATION ONLY — NOT FOR REAL RESEARCH USE         ***
***                                                                 ***
*** The molecular Hamiltonians used here are RANDOMLY GENERATED     ***
*** from SMILES string hashes using np.random. They do NOT          ***
*** correspond to any real electronic structure calculations.       ***
***                                                                 ***
*** Real drug discovery requires quantum chemistry packages like    ***
*** OpenFermion + PySCF to compute actual one- and two-electron     ***
*** integrals. This module is a PEDAGOGICAL PIPELINE SKELETON.     ***
=====================================================================

Uses VQE to compute approximate binding energies of candidate drug molecules.
NOTE: The molecular Hamiltonian is generated from SMILES hashing for
demonstration purposes only.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MoleculeResult:
    """Result for a single molecule screening."""
    smiles: str
    binding_energy: float  # in Hartree (demonstration values)
    confidence: float
    rank: int


class DrugDiscoveryPipeline:
    """
    Quantum drug discovery pipeline.

    Workflow:
    1. Input target protein sequence
    2. Generate candidate molecules (from built-in library)
    3. VQE computes approximate binding energy for each
    4. Rank by binding affinity (Top-K)

    Hamiltonian sources (use_real_chemistry):
        - use_real_chemistry=True: 通过 OpenFermion + PySCF (+ RDKit)
          从 SMILES 生成 3D 构象并计算真实的一/二电子积分,
          经 Jordan-Wigner 变换得到 Pauli 哈密顿量。
          需要: pip install openfermion openfermionpyscf pyscf rdkit
        - use_real_chemistry=False (默认): 演示模式, 使用从 SMILES
          哈希生成的随机哈密顿量 (与真实电子结构无关, 仅用于
          管线演示, 所有结果都会明确标注为演示值)。

    Args:
        target_protein: Target protein sequence/identifier
        num_candidates: Number of candidate molecules to screen
        top_k: Number of top molecules to return
        use_real_chemistry: 是否使用真实量子化学计算
    """

    def __init__(self,
                 target_protein: str = "",
                 num_candidates: int = 10,
                 top_k: int = 5,
                 use_real_chemistry: bool = False):
        self.target = target_protein
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.use_real_chemistry = use_real_chemistry
        self._results: List[MoleculeResult] = []

    def _generate_candidates(self) -> List[str]:
        """Generate candidate molecules (built-in library)."""
        candidates = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",       # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",   # Ibuprofen
            "CC1=CC=C(C=C1)C(=O)O",             # Para-toluic acid
            "C1=CC=C(C=C1)C(=O)O",              # Benzoic acid
            "CC1=CC=C(C=C1)S(=O)(=O)N",         # Sulfonamide
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
            "C1=CC(=CC=C1C(=O)O)O",             # 3-Hydroxybenzoic acid
            "CC(=O)NC1=CC=C(C=C1)O",            # Paracetamol
            "C1=CC=C2C(=C1)C=CC(=O)O2",         # Coumarin
            "CC1=NC=C(C(=O)N1)C",               # Pyrimidine
        ]
        return candidates[:self.num_candidates]

    def _compute_binding_energy(self, molecule_smiles: str) -> Tuple[float, float]:
        """
        Compute an approximate binding energy using VQE.

        use_real_chemistry=True 时使用 OpenFermion+PySCF 计算的真实
        分子哈密顿量; 否则使用演示哈密顿量 (与真实电子结构无关)。
        VQE 失败时抛出异常, 不再静默返回随机数。
        """
        from ..algorithms.vqe import VQE

        if self.use_real_chemistry:
            hamiltonian, n_qubits = self._build_molecular_hamiltonian(
                molecule_smiles)
        else:
            hamiltonian = self._build_demo_hamiltonian(molecule_smiles)
            n_qubits = 4

        vqe = VQE(
            n_qubits=n_qubits,
            hamiltonian_pauli_terms=hamiltonian,
            ansatz_type="he",
            n_layers=2,
        )

        energy, _ = vqe.optimize(maxiter=100, verbose=False)

        binding_energy = energy + 1.0  # shift for demonstration
        confidence = min(0.95, 1.0 / abs(binding_energy + 0.5))
        return binding_energy, confidence

    def _build_molecular_hamiltonian(self, smiles: str) -> Tuple[List[Tuple[float, str]], int]:
        """
        使用 OpenFermion + PySCF + RDKit 计算真实分子哈密顿量。

        流程: SMILES -> RDKit 3D 构象 -> PySCF 一/二电子积分
        -> Jordan-Wigner 变换 -> Pauli 字符串列表。

        返回:
            (hamiltonian_terms, n_qubits)

        失败模式:
            - 缺少依赖 (openfermion/pyscf/rdkit): 抛出
              DependencyNotInstalledError 并附安装说明
        """
        from ..utils.exceptions import DependencyNotInstalledError

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as e:
            raise DependencyNotInstalledError(
                "rdkit",
                "pip install rdkit openfermion openfermionpyscf pyscf"
            ) from e
        try:
            from openfermion import MolecularData, jordan_wigner
            from openfermionpyscf import run_pyscf
        except ImportError as e:
            raise DependencyNotInstalledError(
                "openfermion/openfermionpyscf",
                "pip install openfermion openfermionpyscf pyscf"
            ) from e

        # SMILES -> 3D 坐标
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析 SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        atoms = []
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))

        # PySCF 计算电子积分 (STO-3G 基组)
        geometry = atoms
        molecule = MolecularData(geometry, basis="sto-3g", multiplicity=1)
        molecule = run_pyscf(molecule, run_scf=True, run_fci=False)

        # Jordan-Wigner -> Pauli 字符串
        jw_hamiltonian = jordan_wigner(molecule.get_molecular_hamiltonian())
        terms = []
        for pauli_str, coeff in jw_hamiltonian.terms.items():
            # openfermion 的 pauli_str 是 ((qubit, 'X'), ...) 形式
            pauli = ["I"] * molecule.n_qubits
            for qubit_idx, pauli_op in pauli_str:
                pauli[qubit_idx] = pauli_op
            terms.append((float(np.real(coeff)), "".join(pauli)))

        return terms, molecule.n_qubits

    def _build_demo_hamiltonian(self, smiles: str) -> List[Tuple[float, str]]:
        """
        Build a DEMO molecular Hamiltonian from SMILES.

        WARNING: This is randomly generated and has no relation to real
        electronic structure. Real molecular Hamiltonians must be computed
        using quantum chemistry packages (OpenFermion, PySCF, etc.).
        """
        hash_val = sum(ord(c) for c in smiles)
        rng = np.random.RandomState(hash_val % (2**32))

        pauli_strings = ["IIII", "ZIII", "IZII", "IIZI", "IIIZ",
                         "ZZII", "ZIZI", "ZIIZ", "IZZI", "IIZZ"]

        terms = []
        for pauli_str in pauli_strings:
            coeff = rng.normal(0, 0.5) if 'Z' in pauli_str[1:] else rng.normal(-1, 0.3)
            terms.append((coeff, pauli_str))

        return terms

    def run(self, verbose: bool = True) -> List[MoleculeResult]:
        """
        Run the drug discovery pipeline (demonstration mode).

        Args:
            verbose: Print progress

        Returns:
            List of MoleculeResult, ranked by binding energy
        """
        if verbose:
            mode = ("REAL CHEMISTRY (OpenFermion+PySCF)"
                    if self.use_real_chemistry else "DEMONSTRATION")
            print(f"\n  Drug Discovery Pipeline ({mode})")
            print(f"  Target: {self.target or 'Default protein'}")
            if not self.use_real_chemistry:
                print(f"  WARNING: Using demo Hamiltonians - not real electronic structure")
            print(f"  Screening {self.num_candidates} candidates...\n")

        candidates = self._generate_candidates()

        results = []
        for i, mol in enumerate(candidates):
            if verbose:
                print(f"  [{i+1}/{len(candidates)}] Computing binding energy for {mol[:20]}...")

            energy, confidence = self._compute_binding_energy(mol)
            results.append(MoleculeResult(
                smiles=mol,
                binding_energy=energy,
                confidence=confidence,
                rank=0,
            ))

        results.sort(key=lambda r: r.binding_energy)
        for i, r in enumerate(results):
            r.rank = i + 1

        self._results = results

        if verbose:
            print(f"\n  Top {self.top_k} Candidates:")
            for r in results[:self.top_k]:
                print(f"    #{r.rank}: {r.smiles[:25]:25s} | E_bind = {r.binding_energy:8.6f} H | conf = {r.confidence:.2%}")

        return results[:self.top_k]

    @property
    def results(self) -> List[MoleculeResult]:
        return self._results
