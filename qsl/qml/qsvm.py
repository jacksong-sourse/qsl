"""Quantum Support Vector Machine (QSVM) with sklearn-compatible API."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .kernels import quantum_kernel


def _quantum_kernel_callable(X1, X2, n_qubits=None):
    """Pre-compute kernel function compatible with sklearn's SVC."""
    
    def kernel_func(A, B):
        return quantum_kernel(A, B, n_qubits=n_qubits)
    
    return kernel_func


class QuantumSVM(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Machine.
    
    Uses a quantum kernel (fidelity-based) with sklearn's SVC.
    Fully compatible with sklearn's API: fit(), predict(), predict_proba().
    
    Args:
        n_qubits: Number of qubits for quantum feature map (default: n_features)
        C: Regularization parameter (same as sklearn SVC)
        gamma: Kernel coefficient (for RBF approximation, unused by default)
        use_rbf: If True, use RBF-inspired quantum kernel
        probability: If True, enable probability estimates
        random_state: Random seed
        scale: If True, standardize features before fitting
    """
    
    def __init__(self,
                 n_qubits: int = None,
                 C: float = 1.0,
                 gamma: float = 1.0,
                 use_rbf: bool = False,
                 probability: bool = False,
                 random_state: int = None,
                 scale: bool = True):
        self.n_qubits = n_qubits
        self.C = C
        self.gamma = gamma
        self.use_rbf = use_rbf
        self.probability = probability
        self.random_state = random_state
        self.scale = scale
    
    def fit(self, X, y):
        """
        Fit the QSVM model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        if self.n_qubits is None:
            self.n_qubits_ = X.shape[1]
        else:
            self.n_qubits_ = self.n_qubits
        
        # Scale features
        if self.scale:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
        
        # Build quantum kernel
        def custom_kernel(A, B):
            if self.use_rbf:
                from .kernels import rbf_quantum_kernel
                return rbf_quantum_kernel(A, B, gamma=self.gamma, n_qubits=self.n_qubits_)
            else:
                return quantum_kernel(A, B, n_qubits=self.n_qubits_)
        
        # Pre-compute kernel matrix for training
        n_samples = X.shape[0]
        cost_estimate = self.n_qubits_ * n_samples * n_samples
        if cost_estimate > 10000:
            import warnings
            warnings.warn(
                f"QSVM kernel precomputation may be slow: "
                f"n_qubits={self.n_qubits_}, n_samples={n_samples}, "
                f"n_qubits * n_samples^2 = {cost_estimate}. "
                f"Consider incremental kernel methods or dimensionality reduction.",
                RuntimeWarning, stacklevel=2
            )
        K_train = custom_kernel(X, X)
        
        # Fit SVC with pre-computed kernel
        self.svc_ = SVC(
            kernel='precomputed',
            C=self.C,
            probability=self.probability,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.svc_.fit(K_train, y)
        
        # Store for prediction
        self._X_train = X
        self._custom_kernel = custom_kernel
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        K_test = self._custom_kernel(X, self._X_train)
        return self.svc_.predict(K_test)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Test data
            
        Returns:
            Probability array
        """
        check_is_fitted(self)
        
        if not self.probability:
            raise ValueError("Set probability=True to use predict_proba")
        
        X = check_array(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        K_test = self._custom_kernel(X, self._X_train)
        return self.svc_.predict_proba(K_test)
    
    def score(self, X, y):
        """Return mean accuracy on test data."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    def get_params(self, deep=True):
        """Get parameters (sklearn compatibility)."""
        return super().get_params(deep)
    
    def set_params(self, **params):
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
