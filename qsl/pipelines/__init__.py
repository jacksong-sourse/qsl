"""QSL Application Pipelines - End-to-end quantum solutions."""

try:
    from .drug_discovery import DrugDiscoveryPipeline
except ImportError:
    DrugDiscoveryPipeline = None

try:
    from .crypto_analysis import CryptoAnalysisPipeline
except ImportError:
    CryptoAnalysisPipeline = None

try:
    from .portfolio import PortfolioOptimizer
except ImportError:
    PortfolioOptimizer = None

__all__ = ["DrugDiscoveryPipeline", "CryptoAnalysisPipeline", "PortfolioOptimizer"]
