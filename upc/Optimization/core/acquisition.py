"""
Acquisition Functions: Determine next point to evaluate in Bayesian Optimization.

Acquisition functions balance exploration (trying uncertain regions) and
exploitation (trying regions where we expect good performance).

Common acquisition functions:
- Expected Improvement (EI): Expected improvement over current best
- Upper Confidence Bound (UCB): Mean + kappa * std
- Probability of Improvement (PI): Probability of improving over best
"""

from abc import ABC, abstractmethod
import numpy as np

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class BaseAcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    
    Acquisition functions score candidate points based on GP predictions,
    guiding the search toward promising regions.
    """
    
    @abstractmethod
    def compute(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute acquisition scores.
        
        Args:
            mu: GP mean predictions
            sigma: GP standard deviations (uncertainty)
            best_y: Current best observed value
            **kwargs: Additional parameters
        
        Returns:
            Acquisition scores (higher is better)
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.
    
    EI measures the expected improvement over the current best, accounting for
    both the magnitude of improvement and the probability of achieving it.
    
    Formula for minimization:
        EI(x) = (f_best - μ(x) - ξ) * Φ(Z) + σ(x) * φ(Z)
        where Z = (f_best - μ(x) - ξ) / σ(x)
    
    Parameters:
    - xi: Exploration parameter
      * xi = 0.0: Pure exploitation (greedy)
      * xi = 0.01: Mostly exploit with slight exploration (DEFAULT)
      * xi = 0.1: Balanced exploration-exploitation
      * xi > 0.1: Aggressive exploration
    
    Behavior:
    - High EI when μ(x) is low (good predicted performance)
    - High EI when σ(x) is high (high uncertainty)
    - Balances exploitation and exploration
    
    Pros:
    - Well-balanced exploration-exploitation
    - Intuitive interpretation
    - Works well in practice
    
    Cons:
    - Can be conservative
    - Requires careful tuning of xi
    """
    
    def __init__(self, xi: float = 0.01):
        """
        Initialize EI acquisition function.
        
        Args:
            xi: Exploration-exploitation trade-off parameter
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Expected Improvement")
        
        self.xi = xi
    
    def compute(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Expected Improvement.
        
        Args:
            mu: GP mean predictions
            sigma: GP standard deviations
            best_y: Current best observed value
        
        Returns:
            EI scores for each point
        """
        # Ensure numerical stability
        sigma = np.maximum(sigma, 1e-9)
        
        # Improvement over current best (for minimization)
        # Subtracting xi encourages exploration
        improvement = best_y - mu - self.xi
        
        # Standardized improvement
        Z = improvement / sigma
        
        # Expected Improvement formula
        # First term: exploitation (expected improvement weighted by probability)
        # Second term: exploration (uncertainty bonus)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Handle edge case where sigma is zero (completely certain)
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    def __repr__(self) -> str:
        return f"ExpectedImprovement(xi={self.xi})"


class UpperConfidenceBound(BaseAcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB uses a simple formula that explicitly trades off exploitation (mean)
    and exploration (uncertainty):
    
    Formula for minimization:
        UCB(x) = -μ(x) + κ * σ(x)
    
    (We negate for minimization; higher UCB means lower predicted value
    plus exploration bonus)
    
    Parameters:
    - kappa (κ): Exploration parameter
      * κ = 0: Pure exploitation
      * κ = 2.0: Moderate exploration (~95% confidence)
      * κ = 2.576: Strong exploration (~99% confidence)
      * κ > 3: Aggressive exploration
    
    Behavior:
    - Prefers points with low mean (good predicted performance)
    - Prefers points with high uncertainty (exploration)
    - Simpler than EI, easier to tune
    
    Pros:
    - Simple and intuitive
    - Single parameter to tune
    - Good theoretical properties
    
    Cons:
    - Can be too exploratory
    - Doesn't account for improvement probability
    """
    
    def __init__(self, kappa: float = 2.576):
        """
        Initialize UCB acquisition function.
        
        Args:
            kappa: Exploration-exploitation trade-off parameter
                  (2.576 corresponds to 99% confidence interval)
        """
        self.kappa = kappa
    
    def compute(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Upper Confidence Bound.
        
        Args:
            mu: GP mean predictions
            sigma: GP standard deviations
            best_y: Current best observed value (not used in UCB)
        
        Returns:
            UCB scores for each point
        """
        # For minimization: prefer low mean + high uncertainty
        ucb = -mu + self.kappa * sigma
        
        return ucb
    
    def __repr__(self) -> str:
        return f"UpperConfidenceBound(kappa={self.kappa})"


class ProbabilityOfImprovement(BaseAcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.
    
    PI computes the probability that a point will improve over the current best.
    
    Formula for minimization:
        PI(x) = Φ((f_best - μ(x) - ξ) / σ(x))
    
    Parameters:
    - xi: Exploration parameter (similar to EI)
      * xi = 0.0: Pure exploitation
      * xi = 0.01: Slight exploration (DEFAULT)
      * xi > 0.1: More exploration
    
    Behavior:
    - High PI when μ(x) is low (good predicted performance)
    - Considers uncertainty but doesn't weight by improvement magnitude
    - More conservative than EI
    
    Pros:
    - Simple interpretation
    - Fast to compute
    - Works well for risk-averse optimization
    
    Cons:
    - Doesn't consider magnitude of improvement
    - Can be too greedy
    - Generally inferior to EI
    """
    
    def __init__(self, xi: float = 0.01):
        """
        Initialize PI acquisition function.
        
        Args:
            xi: Exploration-exploitation trade-off parameter
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Probability of Improvement")
        
        self.xi = xi
    
    def compute(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Probability of Improvement.
        
        Args:
            mu: GP mean predictions
            sigma: GP standard deviations
            best_y: Current best observed value
        
        Returns:
            PI scores for each point
        """
        # Ensure numerical stability
        sigma = np.maximum(sigma, 1e-9)
        
        # Standardized improvement
        Z = (best_y - mu - self.xi) / sigma
        
        # Probability of improvement
        pi = norm.cdf(Z)
        
        # Handle edge case
        pi[sigma == 0.0] = 0.0
        
        return pi
    
    def __repr__(self) -> str:
        return f"ProbabilityOfImprovement(xi={self.xi})"


class ThompsonSampling(BaseAcquisitionFunction):
    """
    Thompson Sampling acquisition function.
    
    Instead of using a deterministic acquisition function, Thompson Sampling
    draws a random sample from the GP posterior and optimizes that sample.
    
    This provides a natural way to balance exploration and exploitation through
    the uncertainty in the GP predictions.
    
    Pros:
    - Theoretically well-founded
    - Natural exploration-exploitation trade-off
    - No hyperparameters to tune
    
    Cons:
    - Stochastic (different runs give different results)
    - Can be less stable than EI/UCB
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize Thompson Sampling.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def compute(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        **kwargs
    ) -> np.ndarray:
        """
        Sample from GP posterior.
        
        Args:
            mu: GP mean predictions
            sigma: GP standard deviations
            best_y: Current best observed value (not used)
        
        Returns:
            Random sample from posterior (lower is better for minimization)
        """
        # Sample from Gaussian posterior
        samples = np.random.normal(mu, sigma)
        
        # Return negative (since higher acquisition is better)
        return -samples
    
    def __repr__(self) -> str:
        return f"ThompsonSampling(seed={self.seed})"


def get_acquisition(name: str, **kwargs) -> BaseAcquisitionFunction:
    """
    Factory function to get acquisition function by name.
    
    Args:
        name: Acquisition function name ('ei', 'ucb', 'pi', 'thompson')
        **kwargs: Function-specific parameters
    
    Returns:
        Acquisition function instance
    """
    acquisitions = {
        'ei': ExpectedImprovement,
        'expected_improvement': ExpectedImprovement,
        'ucb': UpperConfidenceBound,
        'upper_confidence_bound': UpperConfidenceBound,
        'pi': ProbabilityOfImprovement,
        'probability_of_improvement': ProbabilityOfImprovement,
        'thompson': ThompsonSampling,
        'thompson_sampling': ThompsonSampling,
    }
    
    name = name.lower()
    if name not in acquisitions:
        raise ValueError(f"Unknown acquisition: {name}. Choose from {list(set(acquisitions.keys()))}")
    
    return acquisitions[name](**kwargs)
