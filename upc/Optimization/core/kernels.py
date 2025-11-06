"""
Kernels: Pluggable kernel functions for Gaussian Process models.

Kernels define the covariance structure of the GP, controlling how
the model interpolates between observed points.

Common kernels:
- Matern: Flexible, good for simulation data
- RBF (Gaussian): Smooth, infinitely differentiable
- Custom: User-defined kernels
"""

from abc import ABC, abstractmethod
from typing import Any

try:
    from sklearn.gaussian_process.kernels import (
        Kernel,
        Matern,
        RBF,
        ConstantKernel as C,
        WhiteKernel,
        Product,
        Sum
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    Kernel = object  # Fallback


class BaseKernel(ABC):
    """
    Abstract base class for GP kernels.
    
    Kernels must provide a method to get sklearn-compatible kernel object.
    """
    
    @abstractmethod
    def get_sklearn_kernel(self) -> Any:
        """
        Get scikit-learn compatible kernel object.
        
        Returns:
            sklearn kernel instance
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaternKernel(BaseKernel):
    """
    Matérn kernel for Gaussian Process.
    
    The Matérn kernel is a generalization of the RBF kernel with a parameter nu
    that controls smoothness. It's particularly good for simulation data where
    the function may not be infinitely differentiable.
    
    Parameters:
    - nu: Smoothness parameter
      * nu = 0.5: Equivalent to Exponential kernel (rough)
      * nu = 1.5: Once differentiable
      * nu = 2.5: Twice differentiable (good default)
      * nu = ∞: Equivalent to RBF kernel (infinitely smooth)
    
    - length_scale: Controls how far the influence of a data point extends
      * Small: Function varies quickly
      * Large: Function varies slowly
      * Can be scalar (isotropic) or vector (different per dimension)
    
    Pros:
    - More flexible than RBF
    - Better for non-smooth functions
    - nu=2.5 is a good default for simulations
    
    Cons:
    - Slightly more complex than RBF
    - Requires choosing nu
    """
    
    def __init__(
        self,
        nu: float = 2.5,
        length_scale: float = 1.0,
        length_scale_bounds: tuple = (1e-5, 1e5),
        constant_value: float = 1.0,
        constant_value_bounds: tuple = (1e-5, 1e5)
    ):
        """
        Initialize Matérn kernel.
        
        Args:
            nu: Smoothness parameter (0.5, 1.5, 2.5, or np.inf)
            length_scale: Initial length scale
            length_scale_bounds: Bounds for length scale optimization
            constant_value: Initial constant multiplier
            constant_value_bounds: Bounds for constant optimization
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Matérn kernel")
        
        self.nu = nu
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds
    
    def get_sklearn_kernel(self) -> Kernel:
        """Get sklearn Matérn kernel with constant multiplier."""
        matern = Matern(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
            nu=self.nu
        )
        
        constant = C(
            constant_value=self.constant_value,
            constant_value_bounds=self.constant_value_bounds
        )
        
        return constant * matern
    
    def __repr__(self) -> str:
        return f"MaternKernel(nu={self.nu}, length_scale={self.length_scale})"


class RBFKernel(BaseKernel):
    """
    Radial Basis Function (RBF) kernel, also known as Gaussian kernel.
    
    The RBF kernel assumes the function is infinitely differentiable,
    producing very smooth interpolations. It's a good default for many problems.
    
    Formula: k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    
    Parameters:
    - length_scale (l): Controls how far the influence extends
    - constant_value (σ²): Overall variance/amplitude
    
    Pros:
    - Simple and well-understood
    - Works well for smooth functions
    - Single hyperparameter to tune
    
    Cons:
    - Assumes infinite smoothness
    - May overfit noisy data
    - Less flexible than Matérn
    """
    
    def __init__(
        self,
        length_scale: float = 1.0,
        length_scale_bounds: tuple = (1e-5, 1e5),
        constant_value: float = 1.0,
        constant_value_bounds: tuple = (1e-5, 1e5)
    ):
        """
        Initialize RBF kernel.
        
        Args:
            length_scale: Initial length scale
            length_scale_bounds: Bounds for length scale optimization
            constant_value: Initial constant multiplier
            constant_value_bounds: Bounds for constant optimization
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RBF kernel")
        
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds
    
    def get_sklearn_kernel(self) -> Kernel:
        """Get sklearn RBF kernel with constant multiplier."""
        rbf = RBF(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds
        )
        
        constant = C(
            constant_value=self.constant_value,
            constant_value_bounds=self.constant_value_bounds
        )
        
        return constant * rbf
    
    def __repr__(self) -> str:
        return f"RBFKernel(length_scale={self.length_scale})"


class CustomKernel(BaseKernel):
    """
    Wrapper for custom sklearn kernel.
    
    Allows using any sklearn-compatible kernel or kernel composition.
    
    Example:
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        
        # Custom kernel: RBF + noise
        custom = CustomKernel(RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0))
        
        # Custom kernel: Product of kernels
        custom = CustomKernel(
            C(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        )
    """
    
    def __init__(self, kernel: Any):
        """
        Initialize custom kernel.
        
        Args:
            kernel: Any sklearn-compatible kernel object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for custom kernel")
        
        self.kernel = kernel
    
    def get_sklearn_kernel(self) -> Kernel:
        """Return the custom kernel."""
        return self.kernel
    
    def __repr__(self) -> str:
        return f"CustomKernel({self.kernel})"


class CompositeKernel(BaseKernel):
    """
    Composite kernel combining multiple kernels.
    
    Supports sum and product operations.
    
    Example:
        # Sum of Matern and White noise
        composite = CompositeKernel(
            [MaternKernel(nu=2.5), WhiteKernel(noise_level=0.1)],
            operation='sum'
        )
        
        # Product of Constant and Matern
        composite = CompositeKernel(
            [C(1.0), MaternKernel(nu=2.5)],
            operation='product'
        )
    """
    
    def __init__(self, kernels: list, operation: str = 'product'):
        """
        Initialize composite kernel.
        
        Args:
            kernels: List of BaseKernel instances or sklearn kernels
            operation: 'sum' or 'product'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for composite kernel")
        
        if operation not in ['sum', 'product']:
            raise ValueError("operation must be 'sum' or 'product'")
        
        self.kernels = kernels
        self.operation = operation
    
    def get_sklearn_kernel(self) -> Kernel:
        """Compose kernels using sum or product."""
        sklearn_kernels = []
        
        for k in self.kernels:
            if isinstance(k, BaseKernel):
                sklearn_kernels.append(k.get_sklearn_kernel())
            else:
                sklearn_kernels.append(k)
        
        if len(sklearn_kernels) == 0:
            raise ValueError("No kernels provided")
        
        if len(sklearn_kernels) == 1:
            return sklearn_kernels[0]
        
        # Compose kernels
        result = sklearn_kernels[0]
        for k in sklearn_kernels[1:]:
            if self.operation == 'sum':
                result = result + k
            else:  # product
                result = result * k
        
        return result
    
    def __repr__(self) -> str:
        op_str = ' + ' if self.operation == 'sum' else ' * '
        kernel_strs = [str(k) for k in self.kernels]
        return f"CompositeKernel({op_str.join(kernel_strs)})"


def get_kernel(name: str, **kwargs) -> BaseKernel:
    """
    Factory function to get kernel by name.
    
    Args:
        name: Kernel name ('matern', 'rbf', 'custom')
        **kwargs: Kernel-specific parameters
    
    Returns:
        Kernel instance
    """
    kernels = {
        'matern': MaternKernel,
        'rbf': RBFKernel,
    }
    
    name = name.lower()
    if name not in kernels:
        raise ValueError(f"Unknown kernel: {name}. Choose from {list(kernels.keys())}")
    
    return kernels[name](**kwargs)
