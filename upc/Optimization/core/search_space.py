"""
SearchSpace: Parse JSON configuration and generate valid design space.

This module handles:
1. Loading parameter configurations from JSON files
2. Validating parameter ranges
3. Generating all valid design points
4. Filtering based on constraints
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
from itertools import product


class SearchSpace:
    """
    Manages the search space for optimization.
    
    Parses JSON configuration files and generates valid design space
    based on constraints (e.g., dp * mp * sp * pp == num_npus).
    
    Example JSON structure:
    {
        "npu_count": [16, 32, 64, 128],
        "parallelism_strategy": {
            "dp": [1, 2, 4, 8],
            "mp": [1, 2, 4, 8, 16],
            "sp": [1, 2, 4],
            "pp": [1, 2, 4],
            "FSDP": [0, 1]
        }
    }
    """
    
    def __init__(self, config_path: str, num_npus: Optional[int] = None):
        """
        Initialize search space from JSON configuration.
        
        Args:
            config_path: Path to JSON configuration file
            num_npus: Number of NPUs (if None, read from config)
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Set num_npus
        if num_npus is not None:
            self.num_npus = num_npus
        elif "npu_count" in self.config:
            # Use first value if list, or the value itself
            npu_count = self.config["npu_count"]
            self.num_npus = npu_count[0] if isinstance(npu_count, list) else npu_count
        else:
            raise ValueError("num_npus not provided and not found in config")
        
        # Extract parameter ranges
        self.parameter_ranges = self._extract_parameter_ranges()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _extract_parameter_ranges(self) -> Dict[str, List]:
        """
        Extract parameter ranges from configuration.
        
        Returns:
            Dictionary mapping parameter names to their possible values
        """
        ranges = {}
        
        # Check if we have parallelism_strategy section
        if "parallelism_strategy" in self.config:
            params = self.config["parallelism_strategy"]
            
            # Extract parallelism parameters
            ranges['dp'] = params.get('dp', [1, 2, 4, 8])
            ranges['mp'] = params.get('mp', [1, 2, 4, 8, 16])
            ranges['sp'] = params.get('sp', [1, 2, 4, 8])
            ranges['pp'] = params.get('pp', [1, 2, 4, 8])
            ranges['sharded'] = [bool(x) for x in params.get('FSDP', [0, 1])]
        else:
            # Default ranges
            ranges['dp'] = [1, 2, 4, 8]
            ranges['mp'] = [1, 2, 4, 8, 16]
            ranges['sp'] = [1, 2, 4, 8]
            ranges['pp'] = [1, 2, 4, 8]
            ranges['sharded'] = [True, False]
        
        return ranges
    
    def get_parameters(self) -> Dict[str, List]:
        """
        Get all parameter ranges.
        
        Returns:
            Dictionary of parameter names to possible values
        """
        return self.parameter_ranges.copy()
    
    def get_design_space(self, num_npus: Optional[int] = None) -> List[Tuple]:
        """
        Generate all valid design points.
        
        A design point is valid if:
        1. dp * mp * sp * pp == num_npus
        2. All values are within specified ranges
        
        Args:
            num_npus: Number of NPUs (uses self.num_npus if None)
        
        Returns:
            List of valid configurations as (dp, mp, sp, pp, sharded) tuples
        """
        target_npus = num_npus if num_npus is not None else self.num_npus
        
        design_space = []
        
        # Get parameter ranges
        dp_range = self.parameter_ranges['dp']
        mp_range = self.parameter_ranges['mp']
        pp_range = self.parameter_ranges['pp']
        sharded_options = self.parameter_ranges['sharded']
        
        # Generate all combinations
        for dp in dp_range:
            for mp in mp_range:
                for pp in pp_range:
                    # Calculate required sp
                    if (dp * mp * pp) > 0 and target_npus % (dp * mp * pp) == 0:
                        sp = target_npus // (dp * mp * pp)
                        
                        # Check if sp is valid
                        if sp >= 1:  # sp must be at least 1
                            for sharded in sharded_options:
                                design_space.append((dp, mp, sp, pp, sharded))
        
        return design_space
    
    def validate_config(self, config: Tuple) -> bool:
        """
        Validate a configuration.
        
        Args:
            config: Configuration tuple (dp, mp, sp, pp, sharded)
        
        Returns:
            True if valid, False otherwise
        """
        dp, mp, sp, pp, sharded = config
        
        # Check if product equals num_npus
        if dp * mp * sp * pp != self.num_npus:
            return False
        
        # Check if all values are positive
        if any(x < 1 for x in [dp, mp, sp, pp]):
            return False
        
        # Check if sharded is boolean
        if not isinstance(sharded, bool):
            return False
        
        return True
    
    def get_design_space_size(self, num_npus: Optional[int] = None) -> int:
        """
        Get the size of the design space without generating it.
        
        Args:
            num_npus: Number of NPUs (uses self.num_npus if None)
        
        Returns:
            Number of valid configurations
        """
        return len(self.get_design_space(num_npus))
    
    def sample_random(self, n_samples: int, num_npus: Optional[int] = None) -> List[Tuple]:
        """
        Sample random configurations from design space.
        
        Args:
            n_samples: Number of samples
            num_npus: Number of NPUs (uses self.num_npus if None)
        
        Returns:
            List of random configurations
        """
        import random
        
        design_space = self.get_design_space(num_npus)
        
        if n_samples >= len(design_space):
            return design_space
        
        return random.sample(design_space, n_samples)
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for continuous optimization methods.
        
        Returns:
            Dictionary of parameter bounds (min, max)
        """
        bounds = {}
        
        for param, values in self.parameter_ranges.items():
            if param == 'sharded':
                bounds[param] = (0.0, 1.0)  # Binary parameter
            else:
                bounds[param] = (float(min(values)), float(max(values)))
        
        return bounds
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SearchSpace(num_npus={self.num_npus}, "
                f"design_space_size={self.get_design_space_size()})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        info = [
            f"SearchSpace for {self.num_npus} NPUs",
            f"Configuration: {self.config_path}",
            f"Design space size: {self.get_design_space_size()}",
            f"Parameter ranges:"
        ]
        
        for param, values in self.parameter_ranges.items():
            info.append(f"  {param}: {values}")
        
        return "\n".join(info)
