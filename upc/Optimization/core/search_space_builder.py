"""
SearchSpaceBuilder: Flexible builder for creating search spaces from JSON configurations.

This module provides a builder pattern for creating search spaces with:
1. Support for all parameter types (parallelism, collective, network, hardware, model)
2. Dynamic constraint parsing and evaluation
3. Flexible parameter subsets (doesn't require all parameters)
4. Integration with various sampling strategies
5. Configuration validation and space generation
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Callable
from itertools import product

try:
    from .sampler import get_sampler
except ImportError:
    from sampler import get_sampler


class SearchSpaceBuilder:
    """
    Builder for creating flexible search spaces from JSON configurations.
    
    Features:
    - Parses JSON with arbitrary parameter subsets
    - Extracts and evaluates constraints dynamically
    - Generates valid configurations based on constraints
    - Integrates with sampling strategies
    - Returns configurations as dictionaries for easy use
    
    Example usage:
        builder = SearchSpaceBuilder('config.json', num_npus=128)
        builder.parse_parameters()
        builder.apply_constraints()
        configs = builder.sample(n_samples=20, strategy='random')
    """
    
    def __init__(self, config_path: str):
        """
        Initialize search space builder.
        
        Args:
            config_path: Path to JSON configuration file
            num_npus: Number of NPUs (if None, read from config)
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Set num_npus
        if "npu_count" in self.config:
            npu_count = self.config["npu_count"]
            self.num_npus = npu_count[0] if isinstance(npu_count, list) else npu_count
        else:
            raise ValueError("'npu_count' must be specified in the configuration file")
        
        # Storage for parsed data
        self.parameters: Dict[str, List[Any]] = {}
        self.constraints: List[Callable] = []
        self.constraint_strings: List[str] = []
        self.design_space: Optional[List[Dict[str, Any]]] = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def parse_parameters(self, include_categories: Optional[List[str]] = None,
                        exclude_categories: Optional[List[str]] = None) -> 'SearchSpaceBuilder':
        """
        Parse parameters from configuration.
        
        Args:
            include_categories: Only include these categories (e.g., ['parallelism_strategy', 'network'])
            exclude_categories: Exclude these categories
        
        Returns:
            Self for method chaining
        """
        self.parameters = {}
        
        # Define parameter categories
        categories = {
            'parallelism_strategy': self._parse_parallelism_strategy,
            'collective': self._parse_collective,
            'network': self._parse_network,
            'hardware': self._parse_hardware,
            'model': self._parse_model,
        }
        
        # Filter categories
        if include_categories:
            categories = {k: v for k, v in categories.items() if k in include_categories}
        if exclude_categories:
            categories = {k: v for k, v in categories.items() if k not in exclude_categories}
        
        # Parse each category
        for category, parser in categories.items():
            if category in self.config:
                parser(self.config[category])
        
        return self
    
    def _parse_parallelism_strategy(self, params: Dict) -> None:
        """Parse parallelism strategy parameters."""
        if 'dp' in params:
            self.parameters['dp'] = params['dp']
        if 'mp' in params:
            self.parameters['mp'] = params['mp']
        if 'sp' in params:
            self.parameters['sp'] = params['sp']
        if 'pp' in params:
            self.parameters['pp'] = params['pp']
        if 'FSDP' in params:
            self.parameters['sharded'] = [bool(x) for x in params['FSDP']]
    
    def _parse_collective(self, params: Dict) -> None:
        """Parse collective communication parameters."""
        if 'scheduling_policy' in params:
            self.parameters['scheduling_policy'] = params['scheduling_policy']
        
        if 'collective_implementation' in params:
            impl = params['collective_implementation']
            for collective_type, algorithms in impl.items():
                # Convert 'all-reduce' to 'all_reduce' for valid Python identifiers
                param_name = collective_type.replace('-', '_')
                self.parameters[param_name] = algorithms
        
        if 'active-chunks-per-dimension' in params:
            self.parameters['active_chunks_per_dimension'] = params['active-chunks-per-dimension']
        
        if 'preferred-dataset-splits' in params:
            self.parameters['preferred_dataset_splits'] = params['preferred-dataset-splits']
        
        if 'collective-optimization' in params:
            self.parameters['collective_optimization'] = params['collective-optimization']
    
    def _parse_network(self, params: Dict) -> None:
        """Parse network parameters."""
        if 'topology' in params:
            self.parameters['topology'] = params['topology']
        if 'inter_node_bw' in params:
            self.parameters['inter_node_bw'] = params['inter_node_bw']
        if 'intra_node_bw' in params:
            self.parameters['intra_node_bw'] = params['intra_node_bw']
        if 'npus_per_node' in params:
            self.parameters['npus_per_node'] = params['npus_per_node']
    
    def _parse_hardware(self, params: Dict) -> None:
        """Parse hardware parameters."""
        if 'local_mem_bw' in params:
            self.parameters['local_mem_bw'] = params['local_mem_bw']
        if 'local_mem_size' in params:
            self.parameters['local_mem_size'] = params['local_mem_size']
        if 'peak_perf' in params:
            self.parameters['peak_perf'] = params['peak_perf']
    
    def _parse_model(self, params: Dict) -> None:
        """Parse model parameters."""
        if 'batch_size' in params:
            self.parameters['batch_size'] = params['batch_size']
        if 'micro_batch_size' in params:
            self.parameters['micro_batch_size'] = params['micro_batch_size']
        if 'mixed_precision' in params:
            self.parameters['mixed_precision'] = [bool(x) for x in params['mixed_precision']]
        if 'sequence_length' in params:
            self.parameters['sequence_length'] = params['sequence_length']
        if 'FNN_hidden_size' in params:
            self.parameters['FNN_hidden_size'] = params['FNN_hidden_size']
        if 'num_attention_heads' in params:
            self.parameters['num_attention_heads'] = params['num_attention_heads']
        if 'num_layers' in params:
            self.parameters['num_layers'] = params['num_layers']
    
    def apply_constraints(self, custom_constraints: Optional[List[str]] = None) -> 'SearchSpaceBuilder':
        """
        Parse and apply constraints from configuration.
        
        Args:
            custom_constraints: Additional constraint strings to apply
        
        Returns:
            Self for method chaining
        """
        self.constraints = []
        self.constraint_strings = []
        
        # Parse constraints from config
        if 'constraints' in self.config:
            for name, constraint_str in self.config['constraints'].items():
                self._parse_constraint(constraint_str)
        
        # Add custom constraints
        if custom_constraints:
            for constraint_str in custom_constraints:
                self._parse_constraint(constraint_str)
        
        return self
    
    def _parse_constraint(self, constraint_str: str) -> None:
        """
        Parse a constraint string and create a validation function.
        
        Supported constraint formats:
        - "dp * mp * sp * pp = npu_count"
        - "dp <= npu_count"
        - "batch_size % micro_batch_size = 0"
        - "micro_batch_size <= batch_size"
        
        Args:
            constraint_str: String representation of constraint
        """
        # Store original constraint string
        self.constraint_strings.append(constraint_str)
        
        # Replace 'npu_count' with actual value if available
        if self.num_npus is not None:
            constraint_str = constraint_str.replace('npu_count', str(self.num_npus))
        
        # Create constraint function
        def constraint_func(config: Dict[str, Any]) -> bool:
            try:
                # Replace parameter names with values from config
                expr = constraint_str
                for param, value in config.items():
                    # Use word boundaries to avoid partial replacements
                    expr = re.sub(r'\b' + re.escape(param) + r'\b', str(value), expr)
                
                # Handle comparison operators
                if '=' in expr and not any(op in expr for op in ['<=', '>=', '==', '!=']):
                    # Convert '=' to '=='
                    expr = expr.replace('=', '==')
                
                # Evaluate the expression
                result = eval(expr)
                return bool(result)
            except Exception:
                # If evaluation fails, assume constraint doesn't apply
                return True
        
        self.constraints.append(constraint_func)
    
    def build(self, max_configs: Optional[int] = None) -> 'SearchSpaceBuilder':
        """
        Build the design space by generating all valid configurations.
        
        Args:
            max_configs: Maximum number of configurations to generate (for large spaces)
        
        Returns:
            Self for method chaining
        """
        if not self.parameters:
            raise ValueError("No parameters parsed. Call parse_parameters() first.")
        
        # Generate all combinations
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name] for name in param_names]
        
        self.design_space = []
        
        # Use product to generate all combinations
        count = 0
        for values in product(*param_values):
            if max_configs and count >= max_configs:
                break
            
            # Create configuration dictionary
            config = dict(zip(param_names, values))
            
            # Apply constraints
            if self._validate_config(config):
                self.design_space.append(config)
                count += 1
        
        return self
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration against all constraints.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            True if valid, False otherwise
        """
        for constraint in self.constraints:
            if not constraint(config):
                return False
        return True
    
    def sample(self, n_samples: int, strategy: str = 'random', 
               seed: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Sample configurations from the design space.
        
        Args:
            n_samples: Number of samples to generate
            strategy: Sampling strategy ('random', 'lhs', 'sobol', 'grid', 'stratified')
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for sampler
        
        Returns:
            List of sampled configuration dictionaries
        """
        if self.design_space is None:
            raise ValueError("Design space not built. Call build() first.")
        
        if len(self.design_space) == 0:
            raise ValueError("Design space is empty. Check your constraints.")
        
        # Convert design space to list of tuples for sampler
        param_names = list(self.design_space[0].keys())
        design_space_tuples = [
            tuple(config[name] for name in param_names)
            for config in self.design_space
        ]
        
        # Get sampler
        sampler = get_sampler(strategy, seed=seed, **kwargs)
        
        # Sample
        sampled_tuples = sampler.sample(design_space_tuples, n_samples)
        
        # Convert back to dictionaries
        sampled_configs = [
            dict(zip(param_names, values))
            for values in sampled_tuples
        ]
        
        return sampled_configs
    
    def get_design_space(self) -> List[Dict[str, Any]]:
        """
        Get the full design space.
        
        Returns:
            List of all valid configuration dictionaries
        """
        if self.design_space is None:
            raise ValueError("Design space not built. Call build() first.")
        
        return self.design_space.copy()
    
    def get_design_space_size(self) -> int:
        """
        Get the size of the design space.
        
        Returns:
            Number of valid configurations
        """
        if self.design_space is None:
            return 0
        return len(self.design_space)
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about parsed parameters.
        
        Returns:
            Dictionary with parameter info (name, type, range, count)
        """
        info = {}
        
        for param, values in self.parameters.items():
            param_type = type(values[0]).__name__ if values else 'unknown'
            info[param] = {
                'type': param_type,
                'values': values,
                'count': len(values),
                'min': min(values) if all(isinstance(v, (int, float)) for v in values) else None,
                'max': max(values) if all(isinstance(v, (int, float)) for v in values) else None,
            }
        
        return info
    
    def summary(self) -> str:
        """
        Get a summary of the search space.
        
        Returns:
            Human-readable summary string
        """
        lines = [
            "=" * 70,
            "SEARCH SPACE SUMMARY",
            "=" * 70,
        ]
        
        # Configuration info
        lines.append(f"\n📁 Configuration: {self.config_path}")
        if self.num_npus:
            lines.append(f"🖥️  NPUs: {self.num_npus}")
        
        # Parameters
        lines.append(f"\n📊 Parameters ({len(self.parameters)}):")
        for param, values in self.parameters.items():
            lines.append(f"   {param}: {len(values)} values")
        
        # Constraints
        lines.append(f"\n🔒 Constraints ({len(self.constraint_strings)}):")
        for i, constraint in enumerate(self.constraint_strings, 1):
            lines.append(f"   {i}. {constraint}")
        
        # Design space
        if self.design_space is not None:
            lines.append(f"\n🌌 Design Space Size: {len(self.design_space):,} configurations")
            
            # Show theoretical size without constraints
            theoretical_size = 1
            for values in self.parameters.values():
                theoretical_size *= len(values)
            lines.append(f"   Theoretical size: {theoretical_size:,}")
            lines.append(f"   Reduction: {100 * (1 - len(self.design_space) / theoretical_size):.1f}%")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_design_space(self, output_path: str) -> None:
        """
        Save design space to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.design_space is None:
            raise ValueError("Design space not built. Call build() first.")
        
        with open(output_path, 'w') as f:
            json.dump(self.design_space, f, indent=2)
        
        print(f"Design space saved to: {output_path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SearchSpaceBuilder(params={len(self.parameters)}, "
                f"constraints={len(self.constraints)}, "
                f"space_size={self.get_design_space_size()})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.summary()


# Convenience function for quick usage
def create_search_space(config_path: str, 
                       include_categories: Optional[List[str]] = None,
                       exclude_categories: Optional[List[str]] = None,
                       custom_constraints: Optional[List[str]] = None,
                       max_configs: Optional[int] = None) -> SearchSpaceBuilder:
    """
    Convenience function to create and build a search space in one call.
    
    Args:
        config_path: Path to JSON configuration file
        num_npus: Number of NPUs
        include_categories: Categories to include
        exclude_categories: Categories to exclude
        custom_constraints: Additional constraints
        max_configs: Maximum configurations to generate
    
    Returns:
        Built SearchSpaceBuilder instance
    """
    builder = SearchSpaceBuilder(config_path)
    builder.parse_parameters(include_categories, exclude_categories)
    builder.apply_constraints(custom_constraints)
    builder.build(max_configs)
    
    return builder
