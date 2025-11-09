"""
SimulationRunner: Handle AstraSim simulation execution.

This module:
1. Generates workloads using workload_generator
2. Runs AstraSim simulations
3. Parses results using output_parser
4. Handles errors and cleanup
"""

import os
import sys
import shutil
from typing import Dict, Optional

# Add UPC to path
sys.path.append('/media/mohammad/extension/experiments/astra-sim/upc')
from run_astrasim import run_astrasim

# Import helper modules
sys.path.append('/media/mohammad/extension/experiments/astra-sim/upc/Optimization')
from ..helper import workload_generator, output_parser, NetworkConfig, config_parser


class SimulationRunner:
    """
    Manages AstraSim simulation execution.
    
    Responsibilities:
    - Setup directories
    - Generate workloads
    - Run simulations
    - Parse results
    - Cleanup
    
    Example:
        runner = SimulationRunner(
            model_num=40,
            model_name="GPT_40B",
            num_npus=64,
            network_name="FoldedClos"
        )
        
        exec_time = runner.run_simulation((dp, mp, sp, pp, sharded))
    """
    
    def __init__(
        self,
        model_num: int,
        model_name: str,
        num_npus: int,
        network_name: str = "FoldedClos",
        sim_type: str = "analytical_unaware",
        base_dir: str = "/media/mohammad/extension/experiments/astra-sim/upc",
        folder_prefix: str = "OPT",
        clean_on_init: bool = True,
        verbose: bool = False,
        memory_config: Optional[str] = None,
        output_dir: Optional[str] = None,
        network_log_dir: Optional[str] = None,
        system_config: Optional[str] = None,
        network_config: Optional[str] = None
    ):
        """
        Initialize simulation runner.
        
        Args:
            model_num: Model identifier (from workload_generator.Model)
            model_name: Human-readable model name
            num_npus: Number of NPUs
            network_name: Network topology (FoldedClos, Switch, Ring, etc.)
            sim_type: Simulation type (analytical_unaware, g2, etc.)
            base_dir: Base directory for AstraSim
            folder_prefix: Prefix for output folders
            clean_on_init: Whether to clean previous results on initialization
            verbose: Whether to print detailed messages
            memory_config: Custom path to memory configuration file (default: ./configuration/RemoteMemory.json)
            output_dir: Custom output directory (default: {base_dir}/output/{folder_prefix}_{model_name}/{network_name})
            network_log_dir: Custom network log directory (default: {base_dir}/network_log/{folder_prefix}_{model_name}/{network_name})
            system_config: Custom system configuration path (overrides NetworkConfig helper)
            network_config: Custom network configuration path (overrides NetworkConfig helper)
        """
        self.model_num = model_num
        self.model_name = model_name
        self.num_npus = num_npus
        self.network_name = network_name
        self.sim_type = sim_type
        self.base_dir = base_dir
        self.verbose = verbose
        
        # Folder names
        self.folder_name = f"{folder_prefix}_{model_name}"
        
        # Setup network configuration (use custom or default from helper)
        if system_config is not None and network_config is not None:
            # User provided both custom configs
            self.system_config = system_config
            self.network_config = network_config
        else:
            # Use NetworkConfig helper
            self.network_config_helper = NetworkConfig(network_name)
            helper_system, helper_network = self.network_config_helper.get_paths()
            self.system_config = system_config if system_config is not None else helper_system
            self.network_config = network_config if network_config is not None else helper_network
        
        # Setup memory config (use custom or default)
        self.memory_config = memory_config if memory_config is not None else "./configuration/RemoteMemory.json"
        
        # Setup directories (use custom or default)
        self.workload_dir = f"{base_dir}/workload/{self.folder_name}"
        self.output_dir = output_dir if output_dir is not None else f"{base_dir}/output/{self.folder_name}/{network_name}"
        self.network_log_dir = network_log_dir if network_log_dir is not None else f"{base_dir}/network_log/{self.folder_name}/{network_name}"
        
        # Clean and create directories
        if clean_on_init:
            self.clean_previous_results()
        self.setup_directories()
    
    def setup_directories(self):
        """Create required directories."""
        os.makedirs(self.workload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.network_log_dir, exist_ok=True)
        
        if self.verbose:
            print(f"Created directories:")
            print(f"  Workload: {self.workload_dir}")
            print(f"  Output: {self.output_dir}")
            print(f"  Network logs: {self.network_log_dir}")
    
    def clean_previous_results(self):
        """Clean previous simulation results."""
        dirs_to_clean = [
            self.workload_dir,
            self.output_dir,
            self.network_log_dir
        ]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                if self.verbose:
                    print(f"Removing: {dir_path}")
                shutil.rmtree(dir_path)
    
    def run_simulation(self, config: Dict, return_paths: bool = False):
        """
        Run simulation for a configuration.
        
        Args:
            config: Configuration dictionary with dp, mp, sp, pp, sharded
            return_paths: If True, return (exec_time, file_paths, metadata) tuple
        
        Returns:
            If return_paths=False: Execution time in seconds, or None if failed
            If return_paths=True: (exec_time, file_paths_dict, metadata_dict) tuple, or None if failed
        """
        # Extract values from config dictionary
        dp = config['dp']
        mp = config['mp']
        sp = config['sp']
        pp = config['pp']
        sharded = config['sharded']
        
        if self.verbose:
            print(f"  Running: dp={dp}, mp={mp}, sp={sp}, pp={pp}, sharded={sharded}")
        
        try:
            # 1. Generate workload
            success = workload_generator.generate_workload_with_env(
                config,
                workload_generator.Model(self.model_num),
                self.folder_name
            )
            
            if not success:
                if self.verbose:
                    print("    ⚠️  Workload generation failed")
                return None
            
            # 2. Find generated workload file
            config_name = f"{dp}_{mp}_{sp}_{pp}_{1 if sharded else 0}"
            workload_file = self._find_workload_file(config_name)
            
            if workload_file is None:
                if self.verbose:
                    print("    ⚠️  Workload file not found")
                return None
            
            # 3. Run simulation
            if self.verbose:
                print(f"    Running simulation: {workload_file}")
            
            result = self._run_astrasim(workload_file)
            
            if result != "":
                if self.verbose:
                    print(f"    ⚠️  Simulation failed: {result}")
                return None
            
            # 4. Extract execution time
            exec_time = self._extract_execution_time(workload_file)
            
            if exec_time is None:
                if self.verbose:
                    print("    ⚠️  Could not extract execution time")
                return None
            
            if self.verbose:
                print(f"    ✓ Execution time: {exec_time:.2f}s")
            
            # 5. Return with file paths and metadata if requested
            if return_paths:
                # Get base filename for output files
                config_basename = os.path.basename(workload_file)
                file_paths = {
                    'workload': workload_file,  # Base path without numbered extension
                    'output_pattern': os.path.join(self.output_dir, config_basename)  # Base path for output files
                }
                
                # Collect metadata about the actual simulation configuration
                metadata = self._get_simulation_metadata()
                
                return exec_time, file_paths, metadata
            else:
                return exec_time
            
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  Error: {e}")
            return None
    
    def _get_simulation_metadata(self) -> Dict:
        """
        Get metadata about the simulation configuration.
        
        Returns:
            Dictionary with model, network, hardware, and simulation parameters
        """
        # Get model parameters
        din, dout, dmodel, dff, batch, seq, head, num_stacks = workload_generator.Model.get_model_params(
            workload_generator.Model(self.model_num)
        )
        
        metadata = {
            # Model information
            'model_name': self.model_name,
            'model_num': self.model_num,
            'vocab_size_in': din,
            'vocab_size_out': dout,
            'hidden_size': dmodel,
            'ffn_hidden_size': dff,
            'batch_size': batch[0] if isinstance(batch, list) else batch,
            'sequence_length': seq,
            'num_attention_heads': head,
            'num_layers': num_stacks,
            
            # Infrastructure
            'num_npus': self.num_npus,
            'sim_type': self.sim_type,
        }
        
        # Parse configuration files and add all parameters
        # Convert relative paths to absolute paths
        system_path = self.system_config if os.path.isabs(self.system_config) else os.path.join(self.base_dir, self.system_config)
        network_path = self.network_config if os.path.isabs(self.network_config) else os.path.join(self.base_dir, self.network_config)
        memory_path = self.memory_config if os.path.isabs(self.memory_config) else os.path.join(self.base_dir, self.memory_config)
        
        config_params = config_parser.parse_all_configs(
            system_path,
            network_path,
            memory_path
        )
        metadata.update(config_params)
        
        return metadata
    
    def _find_workload_file(self, config_name: str) -> Optional[str]:
        """
        Find generated workload file.
        
        Args:
            config_name: Configuration name (e.g., "4_8_2_1_1")
        
        Returns:
            Path to workload file without .0.et extension, or None if not found
        """
        import glob
        
        pattern = f"{self.workload_dir}/{config_name}.seq_*.batch_*.0.et"
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # Remove .0.et suffix
            return matching_files[0][:-5]
        
        return None
    
    def _run_astrasim(self, workload_path: str) -> str:
        """
        Run AstraSim simulation.
        
        Args:
            workload_path: Path to workload file
        
        Returns:
            Empty string if successful, error message otherwise
        """
        # Special handling for g2 backend
        if self.sim_type == "g2":
            g2_path = '/media/mohammad/extension/experiments/astra-sim/extern/network_backend/g2'
            if 'PYTHONPATH' in os.environ:
                os.environ['PYTHONPATH'] = f"{g2_path}:{os.environ['PYTHONPATH']}"
            else:
                os.environ['PYTHONPATH'] = g2_path
        
        # Use user-configured paths (works for both g2 and other sim types)
        return run_astrasim(
            workload_path=workload_path,
            system=self.system_config,
            network=self.network_config,
            memory=self.memory_config,
            output_dir=self.output_dir,
            network_log=self.network_log_dir,
            sim_type=self.sim_type
        )
    
    def _extract_execution_time(self, workload_file: str) -> Optional[float]:
        """
        Extract execution time from simulation log.
        
        Args:
            workload_file: Path to workload file (without extension)
        
        Returns:
            Execution time in seconds, or None if extraction failed
        """
        # Get workload filename
        workload_filename = workload_file.split('/')[-1]
        
        # Construct log file path
        log_file = f"{self.output_dir}/{workload_filename}.log"
        
        if not os.path.exists(log_file):
            if self.verbose:
                print(f"    ⚠️  Log file does not exist: {log_file}")
            return None
        
        # Extract time using output_parser
        exec_time = output_parser.extract_execution_time(log_file)
        
        return exec_time
    
    def batch_run(self, configs: list) -> list:
        """
        Run simulations for multiple configurations.
        
        Args:
            configs: List of configuration tuples
        
        Returns:
            List of execution times (None for failed runs)
        """
        results = []
        
        for i, config in enumerate(configs):
            if self.verbose:
                print(f"[{i+1}/{len(configs)}] ", end="")
            
            exec_time = self.run_simulation(config)
            results.append(exec_time)
        
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SimulationRunner(model={self.model_name}, "
                f"npus={self.num_npus}, network={self.network_name})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        return (f"SimulationRunner\n"
                f"  Model: {self.model_name} (#{self.model_num})\n"
                f"  NPUs: {self.num_npus}\n"
                f"  Network: {self.network_name}\n"
                f"  Sim type: {self.sim_type}\n"
                f"  Output: {self.output_dir}")
