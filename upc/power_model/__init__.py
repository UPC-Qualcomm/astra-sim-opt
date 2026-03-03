"""
Power Model Package

4-mode power model for AstraSim workload analysis.
"""

from .power_config import PowerConfig
from .compute_parser import parse_astrasim_log, ComputeStats
from .network_parser import parse_network_statistics_csv, NetworkStats
from .compute_model import ComputeModel
from .network_model import NetworkModel
from .power_model import PowerModel, compare_all_modes

__version__ = '1.0.0'
__all__ = [
    'PowerConfig',
    'ComputeStats',
    'NetworkStats',
    'parse_astrasim_log',
    'parse_network_statistics_csv',
    'ComputeModel',
    'NetworkModel',
    'PowerModel',
    'compare_all_modes'
]
