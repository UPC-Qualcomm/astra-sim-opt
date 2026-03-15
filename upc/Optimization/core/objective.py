"""
Objective function module for flexible optimization.

This module provides a modular system for defining optimization objectives.
- Use built-in objectives (time, memory, energy, throughput, etc.)
- Create weighted multi-objective optimizations
- Define completely custom objective functions

Example:
    # Default: minimize execution time
    objective = MinimizeExecutionTime()
    
    # Weighted multi-objective
    objective = create_objective(
        "weighted",
        weights={"exec_time": 0.7, "peak_memory_bytes": 0.3}
    )
    
    # Use in optimizer
    optimizer = ScikitBayesianOptimizer(..., objective=objective)
"""
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

PENALTY = float('inf')

class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.
    
    An objective function computes a scalar score from simulation results.
    """
    
    def __init__(self, name: Optional[str] = None, minimize: bool = True):
        """
        Initialize objective function.
        
        Args:
            name: Human-readable name for this objective
            minimize: Whether to minimize (True) or maximize (False) the score
        """
        self.name = name or self.__class__.__name__
        self.minimize = minimize
    
    @abstractmethod
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute objective score from simulation results.
        
        Args:
            exec_time: Execution time in seconds (from simulation)
            metadata: Additional simulation metadata (model params, hardware config, etc.)
            config: Configuration dictionary with hardware/network parameters (e.g., npu_count, local_mem_bw, etc.)
        
        Returns:
            Scalar score to optimize
        """
        pass
    
    def is_better(self, score1, score2) -> bool:
        """
        Check if score1 is better than score2.
        
        Args:
            score1: First score (float or tuple for multi-objective)
            score2: Second score (float or tuple for multi-objective)
        
        Returns:
            True if score1 is better than score2
        
        Note:
            For multi-objective (tuples), uses lexicographic comparison:
            compares first element, then second if equal, etc.
        """
        # Handle None/infinity cases
        if score2 is None or score2 == float('inf'):
            return score1 is not None and score1 != float('inf')
        if score1 is None or score1 == float('inf'):
            return False
            
        # Handle tuple comparison for multi-objective
        if isinstance(score1, tuple) and isinstance(score2, tuple):
            if self.minimize:
                return score1 < score2  # Lexicographic comparison
            else:
                return score1 > score2
        
        # Handle single value comparison
        if self.minimize:
            return score1 < score2
        else:
            return score1 > score2
    
    def get_best_score(self, scores: list) -> float:
        """
        Get the best score from a list.
        
        Args:
            scores: List of scores
        
        Returns:
            Best score (minimum if minimize=True, maximum if minimize=False)
        """
        if not scores:
            return PENALTY if self.minimize else -PENALTY
        return min(scores) if self.minimize else max(scores)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.name


class MinimizeExecutionTime(ObjectiveFunction):
    """
    Minimize execution time (default objective).
    
    Simply returns the execution time as the objective score.
    This is the standard objective for performance optimization.
    """
    
    def __init__(self):
        super().__init__("Minimize Execution Time")
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """Return execution time as objective."""
        if is_oom:
            return PENALTY
        return exec_time
    


class MinimizeExecutionTimeAndNetworkBW(ObjectiveFunction):
    """
    Minimize execution time and the total network bandwidth.
    
    """
    
    def __init__(self):
        super().__init__("Minimize Execution Time and Network Bandwidth Perf per BW/NPU")
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute objective based on COSMIC paper formula.
        
        Formula: reward = 1 / sqrt(power(sim_time * sum(network_bw) - 1, 2))
        where network_bw includes both intra-node and inter-node bandwidths."""
        if is_oom:
            return PENALTY
        
        if config is None:
            # Fallback to execution time only if no config provided
            return exec_time
        
        # Extract configuration parameters
        npu_count = config.get('npu_count', 1)
        intra_node_bw = config.get('intra-node-bw', 0)  # GB/s
        inter_node_bw = config.get('inter-node-bw', 0)  # GB/s
        npus_per_node = 8  # Default: 8 NPUs per node
        
        # Calculate number of nodes
        num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)  # Ceiling division
        
        # Calculate total network bandwidth
        # Intra-node: bandwidth within each node (connections between NPUs in same node)
        # Inter-node: bandwidth between nodes
        if num_nodes == 1:
            # Single node: only intra-node bandwidth matters
            total_network_bw = intra_node_bw * (npu_count - 1)  # Connections between NPUs
        else:
            # Multiple nodes: both intra-node and inter-node bandwidth
            intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
            inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
            total_network_bw = intra_bw_total + inter_bw_total
        
        # Avoid division by zero or negative values
        if total_network_bw <= 0 or exec_time <= 0:
            return PENALTY
        
        # COSMIC formula: reward = 1 / sqrt((sim_time * sum(network_bw) - 1)^2)
        obj = exec_time * total_network_bw
        
        denominator = math.sqrt((exec_time * total_network_bw - 1 ) ** 2)
                
        reward = 1.0 / denominator
        
        return reward

class MinimizePower(ObjectiveFunction):
    """
    Minimize total power consumption (Mode D: Full LPM, Energy-Proportional).

    Reads ``total_power_W`` injected into metadata by
    :meth:`SimulationRunner._run_power_estimation`.  Only meaningful for g2
    simulations; returns ``PENALTY`` if the metric is absent.
    """

    def __init__(self):
        super().__init__("Minimize Total Power (W) [Mode D]")

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        if is_oom:
            return PENALTY
        total_power_W = metadata.get('total_power_W')
        if total_power_W is None:
            return PENALTY
        return float(total_power_W)


class MinimizeEnergy(ObjectiveFunction):
    """
    Minimize total energy consumption (Mode D: Full LPM, Energy-Proportional).

    Reads ``total_energy_J`` injected into metadata by
    :meth:`SimulationRunner._run_power_estimation`.  Only meaningful for g2
    simulations; returns ``PENALTY`` if the metric is absent.
    """

    def __init__(self):
        super().__init__("Minimize Total Energy (J) [Mode D]")

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        if is_oom:
            return PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None:
            return PENALTY
        return float(total_energy_J)


class MinimizePowerAndTime(ObjectiveFunction):
    """
    Multi-objective: jointly minimize total power (W) and execution time.

    Both objectives are log10-transformed so their scales are comparable
    regardless of absolute magnitude.

    Returns:
        tuple: ``(log10(total_power_W), log10(exec_time))`` — both minimized.
    """

    def __init__(self):
        super().__init__("Minimize Power (W) and Execution Time [Mode D, MOO]")
        self.is_multi_objective = True

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_power_W = metadata.get('total_power_W')
        if total_power_W is None or exec_time is None:
            return PENALTY, PENALTY
        return math.log10(max(1.0, total_power_W)), math.log10(max(1.0, exec_time))


class MinimizeEnergyAndTime(ObjectiveFunction):
    """
    Multi-objective: jointly minimize total energy (J) and execution time.

    Both objectives are log10-transformed so their scales are comparable
    regardless of absolute magnitude.

    Returns:
        tuple: ``(log10(total_energy_J), log10(exec_time))`` — both minimized.
    """

    def __init__(self):
        super().__init__("Minimize Energy (J) and Execution Time [Mode D, MOO]")
        self.is_multi_objective = True

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY
        return math.log10(max(1.0, total_energy_J)), math.log10(max(1.0, exec_time))


class MinimizeEDP(ObjectiveFunction):
    """
    Minimize the Energy-Delay Product (EDP).

    EDP = total_energy_J  ×  exec_cycles

    where ``exec_cycles`` is the raw wall-clock time returned by AstraSim
    (``exec_time`` in the optimizer pipeline — the parser reads the
    ``Wall time: N`` log line directly in cycles).

    **Why EDP?**

    Optimizing energy alone can produce designs that are very slow (e.g. low
    bandwidth saves switching power but hurts throughput).  Optimizing cycles
    alone ignores power entirely.  EDP combines both with equal weight:

    * Halving execution time  → EDP halves  (same as halving energy).
    * Doubling energy         → EDP doubles (same as doubling delay).

    This naturally steers the search toward the *knee* of the
    energy-performance Pareto front — configurations that are neither
    power-wasteful nor pathologically slow.

    **Requires** ``estimate_power=1`` in ``net_sim_config`` so that
    ``total_energy_J`` is populated in metadata by the power estimator.
    Returns ``PENALTY`` if the metric is absent (e.g. simulation was killed
    or power estimation was disabled).

    The raw product is log10-transformed before being returned so that
    the surrogate model works with values of manageable magnitude
    (``log10(EDP) ~ log10(energy_J) + log10(cycles)`` ≈ 6-12 + 9-12 → 15-24).
    """

    def __init__(self):
        super().__init__("Minimize EDP (Energy × Cycles) [Mode D]")

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        
        if is_oom:
            return PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY
        edp = total_energy_J * exec_time
        return math.log10(max(1.0, edp))


def _compute_total_network_bw(config: Dict[str, Any], npus_per_node: int = 8) -> float:
    """Compute aggregate network bandwidth (GB/s) from optimization config."""
    npu_count     = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    num_nodes     = max(1, (npu_count + npus_per_node - 1) // npus_per_node)

    if num_nodes == 1:
        return intra_node_bw * (npu_count - 1)

    return (intra_node_bw * npus_per_node * num_nodes
            + inter_node_bw * (num_nodes - 1))


class MinimizeEDPAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: minimize EDP **and** total network bandwidth jointly.

    Objective 0: ``log10(total_energy_J × exec_cycles)``  — minimize EDP.
    Objective 1: ``log10(total_network_bw_GBps)``          — minimize network cost.

    **Why add network BW as a second objective?**

    EDP alone can be improved by throwing unlimited bandwidth at a workload
    (more links → faster execution → lower EDP even if energy rises).  Making
    total network BW an explicit second objective prevents that shortcut and
    forces the optimizer to find configurations that are simultaneously
    energy-delay efficient *and* network-frugal — important when provisioning
    cost or physical link count is a concern.

    Total network BW is derived from the configuration parameters:

    * Single-node cluster: ``intra_node_bw × (npu_count − 1)``
    * Multi-node cluster:  ``intra_node_bw × npus_per_node × num_nodes
                             + inter_node_bw × (num_nodes − 1)``

    Both objectives are log10-transformed for comparable scale.

    **Requires** ``estimate_power=1`` in ``net_sim_config``.
    Returns ``(PENALTY, PENALTY)`` if EDP metrics are absent or OOM.
    """

    def __init__(self, npus_per_node: int = 8):
        """
        Args:
            npus_per_node: Number of NPUs per physical node (default: 8).
                           Used to split ``npu_count`` into intra / inter
                           bandwidth contributions.
        """
        super().__init__("Minimize EDP and Network BW [Mode D, MOO]")
        self.is_multi_objective = True
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY

        # EDP objective
        edp = total_energy_J * exec_time
        log_edp = math.log10(max(1.0, edp))

        # Network BW objective (derived from config)
        if config is None:
            return log_edp, PENALTY
        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)

        log_bw = math.log10(max(1.0, total_network_bw))
        return log_edp, log_bw


class MinimizeED2PAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: minimize ED²P (delay-sensitive) and network bandwidth.

    Objective 0: ``log10(E) + 2*log10(D)``  (ED²P)
    Objective 1: ``log10(total_network_bw_GBps)``

    This objective is useful when delay sensitivity is important while still
    controlling network over-provisioning.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize ED²P and Network BW [Mode D, MOO]")
        self.is_multi_objective = True
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY

        log_ed2p = (math.log10(max(1.0, total_energy_J))
                    + 2.0 * math.log10(max(1.0, exec_time)))

        if config is None:
            return log_ed2p, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_ed2p, log_bw

class MinimizeE2DAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: minimize E²D (energy-sensitive) and network bandwidth.

    Objective 0: ``2*log10(E) + log10(D)``  (E²D)
    Objective 1: ``log10(total_network_bw_GBps)``

    This objective is useful when energy sensitivity is important while still
    controlling network over-provisioning.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize E²D and Network BW [Mode D, MOO]")
        self.is_multi_objective = True
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY

        log_e2d = (2.0 * math.log10(max(1.0, total_energy_J))
                   + math.log10(max(1.0, exec_time)))

        if config is None:
            return log_e2d, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_e2d, log_bw


class MinimizeEnergyCyclesAndNetworkBW(ObjectiveFunction):
    """
    Three-objective optimization: minimize energy, cycles, and network BW.

    Returns:
        tuple: ``(log10(E), log10(D), log10(BW))``
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Energy, Cycles, and Network BW [Mode D, 3-MOO]")
        self.is_multi_objective = True
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY, PENALTY

        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY, PENALTY

        log_energy = math.log10(max(1.0, total_energy_J))
        log_cycles = math.log10(max(1.0, exec_time))

        if config is None:
            return log_energy, log_cycles, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_energy, log_cycles, log_bw


class MinimizePowerCyclesAndNetworkBW(ObjectiveFunction):
    """
    Three-objective optimization: minimize power, cycles, and network BW.

    Returns:
        tuple: ``(log10(P), log10(D), log10(BW))``
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Power, Cycles, and Network BW [Mode D, 3-MOO]")
        self.is_multi_objective = True
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY, PENALTY

        total_power_W = metadata.get('total_power_W')
        if total_power_W is None or exec_time is None:
            return PENALTY, PENALTY, PENALTY

        log_power = math.log10(max(1.0, total_power_W))
        log_cycles = math.log10(max(1.0, exec_time))

        if config is None:
            return log_power, log_cycles, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_power, log_cycles, log_bw


class MinimizeWeightedEDP(ObjectiveFunction):
    r"""
    Minimize the generalised Energy-Delay product  E^alpha × D^beta.

    Plain EDP (alpha=beta=1) is **symmetric**: doubling energy and halving
    delay leaves the score unchanged because the two effects cancel exactly.
    This means the optimizer cannot distinguish between:

    * A fast, power-hungry configuration  (low D, high E)
    * A slow, energy-sipping configuration (high D, low E)

    Choosing alpha ≠ beta **breaks the symmetry** and encodes a preference:

    +---------+---------+---------------------------------------------+
    | alpha   | beta    | Interpretation                              |
    +=========+=========+=============================================+
    | 1       | 2       | ED²P — penalises delay twice as much        |
    |         |         | as energy.  Performance-oriented.           |
    |         |         | Doubling D costs 2× what doubling E costs.  |
    +---------+---------+---------------------------------------------+
    | 2       | 1       | E²D — penalises energy twice as much        |
    |         |         | as delay.  Efficiency-oriented.             |
    |         |         | Doubling E costs 2× what doubling D costs.  |
    +---------+---------+---------------------------------------------+
    | 1       | 1       | Classic EDP (symmetric).                    |
    +---------+---------+---------------------------------------------+

    Under log10-transformation the score becomes::

        alpha * log10(E) + beta * log10(D)

    so the optimizer sees a linear combination of log-energy and log-delay
    with the chosen weights — no cancellation is possible when alpha ≠ beta.

    **Requires** ``estimate_power=1`` in ``net_sim_config``.
    Returns ``PENALTY`` if energy metric is absent or OOM.

    Args:
        alpha: Exponent on energy   (default 1.0).
        beta:  Exponent on delay     (default 2.0 → ED²P).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        name = f"Minimize E^{alpha}×D^{beta} "
        if alpha == 1 and beta == 2:
            name += "(ED²P, performance-oriented) [Mode D]"
        elif alpha == 2 and beta == 1:
            name += "(E²D, efficiency-oriented) [Mode D]"
        else:
            name += "(Weighted EDP) [Mode D]"
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        
        if is_oom:
            return PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY
        # alpha*log(E) + beta*log(D)  ==  log(E^alpha * D^beta)
        return (self.alpha * math.log10(max(1.0, total_energy_J))
                + self.beta  * math.log10(max(1.0, exec_time)))


class MinimizeEnergyAndCycles(ObjectiveFunction):
    """
    Multi-objective: expose energy (J) and execution cycles as **independent**
    objectives.

    This is the *true* fix for the EDP symmetry problem.  Any scalar
    combination of E and D (EDP, ED²P, E²D …) collapses the 2-D Pareto front
    into a single number, which means the optimizer can always trade one for
    the other along a level-set without any penalty.  Treating them as
    separate objectives forces the surrogate model and acquisition function to
    explore the *entire* front and lets the user inspect the trade-off
    post-hoc.

    Objective 0: ``log10(total_energy_J)``  — minimize energy.
    Objective 1: ``log10(exec_cycles)``     — minimize delay.

    Compared to :class:`MinimizeEnergyAndTime`, which was introduced together
    with the power-model objectives, this class is semantically identical but
    uses the same ``exec_time`` (cycles) units throughout, making it the
    natural power-model counterpart to :class:`MinimizeEDP`.

    **Requires** ``estimate_power=1`` in ``net_sim_config``.
    Returns ``(PENALTY, PENALTY)`` if energy metric is absent or OOM.
    """

    def __init__(self):
        super().__init__("Minimize Energy (J) and Cycles independently [Mode D, MOO]")
        self.is_multi_objective = True

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY
        return math.log10(max(1.0, total_energy_J)), math.log10(max(1.0, exec_time))


class WeightedMultiObjective(ObjectiveFunction):
    """
    Weighted combination of multiple objectives.
    
    Combines multiple objectives with configurable weights.
    Useful for multi-objective optimization.
    
    Example:
        # 70% time, 20% memory, 10% energy
        objective = WeightedMultiObjective({
            'exec_time': 0.7,
            'peak_memory_bytes': 0.2,
            'energy_joules': 0.1
        })
    """
    
    def __init__(self, weights: Dict[str, float], normalize: bool = True):
        """
        Initialize weighted multi-objective.
        
        Args:
            weights: Dictionary mapping metric names to weights
                    Special key 'exec_time' uses execution time
                    Other keys should be in metadata
            normalize: Whether to normalize metrics before combining
        """
        super().__init__("Weighted Multi-Objective")
        self.weights = weights
        self.normalize = normalize
        
        # Track metric ranges for normalization
        self.metric_mins: Dict[str, float] = {}
        self.metric_maxs: Dict[str, float] = {}
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """Return weighted combination of metrics."""
        if is_oom:
            return PENALTY
        # Collect all metrics TODO: Not complete list
        metrics = {'exec_time': exec_time}
        metrics.update(metadata)
        
        # Update normalization ranges
        if self.normalize:
            for metric_name in self.weights.keys():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if metric_name not in self.metric_mins:
                        self.metric_mins[metric_name] = value
                        self.metric_maxs[metric_name] = value
                    else:
                        self.metric_mins[metric_name] = min(self.metric_mins[metric_name], value)
                        self.metric_maxs[metric_name] = max(self.metric_maxs[metric_name], value)
        
        # Compute weighted sum
        weighted_sum = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name not in metrics:
                raise ValueError(f"Metric '{metric_name}' not found in results")
            
            value = metrics[metric_name]
            
            # Normalize if enabled
            if self.normalize:
                min_val = self.metric_mins[metric_name]
                max_val = self.metric_maxs[metric_name]
                if max_val > min_val:
                    value = (value - min_val) / (max_val - min_val)
                else:
                    value = 0.0
            
            weighted_sum += weight * value
        
        return weighted_sum



class CustomObjective(ObjectiveFunction):
    """
    Custom objective function from user-provided callable.
    
    Allows defining arbitrary objective functions.
    
    Example:
        # Minimize time + 0.1 * sqrt(memory_gb)
        def my_objective(exec_time, metadata):
            memory_gb = metadata['peak_memory_bytes'] / (1024**3)
            return exec_time + 0.1 * (memory_gb ** 0.5)
        
        objective = CustomObjective(my_objective, "My Custom Objective")
    """
    
    def __init__(
        self, 
        compute_fn: Callable[[float, Dict[str, Any]], float], 
        name: str = "Custom Objective",
        minimize: bool = True,
        is_multi_objective: bool = False
    ):
        """
        Initialize custom objective.
        
        Args:
            compute_fn: Callable that takes (exec_time, is_oom, metadata, config) and returns score or tuple of scores
            name: Name for this objective
            minimize: Whether to minimize (True) or maximize (False)
            is_multi_objective: Whether this objective returns multiple values (tuple)
        """
        super().__init__(name, minimize)
        self.compute_fn = compute_fn
        self.is_multi_objective = is_multi_objective
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Call user-provided compute function with all parameters including config."""
        return self.compute_fn(exec_time, is_oom, metadata, config)


# Convenience factory function
def create_objective(objective_type: str, **kwargs) -> ObjectiveFunction:
    """
    Factory function to create objective functions by name.
    
    Args:
        objective_type: Type of objective ['time', 'time_and_network_bw', 'weighted', 'custom']
        **kwargs: Additional arguments for the objective
    
    Returns:
        ObjectiveFunction instance
    
    Example:
        objective = create_objective('time')
        objective = create_objective('weighted', weights={'exec_time': 0.7, 'peak_memory_bytes': 0.3})
    """
    objectives = {
        'time': MinimizeExecutionTime,
        'time_and_network_bw':  MinimizeExecutionTimeAndNetworkBW,
        'power': MinimizePower,
        'energy': MinimizeEnergy,
        'power_and_time': MinimizePowerAndTime,
        'energy_and_time': MinimizeEnergyAndTime,
        'edp': MinimizeEDP,
        'edp_and_network_bw': MinimizeEDPAndNetworkBW,
        'ed2p_and_network_bw': MinimizeED2PAndNetworkBW,
        'e2d_and_network_bw': MinimizeE2DAndNetworkBW,
        'energy_cycles_and_network_bw': MinimizeEnergyCyclesAndNetworkBW,
        'power_cycles_network_bw': MinimizePowerCyclesAndNetworkBW,
        'weighted_edp': MinimizeWeightedEDP,          # E^alpha x D^beta, default ED2P
        'ed2p': lambda: MinimizeWeightedEDP(1, 2),    # performance-oriented shortcut
        'e2d':  lambda: MinimizeWeightedEDP(2, 1),    # efficiency-oriented shortcut
        'energy_and_cycles': MinimizeEnergyAndCycles, # true MOO, no symmetry issue
        'weighted': WeightedMultiObjective,
        'custom': CustomObjective
    }
    
    if objective_type not in objectives:
        raise ValueError(f"Unknown objective type: {objective_type}. "
                        f"Available: {list(objectives.keys())}")
    
    return objectives[objective_type](**kwargs)
