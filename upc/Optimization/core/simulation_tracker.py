"""
SimulationTracker: Monitor simulation progress and kill slow runs early.

This module provides real-time tracking of simulation execution by monitoring
issue ticks in trace/log files. It implements an adaptive threshold mechanism
to kill simulations that exceed 1.5x the current best execution time.

Features:
- Real-time log file monitoring
- Adaptive threshold management
- Early termination of slow simulations
- Shared state across parallel evaluations
"""

import os
import math
from typing import Optional, Dict, Any, List
import tempfile
import json
import uuid


class SimulationTracker:
    """
    Tracks simulation progress and terminates slow runs early.
    
    The tracker monitors issue ticks in simulation log files and compares
    them against a threshold. If a simulation's ticks exceed 1.5x the
    threshold, it's terminated early to save time.
    
    Uses a small JSON file in /tmp for shared threshold and counters
    across worker processes.
    
    Usage:
        tracker = SimulationTracker(initial_threshold=1e15)
        
        # During simulation
        if tracker.should_kill_simulation(log_file):
            # Kill simulation
            
        # After successful completion
        tracker.update_threshold(exec_time)
    """
    
    _DEFAULT_STATE = {
        'threshold': 1e15,
        'latest_tick': None,
        'total_checked': 0,
        'total_killed': 0,
    }

    def __init__(
        self,
        initial_threshold: float = 1e15,
        kill_multiplier: float = 1.5,
        verbose: bool = False,
        tracker_id: Optional[str] = None,
        minimize: bool = True,
        objective: Optional[Any] = None,
    ):
        """
        Initialize simulation tracker.

        Args:
            initial_threshold: Magnitude of the worst-case score threshold.
                The tracker converts this to the correct sign per objective
                direction automatically (positive for minimize, negative for
                maximize).  Pass a positive value; direction is handled here.
            kill_multiplier: Kill simulations exceeding threshold × multiplier
                (minimize) or below threshold / multiplier (maximize).
            verbose: Print tracking information.
            tracker_id: Optional shared id for the state file across worker
                processes.  A unique id is generated per run when omitted.
            minimize: Fallback primary direction used only when no objective
                is provided.  When ``objective`` is given, direction is derived
                from ``objective.score_directions`` and this parameter is ignored.
            objective: Optional objective function.  When provided, per-objective
                directions are taken from ``objective.score_directions`` so that
                mixed-direction MOO (e.g. minimize time + maximize throughput)
                is handled correctly.
        """
        self.kill_multiplier = kill_multiplier
        self.verbose = verbose
        self.objective = objective

        # ── Derive per-objective directions ──────────────────────────────────
        # Always read from the objective when available so that mixed-direction
        # MOO objectives are handled without the caller needing to pass anything.
        if objective is not None and hasattr(objective, 'score_directions'):
            directions = objective.score_directions
        else:
            directions = [bool(minimize)]

        # Primary direction (used as scalar fallback throughout the class).
        self.minimize = directions[0]

        # ── Build correct initial threshold per direction ──────────────────
        # For minimize: worst case is +inf  → start at  +initial_threshold.
        # For maximize: worst case is -inf  → start at  -initial_threshold.
        # This prevents spurious kills on maximize objectives whose real scores
        # (e.g. log10(throughput) ≈ 3) are far below a naïve +1e15 initial.
        magnitude = abs(float(initial_threshold))
        self._initial_threshold = magnitude
        if len(directions) > 1:
            threshold_init = [magnitude if d else -magnitude for d in directions]
        else:
            threshold_init = magnitude if directions[0] else -magnitude

        # ── File-based shared state (across worker processes) ─────────────
        self._tracker_id = tracker_id or f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._threshold_file = os.path.join(
            tempfile.gettempdir(),
            f'astrasim_tracker_threshold_{self._tracker_id}.json'
        )
        self._initialize_state(threshold_init)

        if self.verbose:
            if len(directions) == 1:
                dir_str = "minimize" if directions[0] else "maximize"
            else:
                dir_str = "[" + ", ".join("min" if d else "max" for d in directions) + "]"
            print(
                f"[Tracker] Initialized with threshold={self._format_threshold(threshold_init)}, "
                f"kill_multiplier={kill_multiplier}, directions={dir_str}"
            )

    def _default_state(self, threshold: Optional[float] = None) -> Dict:
        """Return a default tracker state dict."""
        state = self._DEFAULT_STATE.copy()
        if threshold is not None:
            state['threshold'] = threshold
        return state

    @staticmethod
    def _format_threshold(value: Any) -> str:
        """Format scalar/list threshold for logs."""
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(f"{float(v):.2e}" for v in value) + "]"
        return f"{float(value):.2e}"

    def _write_state(self, data: Dict):
        """Write full tracker state to shared file (atomic)."""
        dir_name = os.path.dirname(self._threshold_file)
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp_f:
            tmp_path = tmp_f.name
            json.dump(data, tmp_f)
        os.replace(tmp_path, self._threshold_file)

    def _read_state(self) -> Dict:
        """Read current tracker state from shared file."""
        if not os.path.exists(self._threshold_file):
            return self._default_state(self._initial_threshold)
        return self._safe_read_json(self._threshold_file, self._default_state(self._initial_threshold))

    def _initialize_state(self, initial_threshold: float):
        """Create a fresh shared state file for this tracker run."""
        data = self._default_state(initial_threshold)
        try:
            self._write_state(data)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not initialize tracker state: {e}")
    
    def _safe_read_json(self, path: str, default: dict) -> dict:
        """Read a JSON file, returning 'default' if the file is empty or corrupt."""
        try:
            with open(path, 'r') as f:
                content = f.read()
            if not content.strip():
                return default
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return default
        except Exception:
            return default

    @property
    def threshold(self):
        """Get current threshold value from shared file."""
        try:
            return self._read_state().get('threshold', 1e15)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not read threshold: {e}")
            return 1e15
    
    @threshold.setter
    def threshold(self, value):
        """Set threshold value in shared file."""
        try:
            data = self._read_state()
            data['threshold'] = value
            self._write_state(data)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not write threshold: {e}")
    
    def _get_counter(self, counter_name: str) -> int:
        """Get a counter value from the shared file."""
        try:
            data = self._read_state()
            return int(data.get(counter_name, 0))
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error reading {counter_name}: {e}")
            return 0
    
    def update_threshold(self, score: Any):
        """
        Update threshold with new best score.
        
        Uses file-based sharing so all worker processes see the update.
        
        Args:
            score: Score from successful simulation.
        """
        score_vector = self._reduce_score(score)
        if score_vector is None:
            return

        current_threshold = self.threshold
        threshold_vector = self._to_vector(current_threshold, len(score_vector))
        directions = self._objective_directions(len(score_vector))

        old_threshold = list(threshold_vector)
        updated = False
        for idx, (value, old_value) in enumerate(zip(score_vector, threshold_vector)):
            if directions[idx]:
                if value < old_value:
                    threshold_vector[idx] = value
                    updated = True
            else:
                if value > old_value:
                    threshold_vector[idx] = value
                    updated = True

        if updated:
            self.threshold = threshold_vector[0] if len(threshold_vector) == 1 else threshold_vector

            if self.verbose:
                print(
                    f"[Tracker] Threshold updated: {self._format_threshold(old_threshold)} "
                    f"→ {self._format_threshold(threshold_vector)}"
                )
                print(
                    f"          Kill threshold: "
                    f"{self._format_threshold(self.get_kill_score_threshold())}"
                )
    
    def get_kill_score_threshold(self) -> Any:
        """
        Get current kill threshold in score-space.

        Minimize direction: kill if score > threshold * multiplier.
        Maximize direction: kill if score < threshold / multiplier.

        Returns a list for MOO objectives, a scalar for single-objective.
        """
        threshold = self.threshold
        if isinstance(threshold, (list, tuple)):
            directions = self._objective_directions(len(threshold))
            return [
                float(v) * self.kill_multiplier if is_min else float(v) / self.kill_multiplier
                for v, is_min in zip(threshold, directions)
            ]

        # Scalar threshold: use the primary direction.
        is_min = self._objective_directions(1)[0]
        if is_min:
            return float(threshold) * self.kill_multiplier
        return float(threshold) / self.kill_multiplier

    def get_kill_threshold(self) -> float:
        """
        Backward-compatible alias for score-space kill threshold.
        """
        return self.get_kill_score_threshold()

    def get_latest_tick_threshold(self) -> float:
        """
        Return last observed trace tick used by the tracker.

        Falls back to current score threshold for compatibility.
        """
        state = self._read_state()
        latest_tick = state.get('latest_tick')
        if latest_tick is None:
            threshold = self.threshold
            if isinstance(threshold, (list, tuple)):
                return float(threshold[0]) if threshold else 0.0
            return float(threshold)
        try:
            return float(latest_tick)
        except (TypeError, ValueError):
            threshold = self.threshold
            if isinstance(threshold, (list, tuple)):
                return float(threshold[0]) if threshold else 0.0
            return float(threshold)

    def _to_vector(self, value: Any, target_len: int) -> List[float]:
        """Normalize scalar/list value to a float vector of target length."""
        if isinstance(value, (tuple, list)):
            vec = [float(v) for v in value]
            if len(vec) == target_len:
                return vec
            if len(vec) == 1:
                return vec * target_len
            if len(vec) > target_len:
                return vec[:target_len]
            # If shorter than target_len, pad by repeating the last value.
            return vec + [vec[-1]] * (target_len - len(vec))
        return [float(value)] * target_len

    def _objective_directions(self, n_objectives: int) -> List[bool]:
        """
        Return per-objective optimization direction.

        True means minimize, False means maximize.

        Delegates to ``objective.score_directions`` when an objective is
        available so that all direction logic stays in one place.
        """
        if self.objective is None:
            return [self.minimize] * n_objectives

        # Use the canonical score_directions property when available.
        if hasattr(self.objective, 'score_directions'):
            directions = self.objective.score_directions
        else:
            raw = getattr(self.objective, "objective_directions", None)
            if raw is None:
                return [self.minimize] * n_objectives
            directions = []
            for d in raw:
                if isinstance(d, str):
                    directions.append(d.strip().lower() != "max")
                else:
                    directions.append(bool(d))

        if len(directions) != n_objectives:
            if self.verbose:
                print(
                    f"[Tracker] Warning: objective_directions length {len(directions)} "
                    f"!= {n_objectives}, using global minimize={self.minimize}"
                )
            return [self.minimize] * n_objectives
        return directions

    # Magnitude threshold matching objective.py PENALTY (1e20).
    # Any score component with abs(value) >= this is treated as invalid.
    _PENALTY_MAGNITUDE = 1e20

    def _reduce_score(self, score: Any) -> Optional[List[float]]:
        """Convert scalar/tuple score into finite, non-penalty float vector (strict, for completed runs)."""
        if score is None:
            return None
        raw_values = list(score) if isinstance(score, (tuple, list)) else [score]
        if not raw_values:
            return None

        values: List[float] = []
        for item in raw_values:
            try:
                value = float(item)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or abs(value) >= self._PENALTY_MAGNITUDE:
                return None
            values.append(value)
        return values

    @staticmethod
    def _tracker_partial_estimate_vector(raw: Any) -> Optional[List[float]]:
        """
        Parse ``objective.compute`` output for mid-run tracking.

        Components that cannot be evaluated yet (non-finite: ``inf``, ``nan``,
        or unparseable) become ``nan`` placeholders.  At least one finite
        component is required; otherwise returns ``None``.
        """
        if raw is None:
            return None
        _pen = SimulationTracker._PENALTY_MAGNITUDE

        if isinstance(raw, (tuple, list)):
            out: List[float] = []
            for item in raw:
                try:
                    v = float(item)
                except (TypeError, ValueError):
                    out.append(float("nan"))
                    continue
                # Non-finite or penalty-magnitude values are unavailable mid-run.
                out.append(v if (math.isfinite(v) and abs(v) < _pen) else float("nan"))
            if not out or not any(math.isfinite(x) for x in out):
                return None
            return out
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v) or abs(v) >= _pen:
            return None
        return [v]

    def _estimate_running_score(
        self,
        latest_tick: float,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[float]]:
        """
        Estimate objective score for an in-flight simulation.

        Uses ``objective.compute`` with the current tick as ``exec_time``.  For
        multi-objective functions, metrics that are not available until the run
        finishes (e.g. peak memory, power) often yield non-finite components;
        those are marked ``nan`` and omitted from the kill decision.  Kill uses
        only objectives with finite partial estimates; when all components are
        finite (e.g. latency + bandwidth from tick + config), the rule matches
        full MOO: kill only if **every** objective is past its kill bound.
        """
        if self.objective is not None:
            try:
                estimated = self.objective.compute(
                    exec_time=latest_tick,
                    is_oom=False,
                    metadata=metadata or {},
                    config=config,
                )
                return self._tracker_partial_estimate_vector(estimated)
            except Exception as e:
                if self.verbose:
                    print(f"[Tracker] Warning: objective-based score estimate failed: {e}")
                return None

        # Fallback: use raw tick as score proxy.
        return self._reduce_score(latest_tick)
    
    def should_kill_simulation(
        self,
        trace_file: str,
        workload_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if simulation should be killed based on current progress.

        Monitors the trace CSV for issue ticks, builds a partial objective vector
        via ``objective.compute(exec_time=tick, ...)``, and compares finite
        components to per-objective kill bounds.  Components that are not
        finite (unavailable mid-run, e.g. power or peak memory) are skipped.
        The run is killed only when **every available** objective is worse than
        its threshold (minimize: above kill line; maximize: below kill line).
        """
        _ = workload_file  # Reserved for future context-specific logic.

        # Check if trace file exists
        if not os.path.exists(trace_file):
            return False
        
        try:
            # Get latest issue tick from trace file
            latest_tick = self._get_latest_issue_tick(trace_file)
            #print(f"Trace file: {trace_file}, Latest tick: {latest_tick}")
            if latest_tick is None:
                return False
            
            # Check against kill threshold and update stats
            state = self._read_state()
            threshold = state.get('threshold', 1e15)
            kill_threshold = self.get_kill_score_threshold()
            state['total_checked'] = state.get('total_checked', 0) + 1
            state['latest_tick'] = latest_tick
            estimated_score = self._estimate_running_score(
                latest_tick=latest_tick,
                config=config,
                metadata=metadata,
            )

            # Debug output every check
            if self.verbose and state['total_checked'] % 100 == 0:
                print(
                    f"[Tracker Debug] tick={latest_tick:.2e}, estimated_score={estimated_score}, "
                    f"threshold={self._format_threshold(threshold)}, "
                    f"kill_at={self._format_threshold(kill_threshold)}"
                )

            if estimated_score is None:
                self._write_state(state)
                return False

            kill_threshold_vector = self._to_vector(kill_threshold, len(estimated_score))
            directions = self._objective_directions(len(estimated_score))
            # Only evaluate objectives with a finite partial estimate.  Metrics
            # that need a finished simulation (power, peak memory, etc.) often
            # return non-finite values mid-run — skip those axes until available.
            # Kill iff every *available* objective is past its kill bound.  When
            # all objectives are available (e.g. time + config-derived BW), this
            # matches full MOO: all must be bad to kill.
            _pen = SimulationTracker._PENALTY_MAGNITUDE
            bad_flags: List[bool] = []
            for score_val, kill_val, is_min in zip(
                estimated_score, kill_threshold_vector, directions
            ):
                # Skip unavailable (nan) or penalty-magnitude components.
                if not math.isfinite(score_val) or abs(score_val) >= _pen:
                    continue
                bad_flags.append(
                    (score_val > kill_val) if is_min else (score_val < kill_val)
                )
            should_kill = bool(bad_flags) and all(bad_flags)

            if should_kill:
                print(
                    f"[Tracker] Killing simulation: score={self._format_threshold(estimated_score)} "
                    f"vs kill_at={self._format_threshold(kill_threshold)}"
                )
                state['total_killed'] = state.get('total_killed', 0) + 1
                self._write_state(state)
                # Always print kill messages (not just in verbose mode)
                #print(f"    Current tick: {latest_tick:.2e} > Kill threshold: {kill_threshold:.2e}")
                if self.verbose:
                    print(f"    Trace file: {os.path.basename(trace_file)}")
                return True
            
            self._write_state(state)
            
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error checking simulation: {e}")
            return False
    
    def _get_latest_issue_tick(self, trace_file: str) -> Optional[float]:
        """
        Extract the latest issue tick from trace CSV file.
        
        Reads the trace file and finds the maximum tick value from issue records.
        The last issue tick represents the current progress of the simulation.
        
        Args:
            trace_file: Path to trace CSV file (e.g., workload_trace.csv)
        
        Returns:
            Latest tick value, or None if not found
        """
        try:
            # Read last chunk for efficiency (avoid reading entire file)
            max_tick = None
            
            # Use tail-like approach: read last chunk
            with open(trace_file, 'r') as f:
                # Seek to end and read backwards
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # Read last chunk (up to 500KB for CSV which can have longer lines)
                chunk_size = min(500000, file_size)
                f.seek(max(0, file_size - chunk_size))
                
                lines = f.readlines()
                
                # Parse tick values from CSV
                # CSV format: [timestamp] [I] <trace>: ,action,sys_id,node_id,node_name,col_type,node_type,num_ops,tensor_size,perf,operational_intensity,issue_tick
                # issue_tick is the LAST column (12 total columns with leading comma)
                for line in lines:
                    # Skip header or empty lines
                    if not line.strip() or 'action,sys_id' in line or line.startswith('#'):
                        continue
                    
                    try:
                        # Remove timestamp prefix if present: [timestamp] [I] <trace>:
                        if '<trace>:' in line:
                            line = line.split('<trace>:', 1)[1]
                        
                        # Split CSV and extract issue_tick (last column)
                        parts = line.strip().split(',')
                        if len(parts) >= 12:
                            tick = float(parts[-1])  # Last column is issue_tick
                            
                            if max_tick is None or tick > max_tick:
                                max_tick = tick
                    except (ValueError, IndexError):
                        continue
            
            return max_tick
            
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error reading trace file {trace_file}: {e}")
            return None
    
    def get_status(self) -> Dict:
        """
        Get tracker status information.

        Returns:
            Dictionary with threshold, kill_threshold, per-objective directions,
            and running statistics.
        """
        state = self._read_state()
        threshold = state.get('threshold', 1e15)
        n = len(threshold) if isinstance(threshold, (list, tuple)) else 1
        directions = self._objective_directions(n)
        return {
            'threshold': threshold,
            'kill_threshold': self.get_kill_score_threshold(),
            'kill_multiplier': self.kill_multiplier,
            'score_directions': directions,
            'latest_tick': state.get('latest_tick'),
            'total_checked': int(state.get('total_checked', 0)),
            'total_killed': int(state.get('total_killed', 0))
        }
    
    def reset(self, initial_threshold: Optional[float] = None):
        """
        Reset tracker to initial state.
        
        Args:
            initial_threshold: New initial threshold (uses current if None)
        """
        threshold = self.threshold if initial_threshold is None else initial_threshold
        self._initialize_state(threshold)

        if self.verbose:
            print(f"[Tracker] Reset with threshold={self._format_threshold(threshold)}")
    
    def __repr__(self) -> str:
        """String representation."""
        status = self.get_status()
        return (
            f"SimulationTracker(threshold={self._format_threshold(status['threshold'])}, "
            f"kill_at={self._format_threshold(status['kill_threshold'])})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        status = self.get_status()
        directions = status['score_directions']
        if len(directions) == 1:
            dir_str = "minimize" if directions[0] else "maximize"
        else:
            dir_str = "[" + ", ".join("min" if d else "max" for d in directions) + "]"
        return (f"SimulationTracker\n"
                f"  Threshold: {self._format_threshold(status['threshold'])}\n"
                f"  Kill at: {self._format_threshold(status['kill_threshold'])}\n"
                f"  Directions: {dir_str}\n"
                f"  Multiplier: {status['kill_multiplier']}x\n"
                f"  Total checked: {status['total_checked']}\n"
                f"  Total killed: {status['total_killed']}")
