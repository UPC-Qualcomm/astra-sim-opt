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
from typing import Optional, Dict
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
        'total_checked': 0,
        'total_killed': 0,
    }

    def __init__(
        self,
        initial_threshold: float = 1e15,
        kill_multiplier: float = 1.5,
        verbose: bool = False,
        tracker_id: Optional[str] = None
    ):
        """
        Initialize simulation tracker.
        
        Args:
            initial_threshold: Initial exec_time threshold (in cycles or seconds)
            kill_multiplier: Multiplier for early termination (default 1.5x)
            verbose: Print tracking information
            tracker_id: Optional shared id for tracker state file.
                       If not provided, a unique id is generated per run.
        """
        # Use file-based sharing for threshold across worker processes, but isolate each run.
        self._tracker_id = tracker_id or f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._threshold_file = os.path.join(
            tempfile.gettempdir(),
            f'astrasim_tracker_threshold_{self._tracker_id}.json'
        )

        self.kill_multiplier = kill_multiplier
        self.verbose = verbose

        # Initialize clean state for this run.
        self._initialize_state(initial_threshold)
        
        if self.verbose:
            print(f"[Tracker] Initialized with threshold={initial_threshold:.2e}, "
                  f"kill_multiplier={kill_multiplier}")

    def _default_state(self, threshold: Optional[float] = None) -> Dict:
        """Return a default tracker state dict."""
        state = self._DEFAULT_STATE.copy()
        if threshold is not None:
            state['threshold'] = threshold
        return state

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
            return self._default_state()
        return self._safe_read_json(self._threshold_file, self._default_state())

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
    
    def update_threshold(self, exec_time: float):
        """
        Update threshold with new best execution time.
        
        Uses file-based sharing so all worker processes see the update.
        
        Args:
            exec_time: Execution time from successful simulation (cycles)
        """
        current_threshold = self.threshold
        if exec_time < current_threshold:
            old_threshold = current_threshold
            self.threshold = exec_time
            
            if self.verbose:
                print(f"[Tracker] Threshold updated: {old_threshold:.2e} → {exec_time:.2e} cycles")
                print(f"          Kill threshold: {exec_time * self.kill_multiplier:.2e} cycles")
    
    def get_kill_threshold(self) -> float:
        """Get current kill threshold (threshold * multiplier)."""
        return self.threshold * self.kill_multiplier
    
    def should_kill_simulation(
        self,
        trace_file: str,
        workload_file: Optional[str] = None
    ) -> bool:
        """
        Check if simulation should be killed based on current progress.
        
        Monitors the trace CSV file for issue ticks and compares against threshold.
        
        Args:
            trace_file: Path to simulation trace CSV file
            workload_file: Optional path to workload file (for additional context)
        
        Returns:
            True if simulation should be killed, False otherwise
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
            kill_threshold = threshold * self.kill_multiplier
            state['total_checked'] = state.get('total_checked', 0) + 1

            # Debug output every check
            if self.verbose and state['total_checked'] % 100 == 0:
                print(f"[Tracker Debug] tick={latest_tick:.2e}, threshold={threshold:.2e}, kill_at={kill_threshold:.2e}")
            
            if latest_tick > kill_threshold:
                print(f"[Tracker] Killing simulation: {latest_tick:.2e} > {kill_threshold:.2e}")
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
            Dictionary with threshold, kill_threshold, and statistics
        """
        state = self._read_state()
        threshold = state.get('threshold', 1e15)
        return {
            'threshold': threshold,
            'kill_threshold': threshold * self.kill_multiplier,
            'kill_multiplier': self.kill_multiplier,
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
            print(f"[Tracker] Reset with threshold={threshold:.2e}")
    
    def __repr__(self) -> str:
        """String representation."""
        status = self.get_status()
        return (f"SimulationTracker(threshold={status['threshold']:.2e}, "
                f"kill_at={status['kill_threshold']:.2e})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        status = self.get_status()
        return (f"SimulationTracker\n"
                f"  Threshold: {status['threshold']:.2e}\n"
                f"  Kill at: {status['kill_threshold']:.2e}\n"
                f"  Multiplier: {status['kill_multiplier']}x\n"
                f"  Total checked: {status['total_checked']}\n"
                f"  Total killed: {status['total_killed']}")
