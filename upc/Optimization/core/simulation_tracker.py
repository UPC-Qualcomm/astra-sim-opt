"""
SimulationTracker: Monitor simulation progress and kill slow runs early.

This module provides real-time tracking of simulation execution by monitoring
issue ticks in trace/log files. It implements an adaptive threshold mechanism
to kill simulations that exceed 1.5x the current best execution time.

Features:
- Real-time log file monitoring
- Adaptive threshold management
- Early termination of slow simulations
- Thread-safe tracking across parallel evaluations
"""

import os
import time
import threading
from typing import Optional, Dict
from pathlib import Path
import tempfile
import json


class SimulationTracker:
    """
    Tracks simulation progress and terminates slow runs early.
    
    The tracker monitors issue ticks in simulation log files and compares
    them against a threshold. If a simulation's ticks exceed 1.5x the
    threshold, it's terminated early to save time.
    
    Uses multiprocessing.Value for shared threshold across worker processes.
    
    Usage:
        tracker = SimulationTracker(initial_threshold=1e15)
        
        # During simulation
        if tracker.should_kill_simulation(log_file):
            # Kill simulation
            
        # After successful completion
        tracker.update_threshold(exec_time)
    """
    
    def __init__(
        self,
        initial_threshold: float = 1e15,
        kill_multiplier: float = 1.5,
        check_interval: float = 1.0,
        verbose: bool = False
    ):
        """
        Initialize simulation tracker.
        
        Args:
            initial_threshold: Initial exec_time threshold (in cycles or seconds)
            kill_multiplier: Multiplier for early termination (default 1.5x)
            check_interval: How often to check log files (seconds)
            verbose: Print tracking information
        """
        # Use file-based sharing for threshold across processes
        self._threshold_file = os.path.join(tempfile.gettempdir(), 'astrasim_tracker_threshold.json')
        self._write_threshold(initial_threshold)
        
        self.kill_multiplier = kill_multiplier
        self.check_interval = check_interval
        self.verbose = verbose
        
        # Thread safety for parallel evaluations
        self._lock = None
        
        # Track statistics
        self.total_checked = 0
        self.total_killed = 0
        
        if self.verbose:
            print(f"[Tracker] Initialized with threshold={initial_threshold:.2e}, "
                  f"kill_multiplier={kill_multiplier}")
    
    def _write_threshold(self, value: float):
        """Write threshold to shared file (atomic write to avoid race conditions)."""
        try:
            # Read existing data
            data = {'threshold': 1e15, 'total_checked': 0, 'total_killed': 0}
            if os.path.exists(self._threshold_file):
                data = self._safe_read_json(self._threshold_file, data)

            # Update threshold
            data['threshold'] = value

            # Atomic write: write to a temp file then rename so readers never
            # see a partially-written (empty) file.
            dir_name = os.path.dirname(self._threshold_file)
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp_f:
                tmp_path = tmp_f.name
                json.dump(data, tmp_f)
            os.replace(tmp_path, self._threshold_file)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not write threshold: {e}")
    
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

    def _read_threshold(self) -> float:
        """Read threshold from shared file."""
        try:
            if os.path.exists(self._threshold_file):
                data = self._safe_read_json(self._threshold_file,
                                            {'threshold': 1e15, 'total_checked': 0, 'total_killed': 0})
                return data.get('threshold', 1e15)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not read threshold: {e}")
        return 1e15
    
    @property
    def threshold(self):
        """Get current threshold value from shared file."""
        return self._read_threshold()
    
    @threshold.setter
    def threshold(self, value):
        """Set threshold value in shared file."""
        self._write_threshold(value)
    
    def _increment_counter(self, counter_name: str):
        """Increment a counter in the shared file (atomic write to avoid race conditions)."""
        try:
            # Read existing data
            default = {'threshold': 1e15, 'total_checked': 0, 'total_killed': 0}
            data = default.copy()
            if os.path.exists(self._threshold_file):
                data = self._safe_read_json(self._threshold_file, default)

            # Increment counter
            data[counter_name] = data.get(counter_name, 0) + 1

            # Atomic write
            dir_name = os.path.dirname(self._threshold_file)
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp_f:
                tmp_path = tmp_f.name
                json.dump(data, tmp_f)
            os.replace(tmp_path, self._threshold_file)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error incrementing {counter_name}: {e}")
    
    def _get_counter(self, counter_name: str) -> int:
        """Get a counter value from the shared file."""
        try:
            if os.path.exists(self._threshold_file):
                data = self._safe_read_json(self._threshold_file,
                                            {'threshold': 1e15, 'total_checked': 0, 'total_killed': 0})
                return data.get(counter_name, 0)
            return 0
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error reading {counter_name}: {e}")
            return 0
    
    @property
    def lock(self):
        """Lazy initialization of lock to support pickling."""
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock
    
    def __getstate__(self):
        """Prepare tracker for pickling (exclude lock)."""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        state['_lock'] = None
        return state
    
    def __setstate__(self, state):
        """Restore tracker after unpickling (recreate lock)."""
        self.__dict__.update(state)
        # Lock will be lazily recreated when needed
        
        # Initialize tracking statistics if they don't exist (for backward compatibility)
        if not hasattr(self, 'total_checked'):
            self.total_checked = 0
        if not hasattr(self, 'total_killed'):
            self.total_killed = 0
        if not hasattr(self, '_threshold_file'):
            self._threshold_file = os.path.join(tempfile.gettempdir(), 'astrasim_tracker_threshold.json')
    
    def update_threshold(self, exec_time: float):
        """
        Update threshold with new best execution time.
        
        Uses file-based sharing so all worker processes see the update.
        
        Args:
            exec_time: Execution time from successful simulation (cycles)
        """
        current_threshold = self._read_threshold()
        if exec_time < current_threshold:
            old_threshold = current_threshold
            self._write_threshold(exec_time)
            
            if self.verbose:
                print(f"[Tracker] Threshold updated: {old_threshold:.2e} → {exec_time:.2e} cycles")
                print(f"          Kill threshold: {exec_time * self.kill_multiplier:.2e} cycles")
    
    def should_kill_by_walltime(self, elapsed_walltime: float) -> bool:
        """
        Check if simulation should be killed based on elapsed wall clock time.
        
        Uses calibrated cycles-per-second from previous successful runs to
        estimate how long a simulation should take.
        
        Args:
            elapsed_walltime: Elapsed wall clock time in seconds
        
        Returns:
            True if simulation should be killed, False otherwise
        """
        with self.lock:
            # If threshold is still at initial value, don't kill anything yet
            if self.threshold >= 1e14:  # Very high initial threshold
                return False
            
            # Need calibration samples first
            if self.cycles_per_second is None or len(self.walltime_samples) < self.min_samples_for_walltime:
                return False
            
            # Calculate expected wall clock time based on threshold and calibration
            expected_walltime = self.threshold / self.cycles_per_second
            kill_walltime_threshold = expected_walltime * self.kill_multiplier
            
            self.total_checked += 1
            
            if elapsed_walltime > kill_walltime_threshold:
                self.total_killed += 1
                # Always print kill messages
                #print(f"\n⚠️  [TRACKER] KILLING SIMULATION #{self.total_killed}")
                #print(f"    Elapsed: {elapsed_walltime:.1f}s > Kill threshold: {kill_walltime_threshold:.1f}s")
                #print(f"    (Based on best: {self.threshold:.2e} cycles @ {self.cycles_per_second:.2e} cyc/s × {self.kill_multiplier})")
                return True
            
            return False
    
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
        # Check if trace file exists
        if not os.path.exists(trace_file):
            return False
        
        try:
            # Get latest issue tick from trace file
            latest_tick = self._get_latest_issue_tick(trace_file)
            #print(f"Trace file: {trace_file}, Latest tick: {latest_tick}")
            if latest_tick is None:
                return False
            
            # Check against kill threshold
            kill_threshold = self.get_kill_threshold()
            #print("kill threshold", kill_threshold)
            # Debug output every check
            if self.verbose and self._get_counter('total_checked') % 10 == 0:
                print(f"[Tracker Debug] tick={latest_tick:.2e}, threshold={self.threshold:.2e}, kill_at={kill_threshold:.2e}")
            
            if latest_tick > kill_threshold:
                self._increment_counter('total_killed')
                killed_count = self._get_counter('total_killed')
                # Always print kill messages (not just in verbose mode)
                #print(f"\n⚠️  [TRACKER] KILLING SIMULATION #{killed_count}")
                #print(f"    Current tick: {latest_tick:.2e} > Kill threshold: {kill_threshold:.2e}")
                #print(f"    Best threshold: {self.threshold:.2e} × {self.kill_multiplier} = {kill_threshold:.2e}")
                if self.verbose:
                    print(f"    Trace file: {os.path.basename(trace_file)}")
                return True
            
            # Track check count
            self._increment_counter('total_checked')
            
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
    
    def start_monitoring(
        self,
        log_file: str,
        kill_callback,
        check_interval: Optional[float] = None
    ) -> threading.Thread:
        """
        Start background thread to monitor simulation progress.
        
        Args:
            log_file: Path to log file to monitor
            kill_callback: Function to call when simulation should be killed
            check_interval: How often to check (uses self.check_interval if None)
        
        Returns:
            Thread object (already started)
        """
        interval = check_interval if check_interval is not None else self.check_interval
        
        def monitor():
            while True:
                if not os.path.exists(log_file):
                    time.sleep(interval)
                    continue
                
                if self.should_kill_simulation(log_file):
                    kill_callback()
                    break
                
                # Check if simulation finished (look for SUMMARY line)
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if '[SUMMARY]' in content and 'finished' in content:
                            break
                except:
                    pass
                
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread
    
    def get_status(self) -> Dict:
        """
        Get tracker status information.
        
        Returns:
            Dictionary with threshold, kill_threshold, and statistics
        """
        return {
            'threshold': self.threshold,
            'kill_threshold': self.threshold * self.kill_multiplier,
            'kill_multiplier': self.kill_multiplier,
            'total_checked': self._get_counter('total_checked'),
            'total_killed': self._get_counter('total_killed')
        }
    
    def reset(self, initial_threshold: Optional[float] = None):
        """
        Reset tracker to initial state.
        
        Args:
            initial_threshold: New initial threshold (uses current if None)
        """
        if initial_threshold is not None:
            with self._shared_threshold.get_lock():
                self._shared_threshold.value = initial_threshold
            
            if self.verbose:
                print(f"[Tracker] Reset with threshold={self.threshold:.2e}")
    
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
