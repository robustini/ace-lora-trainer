"""
Real-time GPU Monitoring for Training

Provides continuous VRAM tracking during training runs, with periodic logging
and alert thresholds. Useful for debugging OOM issues and understanding
memory patterns across epochs.

Inspired by Side-Step's gpu_utils.py.
"""

import threading
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from loguru import logger

import torch


@dataclass
class GPUSnapshot:
    """A single GPU memory snapshot."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_pct: float  # allocated / total


@dataclass
class GPUMonitor:
    """Monitors GPU memory usage during training.

    Can run in the background (threaded) or be polled manually.

    Usage:
        monitor = GPUMonitor(alert_threshold_pct=90.0)
        monitor.start()

        # ... training loop ...

        monitor.stop()
        summary = monitor.get_summary()
    """
    alert_threshold_pct: float = 90.0  # Alert when VRAM usage exceeds this %
    poll_interval_sec: float = 5.0     # How often to sample (seconds)
    device_index: int = 0              # Which GPU to monitor

    # Internal state
    _snapshots: List[GPUSnapshot] = field(default_factory=list, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    _peak_allocated_mb: float = field(default=0.0, repr=False)
    _alert_callback: Optional[Callable] = field(default=None, repr=False)
    _alert_fired: bool = field(default=False, repr=False)

    def snapshot(self) -> Optional[GPUSnapshot]:
        """Take a single GPU memory snapshot.

        Returns:
            GPUSnapshot or None if no GPU available.
        """
        if not torch.cuda.is_available():
            return None

        try:
            allocated = torch.cuda.memory_allocated(self.device_index) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device_index) / (1024 ** 2)
            total = torch.cuda.get_device_properties(self.device_index).total_memory / (1024 ** 2)
            free = total - reserved
            utilization = (allocated / total * 100) if total > 0 else 0.0

            snap = GPUSnapshot(
                timestamp=time.time(),
                allocated_mb=allocated,
                reserved_mb=reserved,
                free_mb=free,
                total_mb=total,
                utilization_pct=utilization,
            )

            self._snapshots.append(snap)

            if allocated > self._peak_allocated_mb:
                self._peak_allocated_mb = allocated

            # Fire alert if threshold exceeded
            if utilization > self.alert_threshold_pct and not self._alert_fired:
                self._alert_fired = True
                msg = (
                    f"⚠️ GPU VRAM alert: {allocated:.0f}MB / {total:.0f}MB "
                    f"({utilization:.1f}%) exceeds {self.alert_threshold_pct}% threshold"
                )
                logger.warning(msg)
                if self._alert_callback:
                    self._alert_callback(msg)
            elif utilization <= self.alert_threshold_pct * 0.9:
                # Reset alert when usage drops significantly
                self._alert_fired = False

            return snap

        except Exception as e:
            logger.debug(f"GPU snapshot failed: {e}")
            return None

    def start(self, alert_callback: Optional[Callable] = None):
        """Start background monitoring thread.

        Args:
            alert_callback: Optional function called when VRAM exceeds threshold.
                            Receives a single string argument (the alert message).
        """
        if self._running:
            return

        self._alert_callback = alert_callback
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"GPU monitor started (interval={self.poll_interval_sec}s, alert={self.alert_threshold_pct}%)")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.poll_interval_sec * 2)
            self._thread = None
        logger.info("GPU monitor stopped")

    def _monitor_loop(self):
        """Background polling loop."""
        while self._running:
            self.snapshot()
            time.sleep(self.poll_interval_sec)

    def get_current(self) -> Dict[str, float]:
        """Get current GPU memory status.

        Returns:
            Dict with allocated_mb, reserved_mb, free_mb, total_mb, utilization_pct.
        """
        snap = self.snapshot()
        if snap is None:
            return {
                "allocated_mb": 0, "reserved_mb": 0,
                "free_mb": 0, "total_mb": 0, "utilization_pct": 0,
            }
        return {
            "allocated_mb": snap.allocated_mb,
            "reserved_mb": snap.reserved_mb,
            "free_mb": snap.free_mb,
            "total_mb": snap.total_mb,
            "utilization_pct": snap.utilization_pct,
        }

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from all collected snapshots.

        Returns:
            Dict with peak, average, and current memory usage.
        """
        if not self._snapshots:
            return {
                "peak_allocated_mb": 0, "avg_allocated_mb": 0,
                "peak_reserved_mb": 0, "avg_utilization_pct": 0,
                "num_snapshots": 0,
            }

        allocated_values = [s.allocated_mb for s in self._snapshots]
        reserved_values = [s.reserved_mb for s in self._snapshots]
        utilization_values = [s.utilization_pct for s in self._snapshots]

        return {
            "peak_allocated_mb": max(allocated_values),
            "avg_allocated_mb": sum(allocated_values) / len(allocated_values),
            "peak_reserved_mb": max(reserved_values),
            "avg_utilization_pct": sum(utilization_values) / len(utilization_values),
            "num_snapshots": len(self._snapshots),
            "total_mb": self._snapshots[-1].total_mb if self._snapshots else 0,
        }

    def format_summary(self) -> str:
        """Get a formatted string summary of GPU usage.

        Returns:
            Human-readable summary string.
        """
        s = self.get_summary()
        if s["num_snapshots"] == 0:
            return "No GPU data collected"

        return (
            f"GPU Memory Summary ({s['num_snapshots']} samples):\n"
            f"  Peak allocated: {s['peak_allocated_mb']:.0f} MB / {s['total_mb']:.0f} MB "
            f"({s['peak_allocated_mb'] / s['total_mb'] * 100:.1f}%)\n"
            f"  Avg allocated:  {s['avg_allocated_mb']:.0f} MB "
            f"({s['avg_utilization_pct']:.1f}%)\n"
            f"  Peak reserved:  {s['peak_reserved_mb']:.0f} MB"
        )

    def reset(self):
        """Clear all collected snapshots."""
        self._snapshots.clear()
        self._peak_allocated_mb = 0.0
        self._alert_fired = False


def detect_gpu() -> Dict[str, str]:
    """Detect GPU information.

    Returns:
        Dict with name, compute_capability, driver_version, total_memory_gb.
    """
    info = {
        "name": "No GPU",
        "compute_capability": "N/A",
        "driver_version": "N/A",
        "total_memory_gb": "0",
        "backend": "cpu",
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["total_memory_gb"] = f"{props.total_memory / (1024**3):.1f}"
        info["backend"] = "cuda"
        try:
            info["driver_version"] = torch.version.cuda or "unknown"
        except Exception:
            pass
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        info["name"] = props.name
        info["total_memory_gb"] = f"{props.total_memory / (1024**3):.1f}"
        info["backend"] = "xpu"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["name"] = "Apple Silicon (MPS)"
        info["backend"] = "mps"
        # MPS doesn't expose memory info directly
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            total_bytes = int(result.stdout.strip())
            # Apple unified memory — report total system memory
            info["total_memory_gb"] = f"{total_bytes / (1024**3):.1f} (unified)"
        except Exception:
            info["total_memory_gb"] = "unknown (unified)"

    return info


def get_available_vram_mb() -> float:
    """Get available (free) VRAM in MB.

    Returns:
        Free VRAM in MB, or 0 if no GPU.
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return (total - reserved) / (1024 ** 2)
    except Exception:
        return 0.0
