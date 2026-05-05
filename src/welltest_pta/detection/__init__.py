"""Event detection submodule — wraps V8.1 spike-boundary detector."""

from welltest_pta.detection.detector import (
    EventDetector,
    EventDetectorConfig,
    detect_events,
)

__all__ = ["EventDetector", "EventDetectorConfig", "detect_events"]
