"""Model-free building blocks of the MissileNotes scan pipeline.

The end-to-end ``scan`` lives here for convenience. ``segment`` is still not
imported: it needs optional torch/litert runtimes and the mobile port only
pulls in the pieces it uses.
"""

from .pipeline import ScanResult, scan

__version__ = "0.1.0"

__all__ = ["marker", "geometry", "enhance", "scan", "ScanResult"]
