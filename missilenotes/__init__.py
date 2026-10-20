"""Model-free building blocks of the MissileNotes scan pipeline.

Submodules are intentionally not imported here: ``segment`` needs optional
torch weights and the mobile port only pulls in the pieces it uses.
"""

__version__ = "0.1.0"

__all__ = ["marker", "geometry", "enhance"]
