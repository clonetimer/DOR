"""Dataset/device adapters.

This package intentionally imports no concrete bridge at module import time.
Use ``isgn.get_adapter(name)`` or import a concrete module such as
``isgn.bridges.ehss`` explicitly.  This keeps the ISGN core device-agnostic.
"""
__all__: list[str] = []
