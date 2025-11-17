"""
VLM-FO1: Bridging the Gap Between High-Level Reasoning and Fine-Grained Perception
"""

try:
    from vlm_fo1._version import __version__
except ImportError:
    # Fallback for development installations
    __version__ = "0.1.0+dev"

__all__ = ["__version__"]
