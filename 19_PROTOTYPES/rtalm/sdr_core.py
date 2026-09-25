"""RT-ALM SDR core — compatibility shim.

The canonical SDR implementation lives in `sdr.py`.
This module re-exports it so `from sdr_core import SDR` keeps working.
"""
from .sdr import SDR

__all__ = ["SDR"]
