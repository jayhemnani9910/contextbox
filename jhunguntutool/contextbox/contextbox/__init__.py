"""
ContextBox - A tool for capturing and organizing digital context.
"""

__version__ = "0.1.0"
__author__ = "ContextBox Team"

from .main import ContextBox
from .capture import ContextCapture
from .database import ContextDatabase
from .extractors import ContextExtractor

__all__ = [
    "ContextBox",
    "ContextCapture", 
    "ContextDatabase",
    "ContextExtractor"
]