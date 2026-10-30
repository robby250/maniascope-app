"""Shared chart-only geometry for difficulty and local recommendations (native)."""
import native_backend
N = native_backend.load()

STRUCTURE_VERSION = 2


def structure(chart, rate=1.0):
    """Small chart descriptors, not a second difficulty calculator."""
    return N.structure(chart, rate)
