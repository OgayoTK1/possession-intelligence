"""Ball-possession inference for More than a Hack 2027 (Phase 1).

Pipeline: tracking -> per-frame features -> per-frame state probabilities (LightGBM)
-> duration-aware HMM decoding -> frame,label CSV.
"""

__version__ = "0.1.0"
