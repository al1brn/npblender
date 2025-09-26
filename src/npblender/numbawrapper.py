# compat_numba.py
from __future__ import annotations

__all__ = ["njit", "prange", "NUMBA_AVAILABLE"]

from .deps import ensure_package

ensure_package("numba")

try:
    # If numba is avaialble, import the true functions

    from numba import njit as _real_njit, prange as _real_prange
    njit = _real_njit
    prange = _real_prange
    NUMBA_AVAILABLE = True

except Exception:
    # mockup functions
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """
        numba.njit mockup, compatible width :
          - @njit
          - @njit()
          - @njit(parallel=True, fastmath=True, cache=True, ...)
        """
        # Cas @njit sans parenthèses
        if args and callable(args[0]) and not kwargs:
            return args[0]

        # Cas @njit(...), on doit renvoyer un décorateur
        def _decorator(func):
            return func

        return _decorator

    def prange(*args):
        """
        Mock de numba.prange -> retombe sur range.
        Compatible avec prange(stop) ou prange(start, stop[, step]).
        """
        return range(*args)

