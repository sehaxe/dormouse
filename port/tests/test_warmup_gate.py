"""The >100M-parameter zero-warmup gate (the 1B pilot opened at peak LR)."""
import pytest

from dormouse.config import MIN_WARMUP_PARAMS, validate_warmup
from dormouse.instrument import LoudError


def test_zero_warmup_refused_above_100m_params():
    with pytest.raises(LoudError, match="warmup_tokens=0"):
        validate_warmup(MIN_WARMUP_PARAMS + 1, 0)
    with pytest.raises(LoudError, match="46,700,000"):
        validate_warmup(1_135_023_553, 0)  # the real 1B parameter count


def test_the_cap_boundary_and_small_models_are_allowed():
    validate_warmup(MIN_WARMUP_PARAMS, 0)        # == cap, not above
    validate_warmup(5_000_000, 0)                # the tiny CPU test models
    validate_warmup(MIN_WARMUP_PARAMS + 1, 1)    # any positive warmup passes


def test_zero_warmup_still_legal_below_the_cap():
    validate_warmup(95_000_000, 0)
