"""Errors raised by the alpha human-review contract."""

from pear_sim.errors import PearSimError


class ReviewError(PearSimError):
    """Base class for review packet and decision failures."""

    code = "review.error"


class ReviewValidationError(ReviewError):
    """A review document does not conform to the alpha contract."""

    code = "review.invalid"


class ReviewIntegrityError(ReviewValidationError):
    """A review packet or annotation log failed an integrity check."""

    code = "review.integrity"


class StaleReviewError(ReviewIntegrityError):
    """A decision is bound to a different model, run, or packet."""

    code = "review.stale"


class ReviewDecisionError(ReviewValidationError):
    """A valid review decision does not authorize the requested action."""

    code = "review.decision_denied"
