"""Session curation gate between prepare_session_async and the Modal training volume.

A judge-backed reviewer reads each prepared session (audio_stereo.wav + transcript),
annotates a turning point for RL credit assignment, and approves or rejects it
against versioned selection criteria. Only approved sessions enter the curated
manifest pushed to the 'rehearse-training' volume. After each training run, the
feedback loop (SIA harness-update pattern) revises the criteria.
"""

from rehearse.train.curation.criteria import DEFAULT_CRITERIA, load_criteria, save_criteria
from rehearse.train.curation.curate import CurationResult, curate_sessions
from rehearse.train.curation.feedback import update_criteria
from rehearse.train.curation.reviewer import SessionReviewer
from rehearse.train.curation.types import SessionReview, TurningPoint

__all__ = [
    "DEFAULT_CRITERIA",
    "CurationResult",
    "SessionReview",
    "SessionReviewer",
    "TurningPoint",
    "curate_sessions",
    "load_criteria",
    "save_criteria",
    "update_criteria",
]
