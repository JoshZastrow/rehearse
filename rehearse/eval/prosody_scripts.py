"""Fault behavior → Hume emotion trajectory mappings (tier-1 prosody).

Each FaultLabel maps to a scripted sequence of ProsodyScores values across
the synthetic user's utterances. E.g., FLAT_AFFECT → sustained low arousal
and low valence; ESCALATING_ANXIETY → rising arousal trajectory.

Trajectories are calibrated against tier-2 (real Hume) distributions on the
same behaviors; correlation is checked as part of the eval run.
"""
