"""Post-call synthesis replayability tests.

Given a frozen session fixture, synthesize_feedback must produce a valid
markdown artifact whose citations all resolve to real utterances in the
fixture's transcript. Model calls are faked; the test asserts structure,
not content.
"""
