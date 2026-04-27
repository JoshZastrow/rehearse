"""Post-call synthesis via Claude.

Two pure functions operating on frozen session artifacts:

  synthesize_story(intake, transcript_phase1) → story.md
      Narrative summary of the situation, for the user's later reference.

  synthesize_feedback(session) → feedback.md
      Grounded coach reflection over transcript + prosody + intake. Every
      claim must cite a turn or prosody moment.

Replayability is a hard requirement: given a frozen session directory,
`synthesize_feedback` must produce a new feedback artifact without touching
any other stage. This is both the architectural asset-sufficiency test and
the mechanism by which we re-score old sessions under new models.
"""
