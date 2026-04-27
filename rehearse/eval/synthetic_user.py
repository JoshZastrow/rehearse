"""Synthetic user agent.

Given a SyntheticUserProfile + an ExampleScenario, emits a paired stream of
TranscriptFrame and ProsodyFrame events that the pipeline consumes as if they
came from Hume. The agent's LLM brain chooses what to say (behaving per the
speaking_style and injected_faults); prosody_scripts.py maps behavior to
emotion trajectories on a per-utterance basis.

Must NOT leak the injected_faults label to the pipeline under test — the
coach must discover faults from transcript + prosody evidence alone.
"""
