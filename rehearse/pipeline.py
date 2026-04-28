"""Runtime wiring entrypoint.

This module names the live-call assembly point for the owned runtime:

    TwilioStream
      → HumeEVIClient
      → FrameBus
      → PhaseProcessor
      → Artifact writers

Production and eval share artifact schemas, not transport infrastructure.
The live runtime owns Twilio/Hume integration; eval uses separate mocked or
provider-driven targets and does not depend on live call plumbing.
"""
