"""Diagnose Twilio number configuration.

Usage:
    uv run python scripts/diagnose_number.py
    uv run python scripts/diagnose_number.py --call-me +15551234567

Three checks:
  1. Account info — confirm we're hitting the right Twilio account.
  2. Number config — fetch the phone number resource and print its voice/SMS URLs.
  3. Optional outbound smoke test — if --call-me is given, place an outbound
     call FROM the Twilio number TO that number, hitting our /twilio/voice
     handler. Proves the number is functional and our webhook is reachable
     end-to-end. Isolates inbound-routing issues from everything else.
"""

from __future__ import annotations

import argparse
import sys

from twilio.rest import Client

from rehearse.config import RuntimeConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--call-me",
        metavar="E164_NUMBER",
        help="Place a test outbound call to this number (e.g. +15551234567).",
    )
    args = parser.parse_args()

    config = RuntimeConfig.from_env()
    client = Client(config.twilio_account_sid, config.twilio_auth_token)

    print("=" * 60)
    print("1. ACCOUNT")
    print("=" * 60)
    account = client.api.accounts(config.twilio_account_sid).fetch()
    print(f"  friendly_name: {account.friendly_name}")
    print(f"  status:        {account.status}")
    print(f"  type:          {account.type}")
    print(f"  sid:           {account.sid}")

    print()
    print("=" * 60)
    print("2. PHONE NUMBERS OWNED BY THIS ACCOUNT")
    print("=" * 60)
    numbers = client.incoming_phone_numbers.list()
    if not numbers:
        print("  (none — buy a number first)")
        return 1

    target = None
    for num in numbers:
        match = "  "
        if num.phone_number == config.twilio_from_number:
            match = "→ "
            target = num
        print(f"{match}{num.phone_number}  ({num.friendly_name})")
        print(f"    sid:           {num.sid}")
        print(f"    voice_url:     {num.voice_url or '(none)'}")
        print(f"    voice_method:  {num.voice_method}")
        print(f"    status_cb_url: {num.status_callback or '(none)'}")
        print(f"    sms_url:       {num.sms_url or '(none)'}")
        print(f"    capabilities:  voice={num.capabilities.get('voice')} "
              f"sms={num.capabilities.get('sms')}")

    if target is None:
        print()
        print(f"  ⚠ TWILIO_PHONE_NUMBER={config.twilio_from_number} "
              f"is NOT in this account's number list.")
        return 1

    expected_voice_url = f"{config.public_base_url}/twilio/voice/inbound"
    print()
    print(f"  Expected voice_url: {expected_voice_url}")
    if target.voice_url == expected_voice_url:
        print("  ✓ voice_url matches")
    else:
        print("  ⚠ voice_url does NOT match — fix in Twilio console.")

    if args.call_me:
        print()
        print("=" * 60)
        print("3. OUTBOUND TEST CALL")
        print("=" * 60)
        print(f"  Calling {args.call_me} from {config.twilio_from_number}")
        print(f"  Twilio will fetch TwiML from: "
              f"{config.public_base_url}/twilio/voice?session_id=diagnostic")
        call = client.calls.create(
            to=args.call_me,
            from_=config.twilio_from_number,
            url=f"{config.public_base_url}/twilio/voice?session_id=diagnostic",
        )
        print(f"  call_sid: {call.sid}")
        print("  → answer the call. If you hear 'Hello. This is rehearse...',")
        print("    the number works and your webhook is reachable. The")
        print("    inbound-call problem is then isolated to upstream routing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
