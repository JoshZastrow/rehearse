.PHONY: help eval-list eval-voice-replay eval-voice-replay-live eval-voice-replay-dogfood eval-voice-smoke eval-voice-smoke-live eval-voice-rollout eval-voice-rollout-live eval-watch nightly-stability test lint

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-28s %s\n", $$1, $$2}'

eval-list: ## list evals, datasets, environments
	uv run rehearse-eval list-evals
	uv run rehearse-eval list-datasets
	uv run rehearse-eval list-environments

eval-voice-replay: ## score 3 production sessions with stub judges (free)
	uv run rehearse-eval run --eval production-voice-replay --limit 3

eval-voice-replay-live: ## score 3 production sessions with real Gemini judges (default: 2.5-flash; override REHEARSE_AUDIO_JUDGE_MODEL=gemini-2.5-pro for stronger judge)
	REHEARSE_AUDIO_JUDGE=live uv run rehearse-eval run --eval production-voice-replay --limit 3

eval-voice-replay-dogfood: ## score 3 sessions ignoring the consent gate (operator-only, never for training data)
	REHEARSE_REQUIRE_CONSENT=0 uv run rehearse-eval run --eval production-voice-replay --limit 3

eval-voice-replay-dogfood-live: ## dogfood + real Gemini judges
	REHEARSE_REQUIRE_CONSENT=0 REHEARSE_AUDIO_JUDGE=live \
		uv run rehearse-eval run --eval production-voice-replay --limit 3

eval-voice-smoke: ## run the fixture-audio smoke eval with stub judges
	uv run rehearse-eval run --eval voice-judges-smoke

eval-voice-smoke-live: ## fixture smoke with real TTS + Gemini judges (needs HUME_API_KEY + GEMINI_API_KEY)
	REHEARSE_AUDIO_JUDGE=live uv run rehearse-eval run --eval voice-judges-smoke

eval-voice-rollout: ## runtime-sandbox rollout with stub TTS (needs ANTHROPIC_API_KEY only)
	uv run rehearse-eval run --eval voice-rollout-judges --limit 3

eval-voice-rollout-live: ## runtime-sandbox rollout with real Hume TTS + audio judges (needs HUME_API_KEY + ANTHROPIC_API_KEY + GOOGLE_API_KEY)
	REHEARSE_AUDIO_JUDGE=live uv run rehearse-eval run --eval voice-rollout-judges --limit 3

eval-watch: ## tail scores.jsonl for a run and render a live aggregate; usage: make eval-watch RUN=<run_id>
	@if [ -z "$(RUN)" ]; then echo "usage: make eval-watch RUN=<run_id>"; exit 1; fi
	uv run rehearse-eval watch --run-id $(RUN)

nightly-stability: ## diagnostic stability run — voice-judges-smoke × repetitions=5 (Spec 8)
	uv run rehearse-eval run --eval voice-judges-smoke --repetitions 5

test: ## run the full pytest suite
	uv run pytest -q

lint: ## ruff check the repo
	uv run ruff check .
