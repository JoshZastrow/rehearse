.PHONY: help init serve serve-memory setup setup-honcho setup-judge deploy-judge smoke-judge deploy-interactive smoke-interactive eval-list eval-voice-replay eval-voice-replay-live eval-voice-replay-dogfood eval-voice-smoke eval-voice-smoke-live eval-voice-rollout eval-voice-rollout-live eval-voice-rollout-audio eval-persona-routing eval-watch nightly-stability test lint rehearse-web _rehearse-web-agent _rehearse-web-app _livekit-server _token-server test-web test-livekit test-livekit-live

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-28s %s\n", $$1, $$2}'

init: ## interactive onboarding wizard — collect API keys + deploy infra (run once)
	uv run rehearse-init

serve: ## start ngrok + Honcho (auto-detected) + rehearse server
	@bash scripts/serve.sh

serve-memory: ## start the rehearse-memory MCP server (Honcho backend)
	uv run python3 -m rehearse.services.memory_mcp_server

setup: ## install deps + link .env (run once per worktree)
	uv sync
	@if [ ! -f .env ]; then \
	  MAIN_ROOT=$$(git worktree list | head -1 | awk '{print $$1}'); \
	  if [ -f "$$MAIN_ROOT/.env" ] && [ "$$(pwd)" != "$$MAIN_ROOT" ]; then \
	    ln -s "$$MAIN_ROOT/.env" .env && echo "Linked .env from $$MAIN_ROOT"; \
	  else \
	    cp .env.example .env && echo "Copied .env.example → .env  (fill in API keys)"; \
	  fi \
	fi

setup-honcho: ## clone + migrate Honcho for self-hosted local dev (no cloud API key needed)
	@if [ ! -d lib/honcho ]; then \
	  echo "Cloning Honcho into lib/honcho..."; \
	  git clone https://github.com/plastic-labs/honcho.git lib/honcho; \
	fi
	@echo "Installing Honcho dependencies..."
	cd lib/honcho && uv sync
	@echo "Writing lib/honcho/.env..."
	@printf 'DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5433/postgres\nAUTH_USE_AUTH=false\nSENTRY_ENABLED=false\n' > lib/honcho/.env
	@echo "Running Honcho migrations..."
	@uv run python3 scripts/pg0_server.py 5433 & \
	  PG0_PID=$$!; \
	  sleep 3; \
	  cd lib/honcho && DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5433/postgres uv run alembic upgrade head; \
	  kill $$PG0_PID 2>/dev/null || true
	@echo ""
	@echo "Done. Add to .env:"
	@echo "  HONCHO_BASE_URL=http://localhost:8001"
	@echo "Then 'make serve' will start Honcho automatically."

setup-judge: ## install inference backend CLI and authenticate (run once)
	uv pip install modal
	modal setup

deploy-judge: ## deploy LLM judge to inference backend (idempotent; safe to re-run)
	modal deploy infra/judge.py
	@echo ""
	@echo "Deployed. Add to .env:"
	@echo "  VLLM_BASE_URL=https://<workspace>--rehearse-gemma-judge-serve.modal.run/v1"
	@echo "  VLLM_API_KEY=<your-modal-token>"
	@echo ""
	@echo "Or get the URL automatically:"
	@echo "  modal app url rehearse-gemma-judge"

smoke-judge: ## run the smoke test against the deployed judge
	modal run infra/judge.py

deploy-interactive: ## deploy Moshi interactive inference server to Modal GPU (A10G)
	modal deploy infra/interactive.py
	@echo ""
	@echo "Deployed. Add to .env:"
	@echo "  INTERACTIVE_PROVIDER_ENDPOINT=wss://<workspace>--rehearse-interactive-providerserver-serve.modal.run/ws"
	@echo "  INTERACTIVE_CALLER_ENDPOINT=wss://<workspace>--rehearse-interactive-callerserver-serve.modal.run/ws"
	@echo ""
	@echo "Or get the URL automatically:"
	@echo "  modal app url rehearse-interactive"

smoke-interactive: ## run the smoke test against the deployed interactive server
	modal run infra/interactive.py

deploy-annotate: ## deploy session annotator (Whisper alignment) to Modal GPU (A10G)
	modal deploy train/pipeline/annotate.py
	@echo ""
	@echo "Deployed. No env vars required — called internally after each call."

eval-persona-routing: ## 3-scenario persona routing eval (requires Modal judge + Honcho)
	uv run pytest tests/eval/test_persona_voice_routing_eval.py \
	  -v -m "live_api and live_honcho" --timeout=180

clean: ## remove generated artifacts (venv, cache, sessions, runs)
	rm -rf .venv .cache sessions evals/runs evals/datasets/mme-emotion/v0-10clip/clips

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
	REHEARSE_AUDIO_JUDGE=live uv run rehearse-eval run --eval voice-rollout-judges --limit 2

eval-voice-rollout-audio: ## live-audio sandbox rollout through EVI (needs HUME_API_KEY + ANTHROPIC_API_KEY)
	uv run rehearse-eval run --eval voice-rollout-judges --environment live-audio-sandbox --limit 2

eval-watch: ## tail scores.jsonl for a run and render a live aggregate; usage: make eval-watch RUN=<run_id>
	@if [ -z "$(RUN)" ]; then echo "usage: make eval-watch RUN=<run_id>"; exit 1; fi
	uv run rehearse-eval watch --run-id $(RUN)

rehearse-web: ## start livekit-server + token server + agent + Vite dev server (WebRTC prototype)
	$(MAKE) -j4 _livekit-server _token-server _rehearse-web-agent _rehearse-web-app

_livekit-server: ## start livekit-server --dev on ws://localhost:7880 (install: brew install livekit)
	livekit-server --dev --bind 0.0.0.0

_token-server: ## start LiveKit JWT token server on http://localhost:8765
	uv run python web/livekit/token_server.py

_rehearse-web-agent:
	uv run python web/livekit/agent/agent.py

_rehearse-web-app:
	cd web/livekit/app && npm run dev

test-web: ## run hermetic LiveKit e2e tests (no external deps; auto-includes new hermetic tests)
	uv run pytest tests/test_livekit_e2e.py -v -m "not live_livekit and not live_modal"

test-livekit: ## run full live_livekit suite (requires livekit-server --dev running)
	uv run pytest tests/test_livekit_e2e.py -v -m live_livekit

test-livekit-live: ## run live_modal + live_livekit (requires both Modal endpoints + livekit-server)
	uv run pytest tests/test_livekit_e2e.py -v -m "live_modal and live_livekit"

nightly-stability: ## diagnostic stability run — voice-judges-smoke × repetitions=5 (Spec 8)
	uv run rehearse-eval run --eval voice-judges-smoke --repetitions 5

test: ## run the full pytest suite
	uv run pytest -q

lint: ## ruff check the repo
	uv run ruff check .
