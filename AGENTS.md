# PrivFace: rules for AI assistants

## Project
Privacy-preserving face verification. Spec: docs/PLAN.pdf (sections referenced as §N).
Amendments that override the PDF: docs/SPEC_AMENDMENTS.md (read it every session).
Services: services/api (FastAPI core), services/ml (image analysis), services/vault (keys + IoM),
services/worker (ARQ cron), web (React + Vite + TS), libs/privface_core (shared crypto/IoM/metrics).

## Hard rules (never break)
1. Never persist or log images, frames, aligned crops, raw embeddings, plaintext IoM codes or keys.
2. IOM_MASTER_SECRET and TEMPLATE_ENC_KEY are read only inside services/vault.
3. Every internal endpoint (ml, vault) requires X-Internal-Token, checked with hmac.compare_digest.
4. Nonces are single-use (Redis GETDEL) with a 60 s TTL.
5. Do not add dependencies that are not listed in the task card without saying so explicitly.
6. Do not change files outside the task card's "Files" list unless strictly necessary; explain why.
7. Responses for unknown user, wrong face and liveness failure must have the same shape (SPEC_AMENDMENTS C2).

## Conventions
- Python 3.11, type hints everywhere, ruff for lint + format, pytest for tests.
- FastAPI: routers in app/routers, business logic in app/services, no logic in routers.
- Async everywhere in api and worker. CPU-bound work in ml runs in a thread pool.
- Errors: raise domain exceptions; map them to JSON {"error": "...", "reason": "..."} in app/errors.py.
- TypeScript strict mode; functional React components; Tailwind for styling; no localStorage for tokens.
- Each task: write or update tests first, implement, run `make lint test`, report what changed.

## Definition of done for any task
All acceptance checks in the task card pass, lint passes, no secrets in code, and a 3-line summary
of what changed and how it was tested.