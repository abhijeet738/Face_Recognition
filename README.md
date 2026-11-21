# PrivFace

A web app that verifies people by face from a phone or laptop camera, with liveness detection, and never stores the face itself. It stores only a cancelable, key-protected IoM-GRP template that can be revoked and re-issued like a password.

**Status:** in development (week 1 of 12). The build plan is in [docs/PLAN.pdf](docs/PLAN.pdf) and its amendments in [docs/SPEC_AMENDMENTS.md](docs/SPEC_AMENDMENTS.md).

## Quick start (development)

```bash
make setup      # venv, tools, git hooks (Python 3.11 required)
make lint test
```

## Licence

Code: Apache-2.0 (see [LICENSE](LICENSE)). The InsightFace `buffalo_l` models are licensed for non-commercial use only, so this project is a portfolio and research demo and is not for commercial deployment. Model weights are never committed to this repository.