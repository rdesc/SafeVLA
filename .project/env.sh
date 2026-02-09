#!/usr/bin/env bash
# env.sh — tracked defaults (NO secrets)
# Usage:
#   source .project/env.sh

set -a

# Resolve project root (repo root if you run from repo root)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Safe defaults
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Optional local overrides (NOT tracked)
# Put secrets/machine-specific stuff in .env.local
if [[ -f "$PROJECT_ROOT/.env.local" ]]; then
  source "$PROJECT_ROOT/.env.local"
fi

set +a