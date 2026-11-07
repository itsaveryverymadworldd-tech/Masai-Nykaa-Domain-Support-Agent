#!/usr/bin/env bash
#
# Pre-upload verification gate.
#
# Run this before you publish the repository. It answers one question:
# "would a grader who clones this and runs it get everything working?"
#
#   ./preflight.sh
#
# Three stages:
#   A. Repository hygiene   -- things that break on someone else's machine
#   B. Cold-state rebuild   -- wipe all generated state, run every script
#   C. Acceptance criteria  -- the 29 brief criteria, checked against artifacts
#
# Exit code 0 = safe to upload. Non-zero = do not upload yet.

set -uo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

PASS=0
FAIL=0
FAILURES=()

ok()  { printf "  \033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS + 1)); }
bad() { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL + 1)); FAILURES+=("$1"); }
hdr() { printf "\n\033[1m%s\033[0m\n%s\n" "$1" "$(printf '%.0s-' {1..74})"; }

# --------------------------------------------------------------------------- #
hdr "A. REPOSITORY HYGIENE  (things that break on another machine)"
# --------------------------------------------------------------------------- #

# A1. No hardcoded absolute paths in code or docs (transcripts may echo $PWD).
HITS=$(git ls-files '*.py' '*.md' '*.toml' '*.sh' 2>/dev/null \
       | xargs grep -l "$HOME" 2>/dev/null || true)
if [ -z "$HITS" ]; then
  ok "no hardcoded absolute home paths in code or docs"
else
  bad "hardcoded absolute paths in: $(echo "$HITS" | tr '\n' ' ')"
fi

# A2. No secrets, env files or key material tracked.
SECRETS=$(git ls-files | grep -Ei '\.env|secret|credential|\.pem$|\.key$|id_rsa' || true)
if [ -z "$SECRETS" ]; then
  ok "no .env / secret / key files tracked"
else
  bad "possible secret files tracked: $(echo "$SECRETS" | tr '\n' ' ')"
fi

# A3. Virtualenv and generated state must not be committed.
GENERATED=$(git ls-files | grep -E '^\.venv/|^data/|__pycache__' || true)
if [ -z "$GENERATED" ]; then
  ok "no .venv / data/ / __pycache__ committed"
else
  bad "generated files committed: $(echo "$GENERATED" | tr '\n' ' ' | head -c 200)"
fi

# A4. No oversized blobs (GitHub warns above 50 MB, blocks above 100 MB).
BIG=$(git ls-files -z | xargs -0 du -m 2>/dev/null | awk '$1 > 5 {print $2" ("$1"MB)"}')
if [ -z "$BIG" ]; then
  ok "no tracked file over 5 MB"
else
  bad "large files tracked: $(echo "$BIG" | tr '\n' ' ')"
fi

# A5. No binary media anywhere (the brief forbids images/slides/video/audio).
MEDIA=$(git ls-files | grep -Ei '\.(png|jpe?g|gif|pdf|pptx?|mp4|mp3|wav|avi|mov)$' || true)
if [ -z "$MEDIA" ]; then
  ok "no images / PDFs / slides / audio / video tracked (brief forbids them)"
else
  bad "media files tracked: $(echo "$MEDIA" | tr '\n' ' ')"
fi

# A6. Working tree is clean -- everything committed.
if [ -z "$(git status --porcelain)" ]; then
  ok "git working tree is clean (everything committed)"
else
  bad "uncommitted changes: $(git status --porcelain | wc -l) file(s)"
fi

# A7. Telemetry kill switch present in the config module.
if grep -q 'CREWAI_DISABLE_TELEMETRY' src/nykaa/config.py; then
  ok "CREWAI_DISABLE_TELEMETRY set in src/nykaa/config.py"
else
  bad "CREWAI_DISABLE_TELEMETRY not set in src/nykaa/config.py"
fi

# A8. README carries the required submission statements.
for phrase in "MOCK_LLM" "zero API keys" "20261106" "CREWAI_DISABLE_TELEMETRY"; do
  if grep -q "$phrase" README.md; then
    ok "README states: $phrase"
  else
    bad "README is missing the required statement: $phrase"
  fi
done

# A9. The environment lock is committed (grader can reproduce the install).
if git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
  ok "uv.lock committed (reproducible dependency install)"
else
  bad "uv.lock is NOT committed -- the grader cannot reproduce the environment"
fi

# --------------------------------------------------------------------------- #
hdr "B. COLD-STATE REBUILD  (simulating a fresh clone)"
# --------------------------------------------------------------------------- #

echo "  wiping all generated state (data/, __pycache__)..."
rm -rf data/
find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
if [ ! -d data ]; then
  ok "generated state wiped; nothing in data/ is required to be committed"
fi

run_script() {
  local label="$1"; shift
  if uv run --no-sync python "$@" >/dev/null 2>&1; then
    ok "$label"
  else
    bad "$label"
  fi
}

run_script "scripts/00_verify_env.py      (stack, CPU-only torch, telemetry off)" scripts/00_verify_env.py
run_script "dataset.py                   (Task 1 dataset report)"                 dataset.py
run_script "scripts/02_build_index.py    (Task 3 dual chunking + 2 collections)"  scripts/02_build_index.py
run_script "scripts/03_calibrate_threshold.py (Task 4 threshold)"                scripts/03_calibrate_threshold.py
run_script "scripts/04_demo_grounded.py  (Task 4 grounded + fallback)"           scripts/04_demo_grounded.py
run_script "scripts/05_eval_chunking.py  (Task 5 precision/recall)"              scripts/05_eval_chunking.py
run_script "scripts/06_escalation_score.py (Task 6 escalation score)"            scripts/06_escalation_score.py
run_script "scripts/07_crew_demo.py      (Task 7 crew + both tools)"             scripts/07_crew_demo.py
run_script "scripts/08_memory_demo.py    (Task 8 session memory)"                scripts/08_memory_demo.py
run_script "scripts/09_schema_demo.py    (Task 9 structured output)"             scripts/09_schema_demo.py
run_script "scripts/10_guardrails_demo.py (Task 10 guardrails firing)"           scripts/10_guardrails_demo.py
run_script "scripts/11_api_demo.py       (Tasks 11+12 API + logging)"            scripts/11_api_demo.py
run_script "scripts/13_eval_judge.py     (Task 13 15-query evaluation)"          scripts/13_eval_judge.py
run_script "scripts/14_autogen_review.py (Task 14 Autogen review)"               scripts/14_autogen_review.py
run_script "scripts/15_governance.py     (Task 15 governance)"                   scripts/15_governance.py
run_script "scripts/16_caching.py        (Task 16 response caching)"             scripts/16_caching.py

# --------------------------------------------------------------------------- #
hdr "C. ACCEPTANCE CRITERIA"
# --------------------------------------------------------------------------- #

ACCEPT_OUT=$(uv run --no-sync python scripts/99_acceptance_check.py 2>&1)
ACCEPT_RC=$?
echo "$ACCEPT_OUT" | tail -6
# NOTE: capture the exit code separately -- piping into `tail` would make the
# pipeline status that of `tail`, which always succeeds, and a real failure
# would sail through as a pass.
if [ "$ACCEPT_RC" -eq 0 ]; then
  ok "all acceptance criteria verified (see table above)"
else
  bad "acceptance criteria check reported failures"
fi

# --------------------------------------------------------------------------- #
hdr "RESULT"
# --------------------------------------------------------------------------- #

echo "  passed: $PASS    failed: $FAIL"
echo

# Transcripts record live timings and trace ids, so running this gate always
# rewrites them. That is expected -- it is not a failure, but they do need
# committing so the next run starts from a clean tree.
CHANGED=$(git status --porcelain | wc -l | tr -d ' ')
if [ "$CHANGED" -gt 0 ]; then
  echo "  NOTE: this run regenerated $CHANGED file(s) (transcripts capture live"
  echo "        timings/trace ids). Commit them so the next run starts clean:"
  echo
  echo "            git add -A && git commit -m 'Refresh transcripts'"
  echo
fi

if [ "$FAIL" -eq 0 ]; then
  printf "  \033[32mREADY TO UPLOAD\033[0m -- every check passed on a cold checkout.\n\n"
  exit 0
fi

printf "  \033[31mDO NOT UPLOAD YET\033[0m -- fix these first:\n\n"
for f in "${FAILURES[@]}"; do
  printf "    - %s\n" "$f"
done
echo
exit 1
