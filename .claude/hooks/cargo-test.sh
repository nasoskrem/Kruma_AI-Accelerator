#!/usr/bin/env bash
# PostToolUse hook: after Claude edits a Rust source file, run `cargo test`.
# If it fails, exit 2 so the failure output is fed back to Claude to react to.
set -uo pipefail

# Hook input arrives as JSON on stdin. Pull out the edited file's path.
input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

# Only react to Rust files; ignore edits to docs, configs, etc.
case "$file" in
  *.rs) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

echo "🦀 Rust file changed ($file) — running cargo test..."
if output=$(cargo test --quiet 2>&1); then
  echo "✅ cargo test passed."
  exit 0
else
  {
    echo "❌ cargo test FAILED after editing ${file}:"
    echo "$output"
  } >&2
  exit 2
fi
