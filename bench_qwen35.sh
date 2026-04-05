#!/bin/bash
# Qwen3.5-122B-A10B benchmark — measures tok/s for different prompt types
# Usage: ./bench_qwen35.sh [label]
# Runs each test 2x and shows both results

LABEL="${1:-test}"
API="http://localhost:8000/v1/chat/completions"
MODEL="qwen"

bench() {
  local name="$1"
  local prompt="$2"
  local max_tokens="${3:-512}"

  local start=$(date +%s%N)
  local response=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
      \"max_tokens\": $max_tokens,
      \"temperature\": 0.0
    }")
  local end=$(date +%s%N)

  local elapsed=$(echo "scale=2; ($end - $start) / 1000000000" | bc)
  local prompt_tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null)
  local completion_tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null)

  if [ -z "$completion_tokens" ] || [ "$completion_tokens" = "0" ]; then
    echo "  [$name] FAILED — no completion tokens"
    echo "$response" | python3 -m json.tool 2>/dev/null | head -10
    return
  fi

  local toks=$(echo "scale=1; $completion_tokens / $elapsed" | bc)
  echo "  [$name] ${completion_tokens} tokens in ${elapsed}s = ${toks} tok/s (prompt: ${prompt_tokens})"
}

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Qwen3.5-122B-A10B Benchmark: $LABEL"
echo "║  $(date)"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for RUN in 1 2; do
  echo "── Run $RUN/2 ──────────────────────────────────────"

  # Test 1: Simple Q&A
  bench "Q&A" "What are the main differences between TCP and UDP? Be concise." 256

  # Test 2: Code generation
  bench "Code" "Write a Python function that implements binary search on a sorted list. Include type hints and docstring." 512

  # Test 3: JSON generation (repetitive structure — ngram should shine here)
  bench "JSON" "Generate a JSON array of 10 fictional employees with fields: name, age, department, salary, email, skills (array of 3). Output ONLY valid JSON, no explanation." 1024

  # Test 4: Math/reasoning (short output)
  bench "Math" "What is 7823 * 4519? Show only the answer." 64

  # Test 5: Long code (repetitive patterns — ngram benefit)
  bench "LongCode" "Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Include all rotation methods." 2048

  echo ""
done

echo "=== Done ==="
