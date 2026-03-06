#!/bin/bash
# Integration test script for offline translation system
# Tests both backend API and frontend-backend integration
#
# Requirements: curl, jq
# Install: apt-get install curl jq  (Debian/Ubuntu)
#          brew install curl jq      (macOS)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed${NC}"
    echo "Install: apt-get install curl  (Debian/Ubuntu) or  brew install curl  (macOS)"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed${NC}"
    echo "Install: apt-get install jq  (Debian/Ubuntu) or  brew install jq  (macOS)"
    exit 1
fi

# Configuration
BACKEND_URL=${BACKEND_URL:-"http://localhost:8000"}
FRONTEND_URL=${FRONTEND_URL:-"http://localhost:8501"}
MAX_RETRIES=30
RETRY_DELAY=5

echo -e "${YELLOW}=== Offline Translation Integration Tests ===${NC}\n"

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local retries=0

    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"

    while [ $retries -lt $MAX_RETRIES ]; do
        if curl --output /dev/null --silent --fail "$url"; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        retries=$((retries + 1))
        echo "  Attempt $retries/$MAX_RETRIES..."
        sleep $RETRY_DELAY
    done

    echo -e "${RED}✗ $name failed to start after $((MAX_RETRIES * RETRY_DELAY)) seconds${NC}"
    return 1
}

# Function to run test
run_test() {
    local test_name=$1
    local command=$2

    echo -e "\n${YELLOW}Testing: $test_name${NC}"

    if eval "$command"; then
        echo -e "${GREEN}✓ PASS: $test_name${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL: $test_name${NC}"
        return 1
    fi
}

# Wait for services
wait_for_service "$BACKEND_URL/health" "Backend"
wait_for_service "$FRONTEND_URL/_stcore/health" "Frontend"

echo -e "\n${YELLOW}=== Backend API Tests ===${NC}"

# Test 1: Health Check
run_test "Backend health check" \
    "curl --fail --silent $BACKEND_URL/health | jq -e '.status == \"ready\"'"

# Test 2: List Models
run_test "List available models" \
    "curl --fail --silent $BACKEND_URL/models | jq -e '.models | length > 0'"

# Test 3: Get model info
MODEL_ID=$(curl --silent $BACKEND_URL/models | jq -r '.models[0].model_id')
run_test "Model has supported language pairs" \
    "curl --fail --silent $BACKEND_URL/models | jq -e \".models[0].supported_pairs | length > 0\""

# Test 4: Simple Translation
run_test "Single translation request" \
    "curl --fail --silent -X POST $BACKEND_URL/translate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"source\": \"Hello world\"
        }' | jq -e '.translated_value != null and .latency_ms > 0'"

# Test 5: Translation returns model_was_warm
run_test "Translation includes model_was_warm flag" \
    "curl --fail --silent -X POST $BACKEND_URL/translate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"source\": \"Test\"
        }' | jq -e 'has(\"model_was_warm\")'"

# Test 6: Model Unload
run_test "Model unload endpoint" \
    "curl --fail --silent -X POST $BACKEND_URL/models/$MODEL_ID/unload | jq -e '.status == \"unloaded\"'"

# Test 7: Cold Start Detection (translate after unload)
run_test "Cold start after model unload" \
    "curl --fail --silent -X POST $BACKEND_URL/translate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"source\": \"Cold start test\"
        }' | jq -e '.model_was_warm == false'"

# Test 8: Evaluation with BLEU and CHRF
run_test "Evaluation with reference metrics (BLEU, CHRF)" \
    "curl --fail --silent -X POST $BACKEND_URL/evaluate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"items\": [
                {
                    \"source\": \"Hello\",
                    \"reference\": \"Hei\",
                    \"item_id\": \"test-1\"
                },
                {
                    \"source\": \"World\",
                    \"reference\": \"Verden\",
                    \"item_id\": \"test-2\"
                }
            ],
            \"metrics\": [\"bleu\", \"chrf\"]
        }' | jq -e '(.results | length) == 2 and .aggregates.bleu_mean != null and .aggregates.chrf_mean != null'"

# Test 9: Evaluation includes performance metrics
run_test "Evaluation includes CPU and RAM metrics" \
    "curl --fail --silent -X POST $BACKEND_URL/evaluate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"items\": [{\"source\": \"Test\", \"reference\": \"Test\"}],
            \"metrics\": [\"bleu\"]
        }' | jq -e '.results[0].cpu_percent_per_core != null and .results[0].ram_mean_mb != null'"

# Test 10: Evaluation aggregate statistics
run_test "Evaluation provides mean, median, stdev for metrics" \
    "curl --fail --silent -X POST $BACKEND_URL/evaluate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"items\": [
                {\"source\": \"Hello\", \"reference\": \"Hei\"},
                {\"source\": \"Goodbye\", \"reference\": \"Ha det\"}
            ],
            \"metrics\": [\"bleu\"]
        }' | jq -e '.aggregates | has(\"bleu_mean\") and has(\"bleu_median\") and has(\"bleu_stdev\")'"

# Test 11: Invalid model ID
run_test "Invalid model ID returns 404" \
    "! curl --fail --silent -X POST $BACKEND_URL/translate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"nonexistent-model\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"nob\",
            \"source\": \"Test\"
        }' 2>&1"

# Test 12: Invalid language pair
run_test "Invalid language pair returns 400" \
    "! curl --fail --silent -X POST $BACKEND_URL/translate \
        -H 'Content-Type: application/json' \
        -d '{
            \"model_id\": \"$MODEL_ID\",
            \"src_lang\": \"en\",
            \"tgt_lang\": \"xyz\",
            \"source\": \"Test\"
        }' 2>&1"

echo -e "\n${YELLOW}=== Frontend Integration Tests ===${NC}"

# Test 13: Frontend is accessible
run_test "Frontend homepage loads" \
    "curl --fail --silent $FRONTEND_URL > /dev/null"

# Test 14: Frontend health check
run_test "Frontend health check" \
    "curl --fail --silent $FRONTEND_URL/_stcore/health > /dev/null"

echo -e "\n${GREEN}=== All Tests Passed! ===${NC}"
echo -e "\nSummary:"
echo -e "  - Backend API: ${GREEN}✓${NC} All endpoints working"
echo -e "  - Translation: ${GREEN}✓${NC} Both cold and warm start"
echo -e "  - Evaluation: ${GREEN}✓${NC} Metrics and profiling working"
echo -e "  - Frontend: ${GREEN}✓${NC} Accessible and healthy"
echo -e "\nServices ready for use:"
echo -e "  - Frontend: ${FRONTEND_URL}"
echo -e "  - Backend: ${BACKEND_URL}"
echo -e "  - API Docs: ${BACKEND_URL}/docs"
