#!/bin/bash
# Runs one script on HF Jobs, on the packages frozen on the Colab T4 at Step 0. Needs HF_TOKEN.
set -euo pipefail
cd "$(dirname "$0")"

WORKSPACE=https://huggingface.co/datasets/GauravRaut/lexsi-track3-workspace/resolve/main
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$WORKSPACE/00_admin/pip_freeze_after.txt" \
    | grep -v -E ' @ |^-' > /tmp/constraints.txt

for name in AgentTune AuditKIT; do
    commit=$(python -c "import config; print(config.${name^^}_COMMIT)")
    git clone -q "https://github.com/Lexsi-Labs/$name" "/tmp/$name"
    git -C "/tmp/$name" checkout -q "$commit"
done
pip install -q --root-user-action=ignore --extra-index-url https://download.pytorch.org/whl/cu130 \
    -c /tmp/constraints.txt torch pytest /tmp/AgentTune /tmp/AuditKIT

mkdir -p /tmp/out
python -m pytest tests -q 2>&1 | tail -3 | tee /tmp/out/tests.txt
exec python "$@"
