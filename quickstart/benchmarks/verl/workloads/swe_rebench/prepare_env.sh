#!/usr/bin/env bash
set -xeuo pipefail

# Prepare the swe_rebench workload on the node it is run on: preprocess the
# datasets, then pre-pull + patch every referenced Docker image on THIS node's
# local Docker daemon. See README.md §2.
#
# Run on the ray-head node, from the directory this file was copied into (see
# README.md §1). Requires: uni_agent installed, a local Docker daemon.
#
#   bash prepare_env.sh
#
# Every ray-worker node needs the same images on its own (independent) Docker
# daemon -- a Ray task can land on any node. Run this same script again on
# each worker node (README.md §2): if `$DATA_DIR` already has both parquet
# files (e.g. copied over from the ray-head node), step 1 is skipped and it
# goes straight to pre-pulling/patching images on that node.
#
# MAX_INSTANCES caps the dataset for a first run so image pre-pull stays
# bounded (tens of minutes, not hours); set MAX_INSTANCES=0 for the full
# 6542-row nebius/SWE-rebench "filtered" split -- at that scale, pre-pulling
# every image is impractical; pre-fix a curated subset instead and accept
# occasional cold pulls for the rest.

DATA_DIR="${DATA_DIR:-/tmp/verl/data/uni_agent}"
MAX_INSTANCES="${MAX_INSTANCES:-64}"       # 0 = full split, no cap
MAX_TEST_INSTANCES="${MAX_TEST_INSTANCES:-16}"

# -- 1. Preprocess: nebius/SWE-rebench (train) + SWE-bench Verified (val) -----
# Skipped if both parquet files are already present (e.g. copied from another
# node) -- preprocessing is deterministic for a given MAX_INSTANCES, so redoing
# it buys nothing and only costs a re-download of the raw HF dataset.
mkdir -p "$DATA_DIR"
if [[ -f "$DATA_DIR/swe_rebench_filtered.parquet" && -f "$DATA_DIR/swe_bench_verified.parquet" ]]; then
  echo "==> $DATA_DIR already has both parquet files -- skipping preprocessing."
else
  train_flags=()
  [[ "$MAX_INSTANCES" != "0" ]] && train_flags+=(--max-instances "$MAX_INSTANCES")
  python3 -m uni_agent.tasks.swe_rebench.preprocess \
      --local-save-dir "$DATA_DIR" "${train_flags[@]}"

  test_flags=()
  [[ "$MAX_TEST_INSTANCES" != "0" ]] && test_flags+=(--max-instances "$MAX_TEST_INSTANCES")
  python3 -m uni_agent.tasks.swe_bench.preprocess \
      --local-save-dir "$DATA_DIR" "${test_flags[@]}"
fi

# -- 2. Write the image list + patch Dockerfile, then pre-pull+fix locally ----
# One image per instance: swerebench/sweb.eval.x86_64.<instance_id> (train) and
# swebench/sweb.eval.x86_64.<instance_id> (val) -- see
# uni_agent/tasks/swe_rebench/preprocess.py:get_image_name. The docker sandbox
# provider uses these refs as-is (no registry rewrite), so they must already
# exist, tagged exactly this way, on each node's daemon.
python3 -c "
import pandas as pd
for f in ['swe_rebench_filtered.parquet', 'swe_bench_verified.parquet']:
    df = pd.read_parquet('$DATA_DIR/' + f)
    for _, row in df.iterrows():
        print(row['extra_info']['tools_kwargs']['task']['sandbox']['image'])
" | sort -u > /tmp/swe_rebench_images.txt

# tmux is needed by the stateful_shell tool; the apt-mirror swap works around
# archive.ubuntu.com/security.ubuntu.com being unreachable from this cluster.
cat > /tmp/image-fix.Dockerfile <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN sed -i 's/archive\.ubuntu\.com/azure.archive.ubuntu.com/g; s/security\.ubuntu\.com/azure.archive.ubuntu.com/g' /etc/apt/sources.list \
 && apt-get update && apt-get install -y tmux
EOF

while read -r img; do
  docker pull "$img" || true
  docker build --build-arg BASE_IMAGE="$img" -t "$img" -f /tmp/image-fix.Dockerfile /tmp || echo "FAILED: $img"
done < /tmp/swe_rebench_images.txt

echo "==> Done on this node. Datasets in $DATA_DIR; images pre-pulled+fixed locally."
echo "==> Repeat on every ray-worker node too (copy $DATA_DIR's two parquet files over"
echo "    first so this script's preprocessing step is skipped there) -- README.md §2."
