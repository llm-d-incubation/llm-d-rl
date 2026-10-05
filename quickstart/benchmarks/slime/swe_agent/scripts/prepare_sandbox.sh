#!/usr/bin/env bash
# Warm up the slime stack before a training run:
#   1. Prepare the SWE-rebench-V2 dataset on the PVC.
#   2. Assign instances to runners and prefetch each runner's Docker images in
#      parallel. Each runner only pulls the images it will actually run.
#
# Sandbox-runner pods are deployed by ../setup_sandbox.sh, which also calls this
# script. Run it directly only to re-prepare against an existing runner pool.
#
# Safe to rerun — already-cached images are skipped.
#
# Required env:
#   NAMESPACE          kubernetes namespace
#   ROLLOUT_BATCH_SIZE number of dataset rows (sourced from benchmark.env)

set -euo pipefail

: "${NAMESPACE:?export NAMESPACE=<your-namespace> first}"
: "${ROLLOUT_BATCH_SIZE:?source benchmark.env first}"
: "${N_SAMPLES:?source benchmark.env first}"
: "${NUM_ROLLOUT:?source benchmark.env first}"
# Each training step consumes the next rollout batch from slime's data source.
# Prepare enough rows that no prompt/image repeats during this run.
MAX_ROWS=$(( NUM_ROLLOUT * ROLLOUT_BATCH_SIZE ))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Helpers
find_running_pod() {
  kubectl get pod -n "$NAMESPACE" -l "$1" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

wait_for_pod() {
  local label="$1" desc="$2" tries=0 pod
  while pod="$(find_running_pod "$label")"; [[ -z "$pod" ]]; do
    (( ++tries > 30 )) && { echo "error: $desc not ready after 5 minutes" >&2; exit 1; }
    # stderr: callers capture stdout, which carries only the pod name.
    echo "  waiting for $desc …" >&2
    sleep 10
  done
  echo "$pod"
}

# Discover sandbox-runner pods
echo "==> Discovering sandbox-runner pods in namespace ${NAMESPACE}"
RUNNER_PODS=()
RUNNER_URLS=()
while IFS= read -r name; do
  RUNNER_PODS+=("$(wait_for_pod "app=${name}" "${name}")")
  RUNNER_URLS+=("http://${name}:8080")
done < <(kubectl get deployment -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" \
           | grep "^sandbox-runner-" | sort -t- -k3 -n)

N_RUNNERS="${#RUNNER_PODS[@]}"
[[ "$N_RUNNERS" -gt 0 ]] || {
  echo "error: no sandbox-runner-* deployments found in ${NAMESPACE}" >&2
  echo "       Run: NAMESPACE=$NAMESPACE bash setup_sandbox.sh" >&2
  exit 1
}
echo "    found ${N_RUNNERS} runner(s): ${RUNNER_URLS[*]}"

# Prepare dataset on the head pod
echo ""
echo "==> [1/2] preparing dataset (max_rows=${MAX_ROWS})"
HEAD_POD="$(wait_for_pod "ray.io/node-type=head" "slime head pod")"
echo "    head pod: ${NAMESPACE}/${HEAD_POD}"

kubectl exec -n "$NAMESPACE" -i "$HEAD_POD" -- \
  python3 - --output "/tmp/slime/data/swe_train.jsonl" --max-rows "$MAX_ROWS" \
  < "${SCRIPT_DIR}/prepare_swe_rebench.py"

# Save runner URLs to the PVC
# Before the prefetch, not after: the launcher routes by this file, so a failed
# image pull must not leave it describing an older, differently sized pool.
URLS_CSV="$(IFS=,; echo "${RUNNER_URLS[*]}")"
kubectl exec -n "$NAMESPACE" "$HEAD_POD" -- \
  bash -c "echo '${URLS_CSV}' > /tmp/slime/data/sandbox_runners.txt"
echo "    runner URLs saved to /tmp/slime/data/sandbox_runners.txt"

# Assign images to runners and prefetch
# Uses round-robin assignment (same logic as dind_sandbox.py) so each runner only
# pulls the images it will actually run. All runners prefetch in parallel.
echo ""
echo "==> [2/2] assigning images to runners and prefetching"

kubectl exec -n "$NAMESPACE" -i "$HEAD_POD" -- python3 -u - \
  "${RUNNER_URLS[*]}" "/tmp/slime/data/swe_train.jsonl" "$N_SAMPLES" \
  "$ROLLOUT_BATCH_SIZE" <<'PYEOF'
import json, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASES = sys.argv[1].split()
N = len(BASES)
JSONL    = sys.argv[2]
N_SAMPLES = int(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
CONCURRENCY_PER_RUNNER = 4

def name_of(base):
    """Hostname from a base URL, so logs name the real runner, not its index."""
    return base.split("//", 1)[-1].split(":", 1)[0]

def wait_ready(base, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"{base}/docs", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"{base} not ready after {timeout}s")

for base in BASES:
    print(f"  waiting for {name_of(base)} to be ready ...", flush=True)
    wait_ready(base)
    print(f"  {name_of(base)} ready", flush=True)

try:
    lines = open(JSONL).readlines()
except FileNotFoundError:
    print(f"error: {JSONL} not found — did step 1 succeed?", flush=True)
    sys.exit(1)

# Slime consumes successive contiguous batches without replacement. Assign in
# dataset order so every step is balanced across runners. Slime passes only
# image metadata, and every sample in a GRPO group shares that image, so
# image is both the routing key and the unit of capacity.
assignment = {}
runner_images = [set() for _ in range(N)]
rows = [json.loads(line) for line in lines]
for row_index, row in enumerate(rows):
    metadata = row.get("metadata") or {}
    remote = metadata.get("remote_env_info") or {}
    img = metadata.get("image") or remote.get("image") or remote.get("image_url")
    if not img:
        raise ValueError(f"row {row.get('label', '<unknown>')} has no sandbox image")
    if img in assignment:
        continue
    runner = row_index % N
    assignment[img] = runner
    runner_images[runner].add(img)

assignment_path = "/tmp/slime/data/sandbox_assignment.json"
with open(assignment_path, "w") as f:
    import json as _j; _j.dump(assignment, f)
print(f"  wrote {len(assignment)} image assignments to {assignment_path}", flush=True)

peak_groups = [0] * N
for start in range(0, len(rows), BATCH_SIZE):
    groups = [0] * N
    for row in rows[start:start + BATCH_SIZE]:
        metadata = row.get("metadata") or {}
        remote = metadata.get("remote_env_info") or {}
        img = metadata.get("image") or remote.get("image") or remote.get("image_url")
        groups[assignment[img]] += 1
    peak_groups = [max(old, new) for old, new in zip(peak_groups, groups)]

for i, imgs in enumerate(runner_images):
    groups = len(imgs)
    print(
        f"  {name_of(BASES[i])}: {groups} unique image(s), "
        f"{peak_groups[i] * N_SAMPLES} peak concurrent container(s)",
        flush=True,
    )

STATUS_LABEL = {
    "cached":          "cached      ",
    "loaded_from_tar": "from-tar    ",
    "pulled":          "PULLED      ",
}

def prefetch_one(runner, image):
    body = json.dumps({"images": [image]}).encode()
    req = urllib.request.Request(
        f"{BASES[runner]}/images/prefetch", method="POST",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["results"][0]

total  = sum(len(imgs) for imgs in runner_images)
lock   = threading.Lock()
done   = 0
counts = {}

with ThreadPoolExecutor(max_workers=N * CONCURRENCY_PER_RUNNER) as ex:
    futures = {
        ex.submit(prefetch_one, i, img): (i, img)
        for i, imgs in enumerate(runner_images)
        for img in sorted(imgs)
    }
    for fut in as_completed(futures):
        runner, img = futures[fut]
        try:
            result = fut.result()
            status = result.get("status", "unknown")
        except Exception as e:
            status = "failed"
            print(f"    error {name_of(BASES[runner])} {img}: {e}", flush=True)
        with lock:
            done += 1
            counts[status] = counts.get(status, 0) + 1
            label = STATUS_LABEL.get(status, f"{status:<12}")
            print(f"  [{done:3}/{total}] {name_of(BASES[runner])} {label} {img.split('/')[-1]}", flush=True)

print("", flush=True)
print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
successful = {"cached", "loaded_from_tar", "pulled"}
failed = sum(count for status, count in counts.items() if status not in successful)
if failed:
    print(f"error: {failed} image(s) could not be prefetched", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "==> sandbox ready (${N_RUNNERS} runner(s), ${MAX_ROWS} rows)"
