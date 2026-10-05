#!/usr/bin/env python3
"""Download SWE-rebench-V2 (or V1) and write slime-compatible JSONL."""
import argparse, base64, json, os, shlex, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--dataset", default="nebius/SWE-rebench-V2",
                   help="HuggingFace dataset name (default: nebius/SWE-rebench-V2)")
    p.add_argument("--split", default=None,
                   help="Dataset split (default: train for V2, filtered for V1)")
    args = p.parse_args()

    # V2 uses split='train'; V1 uses split='filtered'
    if args.split is None:
        args.split = "filtered" if args.dataset == "nebius/SWE-rebench" else "train"

    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "--no-cache-dir", "datasets"])
        from datasets import load_dataset

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split)
    # V2 covers 20 languages; eval_cmd template is Python-specific (conda/pytest).
    is_v2 = args.dataset != "nebius/SWE-rebench"
    written = 0
    with open(args.output, "w") as f:
        for row in ds:
            if written >= args.max_rows:
                break
            if is_v2 and row.get("language", "python") != "python":
                continue
            f2p = row.get("FAIL_TO_PASS") or []
            if isinstance(f2p, str):
                f2p = json.loads(f2p)
            p2p = row.get("PASS_TO_PASS") or []
            if isinstance(p2p, str):
                p2p = json.loads(p2p)

            # V2: workdir=/<repo_name>, conda at /opt/conda, env=swebench_matterhorn
            # V1: workdir=/testbed,     conda at /opt/miniconda3, env=testbed
            if is_v2:
                repo_name = row["repo"].split("/")[-1]
                workdir = f"/{repo_name}"
                conda_sh = "/opt/conda/etc/profile.d/conda.sh"
                conda_env = "swebench_matterhorn"
            else:
                workdir = "/testbed"
                conda_sh = "/opt/miniconda3/etc/profile.d/conda.sh"
                conda_env = "testbed"

            tests = list(dict.fromkeys([*f2p, *p2p]))
            tests_args = " ".join(shlex.quote(t) for t in tests)
            if tests_args:
                install_config = row.get("install_config") or {}
                install_commands = install_config.get("install") or []
                if isinstance(install_commands, str):
                    install_commands = [install_commands]
                # Prefer the benchmark's canonical command. Some parameterized
                # V2 node IDs contain whitespace and cannot be reconstructed
                # reliably as individual pytest CLI selectors.
                test_command = install_config.get("test_cmd") or (
                    f"python -m pytest --no-header -rN --tb=no -q {tests_args}"
                )
                # The V2 image is based on the buggy commit. Apply the model
                # diff first (slime's scaleswe grader), then materialize the
                # benchmark test patch here before running F2P and P2P tests.
                # Base64 avoids embedding arbitrary patch text in shell syntax.
                test_patch = row.get("test_patch", "") or ""
                patch_b64 = base64.b64encode(test_patch.encode()).decode()
                script = "\n".join([
                    "set -eo pipefail",
                    f"cd {shlex.quote(workdir)}",
                    f"printf %s {shlex.quote(patch_b64)} | base64 -d > /tmp/swe_test_patch.diff",
                    "if [[ -s /tmp/swe_test_patch.diff ]]; then",
                    "  if git apply --check /tmp/swe_test_patch.diff; then",
                    "    git apply --whitespace=nowarn /tmp/swe_test_patch.diff",
                    "  elif git apply --reverse --check /tmp/swe_test_patch.diff; then",
                    "    : # image already contains the test patch",
                    "  else",
                    "    echo 'test patch does not apply cleanly' >&2; exit 2",
                    "  fi",
                    "fi",
                    f"source {shlex.quote(conda_sh)}",
                    f"conda activate {shlex.quote(conda_env)}",
                    *install_commands,
                    test_command,
                ])
                eval_cmd = f"bash -lc {shlex.quote(script)}"
            else:
                eval_cmd = "false"
            f.write(json.dumps({
                "prompt": row["problem_statement"],
                "label":  row["instance_id"],
                "metadata": {
                    "image":       row.get("image_name") or row.get("docker_image"),
                    "workdir":     workdir,
                    "instance_id": row["instance_id"],
                    "eval_cmd":    eval_cmd,
                    "remote_env_info": {
                        "image":                    row.get("image_name") or row.get("docker_image"),
                        "workdir":                  workdir,
                        "instance_id":              row["instance_id"],
                        "repo":                     row["repo"],
                        "base_commit":              row.get("base_commit", ""),
                        "test_patch":               row.get("test_patch", ""),
                        "FAIL_TO_PASS":             f2p,
                        "PASS_TO_PASS":             p2p,
                        "version":                  row.get("version"),
                        "hints_text":               row.get("hints_text", ""),
                        "environment_setup_commit": row.get("environment_setup_commit", ""),
                        "problem_statement":        row["problem_statement"],
                    },
                },
            }) + "\n")
            written += 1

    print(f"Wrote {written} row(s) to {args.output}")

if __name__ == "__main__":
    main()
