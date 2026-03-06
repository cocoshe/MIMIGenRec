"""Upload one checkpoint folder to Hugging Face.

python LlamaFactory/scripts/upload_ckpt_to_hf.py --user "$HF_USER" --ckpt /path/to/checkpoint --repo_id "$HF_USER/repo-name"
"""

import argparse
import os
import sys


def _normpath(p):
    return os.path.normpath(os.path.abspath(p))


def main():
    parser = argparse.ArgumentParser(description="Upload one folder to Hugging Face.")
    parser.add_argument(
        "--user",
        type=str,
        default=os.environ.get("HF_USER", "").strip(),
        help="Hugging Face username (or HF_USER env).",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Folder to upload.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo id (e.g. user/repo-name).")
    args = parser.parse_args()

    local_path = _normpath(args.ckpt)
    repo_id = args.repo_id.strip()

    if not args.user and "/" in repo_id:
        args.user = repo_id.split("/", 1)[0]
    if not args.user:
        print("Error: set HF_USER or --user.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(local_path):
        print(f"Error: not a directory: {local_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    ignore = [
        "**/*rng_state_*.pth",
        "**/*scheduler.pt",
        "**/*trainer_state.json",
        "**/*training_args.bin",
        "**/latest",
        "**/global_step*",
        "**/*optim_states.pt",
        "**/*model_states.pt",
    ]

    print(f"Uploading: {local_path}")
    print(f"  -> repo: {repo_id}")
    try:
        create_repo(repo_id, private=False, exist_ok=True)
        api.upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model", ignore_patterns=ignore)
        print(f"  Done: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        if "authenticate" in str(e).lower() or "401" in str(e) or "login" in str(e).lower():
            print("Run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
