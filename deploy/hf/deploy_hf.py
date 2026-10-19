#!/usr/bin/env python3
"""
deploy_hf.py
============
Automated deployment script to deploy the Legal Search Engine to Hugging Face Spaces.
Uses `huggingface_hub` to upload code, frontend, and the pre-built FAISS index corpus.
"""

import sys
import os
import getpass
from pathlib import Path
from pathlib import Path

def main():
    print("=" * 65)
    print(" 🚀 Deploying Legal Search Engine to Hugging Face Spaces")
    print("=" * 65)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("\nInstalling huggingface_hub...")
        os.system(f"{sys.executable} -m pip install -U huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n🔑 Hugging Face User Access Token is required.")
        print("   Get one with 'Write' permissions at: https://huggingface.co/settings/tokens")
        token = getpass.getpass("Enter your Hugging Face Token: ").strip()

    if not token:
        print("❌ Error: Hugging Face Token cannot be empty.")
        sys.exit(1)

    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"\n✅ Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"\n❌ Invalid Hugging Face Token: {e}")
        sys.exit(1)

    default_space_name = "legal-search-engine"
    space_name = input(f"\nEnter Space Name [{default_space_name}]: ").strip() or default_space_name
    repo_id = f"{username}/{space_name}"

    print(f"\n📁 Target Space: https://huggingface.co/spaces/{repo_id}")
    print("Creating/updating Docker Space on Hugging Face...")

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token,
        )
        print("✅ Space repository created/verified.")
    except Exception as e:
        print(f"❌ Failed to create Space repo: {e}")
        sys.exit(1)

    # Optional Groq Secret key setting
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        groq_input = getpass.getpass("Enter your GROQ_API_KEY (press Enter to skip for offline mode): ").strip()
        if groq_input:
            groq_key = groq_input

    try:
        api.add_space_variable(repo_id=repo_id, key="LLM_MODEL", value="openai/gpt-oss-20b")
    except Exception as e:
        print(f"⚠️ Warning: Could not set LLM_MODEL variable: {e}")

    if groq_key:
        try:
            api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=groq_key)
            print("✅ Secret GROQ_API_KEY set in Space environment.")
        except Exception as e:
            print(f"⚠️ Warning: Could not set GROQ_API_KEY secret: {e}")

    # Only what the container needs. An allowlist (not an ignore list) so that
    # secrets such as .sesskey, data/, logs and reports can never be published.
    print("\n📦 Uploading code and FAISS index to Hugging Face (index + metadata ~80MB)...\n")
    repo_root = Path(__file__).resolve().parents[2]
    try:
        # The Space README needs Docker front matter; the GitHub README has none.
        readme_body = (repo_root / "README.md").read_text(encoding="utf-8")
        space_readme = (
            "---\n"
            "title: Indian Supreme Court Legal Search Engine\n"
            "emoji: ⚖️\n"
            "colorFrom: blue\n"
            "colorTo: indigo\n"
            "sdk: docker\n"
            "app_port: 7860\n"
            "pinned: false\n"
            "---\n\n" + readme_body
        )
        api.upload_file(path_or_fileobj=space_readme.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space", token=token)
        api.upload_file(path_or_fileobj=str(repo_root / "deploy/hf/Dockerfile"), path_in_repo="Dockerfile",
                        repo_id=repo_id, repo_type="space", token=token)
        api.upload_folder(
            folder_path=str(repo_root),
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=[
                "requirements.txt",
                "app.py", "search_engine.py", "utils.py", "validator.py", "agentic_layer.py",
                "frontend/*",
                "LAWdata_Corpus_2024/legal_precedents.index",
                "LAWdata_Corpus_2024/metadata_map.json",
                "LAWdata_Corpus_2024/citation_map.json",
                "LAWdata_Corpus_2024/manifest.json",
            ],
            token=token,
        )
        print("\n" + "=" * 65)
        print(" 🎉 Deployment Upload Complete!")
        print(f" 🔗 Live Space URL: https://huggingface.co/spaces/{repo_id}")
        print(" ⏳ Build status: Hugging Face is building the Docker container.")
        print("    You can view build logs directly at the link above.")
        print("=" * 65)
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
