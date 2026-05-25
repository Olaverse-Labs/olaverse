import os
import sys
from huggingface_hub import HfApi

def main():
    api = HfApi()
    
    # Check for token in env or Hugging Face local config
    token = os.environ.get("HF_TOKEN")
    
    repo_id = "olaverse/otk-bpe-50k"
    models_dir = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models"
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        sys.exit(1)
        
    files_to_upload = [f for f in os.listdir(models_dir) if f.endswith(".json") and f.startswith("otk-bpe-50k-")]
    
    if not files_to_upload:
        print("No .json files found in models directory to upload.")
        sys.exit(1)
        
    print(f"Found {len(files_to_upload)} files to upload to HF repo '{repo_id}':")
    for f in files_to_upload:
        print(f" - {f}")
        
    print(f"\nEnsuring repository '{repo_id}' exists on the Hugging Face Hub...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        print("✅ Repository verified/created successfully.")
    except Exception as e:
        print(f"⚠️ Could not verify/create repository (will try uploading anyway): {e}")
        
    print("\nStarting upload...")
    success_count = 0
    for filename in files_to_upload:
        local_path = os.path.join(models_dir, filename)
        print(f"Uploading {filename}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"✅ Successfully uploaded {filename}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
            
    print(f"\nUpload run complete. Successfully uploaded {success_count}/{len(files_to_upload)} files.")
    if success_count < len(files_to_upload):
        print("Tip: If you saw authorization errors, please make sure you are logged in by running 'hf auth login' or setting the 'HF_TOKEN' environment variable with a write-access token.")

if __name__ == "__main__":
    main()
