import os
import sys
import shutil
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi

def main():
    models_dir = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models"
    output_base_dir = "/Users/olumideola/Desktop/olaverse-ai/projects/tokenizers/hf_formats"
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        sys.exit(1)
        
    # Language/file mappings
    tokenizers_map = {
        "yo": "otk-bpe-50k-yo.json",
        "ig": "otk-bpe-50k-ig.json",
        "ha": "otk-bpe-50k-ha.json",
        "pcm": "otk-bpe-50k-pcm.json",
        "naija": "otk-bpe-50k-naija.json",
    }
    
    print("Starting conversion to Hugging Face transformers format...\n")
    
    # Ensure base output directory exists and is clean
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    for lang, filename in tokenizers_map.items():
        local_json_path = os.path.join(models_dir, filename)
        if not os.path.exists(local_json_path):
            print(f"⚠️ Warning: Could not find model file {filename} at {local_json_path}. Skipping.")
            continue
            
        lang_output_dir = os.path.join(output_base_dir, lang)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        print(f"Converting '{lang}' ({filename})...")
        try:
            # Wrap the raw BPE JSON using PreTrainedTokenizerFast
            # Map special tokens used during tokenizer training
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=local_json_path,
                bos_token="[CLS]",
                eos_token="[SEP]",
                unk_token="[UNK]",
                pad_token="[PAD]",
                mask_token="[MASK]",
                clean_up_tokenization_spaces=True
            )
            
            # Save the pretrained tokenizer (generates tokenizer.json, tokenizer_config.json, special_tokens_map.json)
            tokenizer.save_pretrained(lang_output_dir)
            print(f"✅ Successfully converted and saved to: {lang_output_dir}")
            
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
            
    print(f"\nAll local conversions complete. Output folders are located at: {output_base_dir}\n")
    
    # Prompt user for optional automatic upload to Hugging Face
    upload = input("Do you want to upload these directories to the Hugging Face Hub (olaverse/otk-bpe-50k)? (y/n): ").strip().lower()
    if upload == 'y':
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Error: HF_TOKEN environment variable not set. Please set it before uploading.")
            sys.exit(1)
            
        api = HfApi()
        repo_id = "olaverse/otk-bpe-50k"
        
        print(f"\nUploading directories to repo '{repo_id}'...")
        for lang in tokenizers_map.keys():
            lang_output_dir = os.path.join(output_base_dir, lang)
            if os.path.exists(lang_output_dir):
                print(f"Uploading folder '{lang}'...")
                try:
                    api.upload_folder(
                        folder_path=lang_output_dir,
                        path_in_repo=lang,
                        repo_id=repo_id,
                        repo_type="model",
                        token=token
                    )
                    print(f"✅ Successfully uploaded '{lang}' folder.")
                except Exception as e:
                    print(f"❌ Failed to upload '{lang}' folder: {e}")
        print("\nAll uploads complete!")
    else:
        print("Skipped Hugging Face Hub upload. You can run this script later or upload folders manually.")

if __name__ == "__main__":
    main()
