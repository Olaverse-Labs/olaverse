import os
import sys
import urllib.request
import urllib.error

# Default Hugging Face repository hosting the serialized model JSON files
HF_REPO_ID = "olaverse/otk-bpe-50k"

def get_cache_dir():
    """
    Get the default cross-platform local cache directory for olaverse models.
    Respects XDG_CACHE_HOME, falls back to ~/.cache/olaverse/models.
    """
    cache_base = os.environ.get("XDG_CACHE_HOME")
    if not cache_base:
        cache_base = os.path.expanduser("~/.cache")
    return os.path.join(cache_base, "olaverse", "models")

def get_model_path(filename, repo_id=None):
    """
    Get the path to a model file.
    Resolves locations in the following order:
    1. Custom directory specified via OLAVERSE_MODELS_DIR environment variable.
    2. Local cache directory (~/.cache/olaverse/models/).
    3. Bundled local package directory (olaverse/models/) for development/fallback.
    4. Downloads the model on-demand from Hugging Face if not found locally.
    
    Args:
        filename (str): Name of the model file (with or without .json extension).
        repo_id (str, optional): Hugging Face repo ID to download from.
        
    Returns:
        str: Absolute path to the resolved model file.
    """
    # 1. Check custom models directory specified via environment variable
    custom_dir = os.environ.get("OLAVERSE_MODELS_DIR")
    if custom_dir:
        custom_path = os.path.join(custom_dir, filename)
        if os.path.exists(custom_path):
            return custom_path

    # 2. Check the local cache directory
    # Cache is namespaced by repo_id (when given) so repos that reuse generic
    # filenames (e.g. every olaverse/prism-* repo ships "model.py") don't
    # collide with each other in a shared flat cache directory.
    cache_dir = get_cache_dir()
    cache_path = os.path.join(cache_dir, repo_id, filename) if repo_id else os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        return cache_path

    # 3. Check the local package directory (development fallback)
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    pkg_path = os.path.join(pkg_dir, filename)
    if os.path.exists(pkg_path):
        return pkg_path

    # 4. Download from Hugging Face
    if repo_id is None:
        repo_id = os.environ.get("OLAVERSE_HF_REPO", HF_REPO_ID)

    os.makedirs(os.path.dirname(cache_path) or cache_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    print(f"Downloading {filename} from Hugging Face ({url})...", file=sys.stderr)
    try:
        import ssl
        try:
            # macOS Python builds often ship without root certificates; certifi
            # (already pulled in via the requests dependency) provides a CA
            # bundle so verification can stay on.
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        # Escape hatch for environments where verification cannot succeed
        # (e.g. corporate TLS interception with no CA bundle available).
        if os.environ.get("OLAVERSE_INSECURE_SSL") == "1":
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            req.add_header("Authorization", f"Bearer {hf_token}")
            
        # Use urllib to avoid adding external dependencies
        with urllib.request.urlopen(req, context=ctx) as response, open(cache_path, 'wb') as out_file:
            out_file.write(response.read())
            
        return cache_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model file '{filename}' from Hugging Face. "
            f"Please check your internet connection or ensure the file exists. "
            f"To load offline, you can download the model file and place it in '{cache_dir}' or "
            f"set the OLAVERSE_MODELS_DIR environment variable to its folder. "
            f"If this is an SSL certificate error (common on macOS Python without root "
            f"certificates installed), fix your certificate store or set "
            f"OLAVERSE_INSECURE_SSL=1 to skip verification at your own risk. "
            f"Error: {e}"
        ) from e
