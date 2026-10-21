import os
import subprocess
from modal import App, Image, Volume, web_server

# 1. Definiere das persistente Volume für deine Modelle und ComfyUI-Struktur
comfy_volume = Volume.from_name("comfyui-krea2-storage", create_if_missing=True)

# 2. Definiere das Docker-Image für Modal
# Wir installieren git, git-lfs und comfy-cli, um ComfyUI sauber aufzusetzen.
comfy_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "libsndfile1")
    .run_commands("git lfs install")
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio", 
        extra_index_url="https://pytorch.org"
    )
    .pip_install("comfy-cli", "huggingface_hub[hf_transfer]")
)

app = App("comfyui-krea2-turbo", image=comfy_image)

# Pfade innerhalb unseres Containers, gemountet auf das Volume
VOLUME_MOUNT_PATH = "/data"
COMFYUI_DIR = f"{VOLUME_MOUNT_PATH}/ComfyUI"

# --- DOWNLOAD PHASE (Läuft vorab im Hintergrund ab) ---
@app.function(
    volumes={VOLUME_MOUNT_PATH: comfy_volume}, 
    timeout=1800 # 30 Minuten Zeitlimit für den initialen Download
)
def download_krea2_models():
    """Lädt ComfyUI und die Krea 2 Turbo Modellgewichte ins Volume."""
    from huggingface_hub import snapshot_download

    # Falls ComfyUI noch nicht im Volume existiert, klonen wir es
    #if not os.path.exists(COMFYUI_DIR):
    #    print("Cloning ComfyUI into Volume...")
    #    subprocess.run(["git", "clone", "https://github.com", COMFYUI_DIR], check=True)
    
    # Pfade für die Krea-Modelle definieren (Comfy-Org Struktur)
    checkpoint_dir = f"{COMFYUI_DIR}/models/checkpoints"
    clip_dir = f"{COMFYUI_DIR}/models/clip"
    vae_dir = f"{COMFYUI_DIR}/models/vae"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)

    print("Starting download of Krea 2 Turbo weights from Comfy-Org...")
    # HF_TRANSFER für maximale Bandbreitennutzung aktivieren
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Download der Krea 2 Dateien von Comfy-Org
    snapshot_download(
        repo_id="Comfy-Org/Krea-2",
        repo_type="model",
        allow_patterns=[
            "krea2_turbo_fp8_scaled.safetensors", # Das Haupt-Unet Modell
            "qwen3vl_4b_text_encoder_bf16.safetensors", # Der Text-Encoder
            "qwen_image_vae.safetensors" # Die VAE
        ],
        local_dir=VOLUME_MOUNT_PATH + "/temp_krea",
        local_dir_use_symlinks=False
    )
    
    # Verschiebe die Dateien in die richtigen ComfyUI-Ordner
    temp_path = VOLUME_MOUNT_PATH + "/temp_krea"
    subprocess.run(f"mv {temp_path}/krea2_turbo_fp8_scaled.safetensors {checkpoint_dir}/", shell=True)
    subprocess.run(f"mv {temp_path}/qwen3vl_4b_text_encoder_bf16.safetensors {clip_dir}/", shell=True)
    subprocess.run(f"mv {temp_path}/qwen_image_vae.safetensors {vae_dir}/", shell=True)
    
    # Installiere notwendige Krea-Erweiterungen (Custom Nodes)
    nodes_dir = f"{COMFYUI_DIR}/custom_nodes"
    if not os.path.exists(f"{nodes_dir}/comfyui-modal"):
        subprocess.run(["git", "clone", "https://github.com/JunnyWon/comfyui-modal.git", f"{nodes_dir}/comfyui-modal"], check=True)

    # Volume-Änderungen sichern
    comfy_volume.commit()
    print("Krea 2 Turbo Setup completed successfully!")


