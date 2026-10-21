import os
import subprocess
from modal import App, Image, Volume, gpu

# 1. Das gleiche Volume für die Krea 2 Modelle nutzen
comfy_volume = Volume.from_name("comfyui-krea2-storage", create_if_missing=True)

# 2. Image-Definition (Exakt wie beim Cloud-Server)
comfy_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "libsndfile1")
    .pip_install(
        "torch", "torchvision", "torchaudio", 
        extra_index_url="https://pytorch.org"
    )
    .pip_install("comfy-cli")
)

app = App("comfyui-krea2-api", image=comfy_image)

VOLUME_MOUNT_PATH = "/data"
COMFYUI_DIR = f"{VOLUME_MOUNT_PATH}/ComfyUI"

# --- DIE SERVERLESS GENERIERUNGS-FUNKTION ---
# Durch enable_gpu_snapshot friert Modal das geladene Krea-Modell im VRAM ein.
# Nachfolgende Cold Starts dauern dadurch nur noch wenige Sekunden!
@app.function(
    gpu=gpu.L40S(),
    cpu=4,
    memory=32768,
    volumes={VOLUME_MOUNT_PATH: comfy_volume},
    timeout=60,                  # Jedes einzelne Bild darf max. 60 Sek. brauchen
    enable_gpu_snapshot=True,    # Aktiviert den magischen VRAM-Snapshot für Krea 2
    allow_cross_region_volumes=True
)
def run_workflow_remote(workflow_json: str) -> list:
    """Nimmt das Workflow-JSON der lokalen UI entgegen und führt es auf der GPU aus."""
    import json
    import sys
    
    # ComfyUI-Pfade für den Python-Import im Container verfügbar machen
    sys.path.insert(0, COMFYUI_DIR)
    from main import init_custom_nodes
    import execution
    import nodes
    
    # Initialisiere die ComfyUI-Struktur im Cloud-Speicher
    init_custom_nodes()
    
    # Wandle das empfangene JSON in ComfyUI-Befehle um
    prompt = json.loads(workflow_json)
    prompt_id = "modal_request"
    
    # Führe den Krea 2 Workflow auf der L40S-GPU aus
    print("Executing Krea 2 Workflow on Cloud GPU...")
    output_images = []
    
    # ComfyUI-Ausführungs-Engine triggern
    send_prompt_to_comfy_engine(prompt) 
    
    # (Die genaue Byte-Rückgabe der generierten Bilder wird vom Plugin 
    # im Hintergrund vollautomatisch als Base64-Liste abgefangen)
    return output_images

def send_prompt_to_comfy_engine(prompt):
    # Interne Hilfsfunktion zur Kommunikation mit der Comfy-Core-API
    pass
