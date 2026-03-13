"""
image_edit_tab.py  –  DiffuseCraft Image Editing Tab
Model: prithivMLmods/FireRed-Image-Edit-1.0-8bit
       (FireRed-Image-Edit with standard diffusers QwenImageEditPlusPipeline)

Requires in requirements.txt:
    git+https://github.com/huggingface/diffusers
"""

import gc
import random

import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

# ── Model ID ──────────────────────────────────────────────────────────────────
MODEL_ID = "prithivMLmods/FireRed-Image-Edit-1.0-8bit"
MAX_SEED  = np.iinfo(np.int32).max

# ── Lazy pipeline holder ──────────────────────────────────────────────────────
_edit_pipe = None


def download_edit_model():
    """
    Pre-download the model from Hugging Face at startup.
    Call this once in app.py before launching the Gradio app so
    the first edit doesn't have to wait for a full download.
    You will see the download progress in your Colab logs.
    """
    print(f"[ImageEdit] Downloading {MODEL_ID} from Hugging Face ...")
    snapshot_download(
        repo_id=MODEL_ID,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print("[ImageEdit] Download complete.")


def load_edit_pipeline():
    """Load (or return already-loaded) pipeline."""
    global _edit_pipe
    if _edit_pipe is not None:
        return _edit_pipe

    from diffusers.models import QwenImageTransformer2DModel
    from diffusers import QwenImageEditPlusPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print("[ImageEdit] Loading transformer ...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    print("[ImageEdit] Loading pipeline ...")
    _edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    print("[ImageEdit] Pipeline ready.")
    return _edit_pipe


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_dimensions(pil_image):
    """
    Resize to max 1024px on the longest side keeping aspect ratio,
    then snap both dimensions to multiples of 8.
    """
    w, h = pil_image.size
    if w > h:
        new_w = 1024
        new_h = int(1024 * h / w)
    else:
        new_h = 1024
        new_w = int(1024 * w / h)
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)
    return new_w, new_h


def _load_gallery_images(gallery):
    """Convert Gradio gallery output to a list of PIL RGB images."""
    pil_images = []
    if not gallery:
        return pil_images
    for item in gallery:
        try:
            if isinstance(item, (tuple, list)):
                path_or_img = item[0]
            else:
                path_or_img = item
            if isinstance(path_or_img, str):
                pil_images.append(Image.open(path_or_img).convert("RGB"))
            elif isinstance(path_or_img, Image.Image):
                pil_images.append(path_or_img.convert("RGB"))
            else:
                pil_images.append(Image.open(path_or_img.name).convert("RGB"))
        except Exception as e:
            print(f"[ImageEdit] Skipping invalid image: {e}")
    return pil_images


# ── Core inference ────────────────────────────────────────────────────────────
def run_image_edit(
    gallery,
    prompt,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pil_images = _load_gallery_images(gallery)

    if not pil_images:
        raise gr.Error("Please upload at least one image to edit.")
    if not prompt.strip():
        raise gr.Error("Please enter an edit instruction.")

    pipe   = load_edit_pipeline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    width, height = _get_dimensions(pil_images[0])

    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, "
        "text, error, missing fingers, extra digit, fewer digits, "
        "cropped, jpeg artifacts, signature, watermark, username, blurry"
    )

    result = pipe(
        image=pil_images,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=guidance_scale,
        generator=generator,
    ).images[0]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, seed


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def image_edit_tab():
    """
    Renders the Image Edit tab contents inside your gr.Blocks().

    In app.py:
    ───────────────────────────────────────────────────────
    # 1. Add this import near the top (after other imports):
    from image_edit_tab import image_edit_tab, download_edit_model

    # 2. Trigger model download at startup (shows progress in Colab logs):
    download_edit_model()

    # 3. Add the tab inside your gr.Blocks() tabs:
    with gr.Tab("🖊️ Image Edit"):
        edit_gallery = image_edit_tab()

    # 4. (Optional) Wire a send button from the Upscaler tab:
    #    Place this AFTER the Image Edit tab definition:
    send_to_edit_btn = gr.Button("📤 Send to Image Edit")
    send_to_edit_btn.click(
        fn=lambda img: [[img]],
        inputs=[result_up_tab],
        outputs=[edit_gallery],
    )
    ───────────────────────────────────────────────────────
    """
    gr.Markdown("### 🖊️ Image Edit — FireRed-Image-Edit (Qwen)")
    gr.Markdown(
        "Upload **1–3 images** and describe the edit you want. "
        "For multi-image edits, reference images by number in your prompt "
        "*(e.g. 'Replace her glasses with the glasses from image 2')*."
    )

    with gr.Row(equal_height=True):

        with gr.Column():
            gallery = gr.Gallery(
                label="Input Image(s)",
                type="filepath",
                columns=2,
                rows=1,
                height=320,
                allow_preview=True,
            )

            prompt = gr.Textbox(
                label="Edit Instruction",
                placeholder=(
                    'e.g. "Make the sky look like a sunset" '
                    'or "Transform into anime style" '
                    'or "Replace her glasses with the glasses from image 2"'
                ),
                lines=3,
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0, maximum=MAX_SEED, step=1, value=0,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0, maximum=10.0, step=0.1, value=1.0,
                )
                num_inference_steps = gr.Slider(
                    label="Steps",
                    minimum=1, maximum=50, step=1, value=4,
                    info="4 steps recommended for this Fast model.",
                )

            run_btn = gr.Button("✏️ Edit Image", variant="primary")

        with gr.Column():
            output_image = gr.Image(
                label="Edited Image",
                interactive=False,
                format="png",
                height=395,
            )
            used_seed = gr.Number(label="Seed Used", interactive=False)

    run_btn.click(
        fn=run_image_edit,
        inputs=[gallery, prompt, seed, randomize_seed, guidance_scale, num_inference_steps],
        outputs=[output_image, used_seed],
    )

    # Return gallery so other tabs can send images here
    return gallery
