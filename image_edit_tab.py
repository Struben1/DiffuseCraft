"""
image_edit_tab.py  -  DiffuseCraft Image Editing Tab
Uses the official FireRed-Image-Edit HF Space API via gradio_client.
No local model loading = no diffusers version conflict with stablepy.
"""

import random
import tempfile
import os

import gradio as gr
import numpy as np
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max

# The official public HF Space
HF_SPACE = "prithivMLmods/FireRed-Image-Edit-1.0-Fast"


def _get_client():
    from gradio_client import Client
    return Client(HF_SPACE)


def _pil_to_tempfile(pil_image):
    """Save a PIL image to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_image.save(tmp.name)
    return tmp.name


def _load_gallery_images(gallery):
    pil_images = []
    if not gallery:
        return pil_images
    for item in gallery:
        try:
            path = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(path, str):
                pil_images.append(Image.open(path).convert("RGB"))
            elif isinstance(path, Image.Image):
                pil_images.append(path.convert("RGB"))
            else:
                pil_images.append(Image.open(path.name).convert("RGB"))
        except Exception as e:
            print(f"[ImageEdit] Skipping image: {e}")
    return pil_images


def run_image_edit(
    gallery,
    prompt,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
):
    pil_images = _load_gallery_images(gallery)
    if not pil_images:
        raise gr.Error("Please upload at least one image.")
    if not prompt.strip():
        raise gr.Error("Please enter an edit instruction.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Save images to temp files so gradio_client can upload them
    temp_paths = [_pil_to_tempfile(img) for img in pil_images]

    try:
        client = _get_client()

        result = client.predict(
            images=temp_paths,
            prompt=prompt,
            seed=seed,
            randomize_seed=False,
            guidance_scale=guidance_scale,
            steps=num_inference_steps,
            api_name="/infer",
        )

        # result is (output_image_path, seed_used)
        output_path = result[0] if isinstance(result, (list, tuple)) else result
        output_seed = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else seed

        output_image = Image.open(output_path).convert("RGB")
        return output_image, output_seed

    except Exception as e:
        raise gr.Error(f"Image edit failed: {str(e)}")
    finally:
        # Clean up temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass


def download_edit_model():
    """No-op — model runs on HF Space, nothing to download locally."""
    print("[ImageEdit] Using HF Space API — no local download needed.")


def image_edit_tab():
    gr.Markdown("### 🖊️ Image Edit — FireRed-Image-Edit-1.0 (via HF Space)")
    gr.Markdown(
        "Upload **1–3 images** and describe your edit. "
        "Uses the official FireRed HF Space — no local GPU needed for this tab. "
        "For multi-image edits, reference images by number "
        "*(e.g. 'Replace her glasses with the glasses from image 2')*."
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            gallery = gr.Gallery(
                label="Input Image(s)",
                type="filepath",
                columns=2, rows=1,
                height=320,
                allow_preview=True,
            )
            prompt = gr.Textbox(
                label="Edit Instruction",
                placeholder='e.g. "Make his hair white" or "Transform into anime style"',
                lines=3,
            )
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                guidance_scale = gr.Slider(
                    label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0
                )
                num_inference_steps = gr.Slider(
                    label="Steps", minimum=1, maximum=50, step=1, value=4,
                    info="4 steps recommended.",
                )
            run_btn = gr.Button("✏️ Edit Image", variant="primary")

        with gr.Column():
            output_image = gr.Image(
                label="Edited Image", interactive=False, format="png", height=395
            )
            used_seed = gr.Number(label="Seed Used", interactive=False)

    run_btn.click(
        fn=run_image_edit,
        inputs=[gallery, prompt, seed, randomize_seed, guidance_scale, num_inference_steps],
        outputs=[output_image, used_seed],
    )

    return gallery
    
