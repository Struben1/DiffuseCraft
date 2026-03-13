"""
image_edit_tab.py  –  DiffuseCraft Image Editing Tab
Model: prithivMLmods/FireRed-Image-Edit-1.0-Fast
"""

import gc
import random

import gradio as gr
import numpy as np
import torch
from PIL import Image

MODEL_ID = "prithivMLmods/FireRed-Image-Edit-1.0-Fast"
MAX_SEED = np.iinfo(np.int32).max

_edit_pipe = None


def load_edit_pipeline():
    """Load (or return cached) QwenImageEditPlusPipeline."""
    global _edit_pipe
    if _edit_pipe is not None:
        return _edit_pipe

    from diffusers import QwenImageEditPlusPipeline
    from diffusers.models import QwenImageTransformer2DModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[ImageEdit] Loading transformer from {MODEL_ID} ...")
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


def _round64(v):
    return max(64, round(v / 64) * 64)


def auto_size_from_image(image):
    """Return (width, height) snapped to multiples of 64, max 1280."""
    if image is None:
        return 1024, 1024
    w, h = Image.fromarray(image).size
    scale = min(1.0, 1280 / max(w, h))
    return _round64(int(w * scale)), _round64(int(h * scale))


def run_image_edit(
    input_image,
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
    width,
    height,
):
    if input_image is None:
        raise gr.Error("Please upload or send an image to edit.")
    if not prompt.strip():
        raise gr.Error("Please enter an edit instruction.")

    pipe = load_edit_pipeline()

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(int(seed))

    pil_image = Image.fromarray(input_image).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )

    result = pipe(
        image=pil_image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    )

    output_image = result.images[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return output_image, seed


def image_edit_tab():
    """
    Renders the Image Edit tab contents.
    Call this inside:  with gr.Tab("🖊️ Image Edit"):
        image_edit_input = image_edit_tab()
    The returned component lets other tabs send images here.
    """
    gr.Markdown("### 🖊️ Image Edit — powered by FireRed-Image-Edit-1.0-Fast (Qwen)")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image  (upload or use 'Send to Image Edit' from another tab)",
                type="numpy",
                height=420,
            )

            prompt = gr.Textbox(
                label="Edit Instruction",
                placeholder='e.g. "Make the sky look like a sunset" or "Add a hat to the person"',
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt (optional)",
                placeholder="blurry, low quality, artifacts",
                lines=2,
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)

                guidance_scale = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0, maximum=10.0, step=0.5, value=3.5,
                )
                num_inference_steps = gr.Slider(
                    label="Steps",
                    minimum=1, maximum=30, step=1, value=4,
                    info="4 steps recommended for this Fast model.",
                )
                with gr.Row():
                    width  = gr.Slider(label="Width",  minimum=256, maximum=1280, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=1280, step=64, value=1024)

            run_btn = gr.Button("✏️ Edit Image", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Edited Image", type="pil",
                height=420, format="png", interactive=False,
            )
            used_seed = gr.Number(label="Seed Used", interactive=False)

    # Auto-fill width/height sliders when an image is uploaded
    input_image.upload(
        fn=auto_size_from_image,
        inputs=[input_image],
        outputs=[width, height],
    )

    run_btn.click(
        fn=run_image_edit,
        inputs=[
            input_image, prompt, negative_prompt,
            seed, randomize_seed, guidance_scale,
            num_inference_steps, width, height,
        ],
        outputs=[output_image, used_seed],
    )

    # Return the input component so other tabs can wire a "Send to Image Edit" button
    return input_image
