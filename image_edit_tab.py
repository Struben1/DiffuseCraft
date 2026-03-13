"""
image_edit_tab.py  -  DiffuseCraft Image Editing Tab
Uses local qwenimage/ folder - no diffusers version conflict.
Models:
  - FireRedTeam/FireRed-Image-Edit-1.1  (base pipeline)
  - prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19  (transformer)
"""

import gc
import random

import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

MAX_SEED = np.iinfo(np.int32).max

BASE_MODEL_ID        = "FireRedTeam/FireRed-Image-Edit-1.1"
TRANSFORMER_MODEL_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19"

_edit_pipe = None


def download_edit_model():
    """Pre-download both models from Hugging Face at startup."""
    try:
        print(f"[ImageEdit] Downloading {BASE_MODEL_ID} ...")
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print(f"[ImageEdit] Downloading {TRANSFORMER_MODEL_ID} ...")
        snapshot_download(
            repo_id=TRANSFORMER_MODEL_ID,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print("[ImageEdit] All downloads complete.")
    except Exception as e:
        print(f"[ImageEdit] Download warning: {e}")


def load_edit_pipeline():
    """Load (or return cached) pipeline using local qwenimage module."""
    global _edit_pipe
    if _edit_pipe is not None:
        return _edit_pipe

    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print("[ImageEdit] Loading transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        TRANSFORMER_MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else None,
    )

    print("[ImageEdit] Loading pipeline...")
    _edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL_ID,
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    try:
        _edit_pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        print("[ImageEdit] Flash Attention 3 processor set.")
    except Exception as e:
        print(f"[ImageEdit] FA3 not available, using default: {e}")

    print("[ImageEdit] Pipeline ready.")
    return _edit_pipe


def _get_dimensions(pil_image):
    w, h = pil_image.size
    if w > h:
        new_w, new_h = 1024, int(1024 * h / w)
    else:
        new_h, new_w = 1024, int(1024 * w / h)
    return max(8, (new_w // 8) * 8), max(8, (new_h // 8) * 8)


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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pil_images = _load_gallery_images(gallery)
    if not pil_images:
        raise gr.Error("Please upload at least one image.")
    if not prompt.strip():
        raise gr.Error("Please enter an edit instruction.")

    pipe   = load_edit_pipeline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator     = torch.Generator(device=device).manual_seed(int(seed))
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
        generator=generator,
        true_cfg_scale=guidance_scale,
    ).images[0]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, seed


def image_edit_tab():
    gr.Markdown("### 🖊️ Image Edit — FireRed-Image-Edit-1.1 (local GPU)")
    gr.Markdown(
        "Upload **1–3 images** and describe your edit. "
        "Runs locally on Colab GPU. "
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
    
