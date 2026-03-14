import spaces
import os
from argparse import ArgumentParser
from stablepy import (
    Model_Diffusers,
    SCHEDULE_TYPE_OPTIONS,
    SCHEDULE_PREDICTION_TYPE_OPTIONS,
    check_scheduler_compatibility,
    TASK_AND_PREPROCESSORS,
    FACE_RESTORATION_MODELS,
    scheduler_names,
    PROMPT_WEIGHT_OPTIONS_PRIORITY,
)
from constants import (
    DIRECTORY_MODELS,
    DIRECTORY_LORAS,
    DIRECTORY_VAES,
    DIRECTORY_EMBEDS,
    DIRECTORY_UPSCALERS,
    DOWNLOAD_MODEL,
    DOWNLOAD_VAE,
    DOWNLOAD_LORA,
    LOAD_DIFFUSERS_FORMAT_MODEL,
    DIFFUSERS_FORMAT_LORAS,
    DOWNLOAD_EMBEDS,
    CIVITAI_API_KEY,
    HF_TOKEN,
    TASK_STABLEPY,
    TASK_MODEL_LIST,
    UPSCALER_DICT_GUI,
    UPSCALER_KEYS,
    WARNING_MSG_VAE,
    SDXL_TASK,
    MODEL_TYPE_TASK,
    POST_PROCESSING_SAMPLER,
    SUBTITLE_GUI,
    HELP_GUI,
    EXAMPLES_GUI_HELP,
    EXAMPLES_GUI,
    RESOURCES,
    DIFFUSERS_CONTROLNET_MODEL,
    IP_MODELS,
    MODE_IP_OPTIONS,
    CACHE_HF_ROOT,
    CACHE_HF,
)
from stablepy.diffusers_vanilla.style_prompt_config import STYLE_NAMES
import torch
import re
import time
from PIL import ImageFile
from utils import (
    download_things,
    get_model_list,
    extract_parameters,
    get_my_lora,
    get_model_type,
    extract_exif_data,
    create_mask_now,
    download_diffuser_repo,
    get_used_storage_gb,
    delete_model,
    progress_step_bar,
    html_template_message,
    escape_html,
    clear_hf_cache,
)
from image_processor import preprocessor_tab
from datetime import datetime
import gradio as gr
import logging
import diffusers
import warnings
from stablepy import logger
from diffusers import FluxPipeline
import subprocess

IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
HIDE_API = bool(os.getenv("HIDE_API"))
if IS_ZERO_GPU:
    subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
IS_GPU_MODE = True if IS_ZERO_GPU else (True if torch.cuda.is_available() else False)
img_path = "./images/"
allowed_path = os.path.abspath(img_path)
delete_cache_time = (9600, 9600) if IS_ZERO_GPU else (86400, 86400)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True

directories = [DIRECTORY_MODELS, DIRECTORY_LORAS, DIRECTORY_VAES, DIRECTORY_EMBEDS, DIRECTORY_UPSCALERS]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Download stuffs
for url in [url.strip() for url in DOWNLOAD_MODEL.split(',')]:
    download_things(DIRECTORY_MODELS, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in DOWNLOAD_VAE.split(',')]:
    download_things(DIRECTORY_VAES, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in DOWNLOAD_LORA.split(',')]:
    download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY)

# Download Embeddings
for url_embed in DOWNLOAD_EMBEDS:
    download_things(DIRECTORY_EMBEDS, url_embed, HF_TOKEN, CIVITAI_API_KEY)

# Build list models
embed_list = get_model_list(DIRECTORY_EMBEDS)
embed_list = [
    (os.path.splitext(os.path.basename(emb))[0], emb) for emb in embed_list
]
single_file_model_list = get_model_list(DIRECTORY_MODELS)
model_list = LOAD_DIFFUSERS_FORMAT_MODEL + single_file_model_list
lora_model_list = get_model_list(DIRECTORY_LORAS)
lora_model_list.insert(0, "None")
lora_model_list = lora_model_list + DIFFUSERS_FORMAT_LORAS
vae_model_list = get_model_list(DIRECTORY_VAES)
vae_model_list.insert(0, "BakedVAE")
vae_model_list.insert(0, "None")

print('\033[33m🏁 Download and listing of valid models completed.\033[0m')

components = None
if IS_ZERO_GPU:
    flux_repo = "camenduru/FLUX.1-dev-diffusers"
    flux_pipe = FluxPipeline.from_pretrained(
        flux_repo,
        transformer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    components = flux_pipe.components
    delete_model(flux_repo)

#######################
# GUI
#######################
logging.getLogger("diffusers").setLevel(logging.ERROR)
diffusers.utils.logging.set_verbosity(40)
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")

parser = ArgumentParser(description='DiffuseCraft: Create images from text prompts.', add_help=True)
parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
parser.add_argument('--theme', type=str, default="NoCrypt/miku", help='Set the theme (default: NoCrypt/miku)')
parser.add_argument("--ssr", action="store_true", help="Enable SSR (Server-Side Rendering)")
parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level (default: INFO)")
args = parser.parse_args()

logger.setLevel(
    "INFO" if IS_ZERO_GPU else getattr(logging, args.log_level.upper())
)

CSS = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#gallery { flex-grow: 1; }
#load_model { height: 50px; }
"""


def lora_chk(lora_):
    if isinstance(lora_, str) and lora_.strip() not in ["", "None"]:
        return lora_
    return None


class GuiSD:
    def __init__(self, stream=True):
        self.model = None
        self.status_loading = False
        self.sleep_loading = 4
        self.last_load = datetime.now()
        self.inventory = []

    def update_storage_models(self, storage_floor_gb=30, required_inventory_for_purge=3):
        while get_used_storage_gb() > storage_floor_gb:
            if len(self.inventory) < required_inventory_for_purge:
                break
            removal_candidate = self.inventory.pop(0)
            delete_model(removal_candidate)

        lowPrioCleanup = max((datetime.now() - self.last_load).total_seconds(), 0) > 60
        if lowPrioCleanup and (len(self.inventory) >= required_inventory_for_purge - 1) and not self.status_loading and get_used_storage_gb(CACHE_HF_ROOT) > (storage_floor_gb * 2):
            print("Cleaning up Hugging Face cache...")
            clear_hf_cache()
            self.inventory = [
                m for m in self.inventory if os.path.exists(m)
            ]

    def update_inventory(self, model_name):
        if model_name not in single_file_model_list:
            self.inventory = [
                m for m in self.inventory if m != model_name
            ] + [model_name]
        print(self.inventory)

    def load_new_model(self, model_name, vae_model, task, controlnet_model, progress=gr.Progress(track_tqdm=True)):

        if model_name.startswith("http"):
            yield f"Downloading model: {model_name}"
            model_name = download_things(DIRECTORY_MODELS, model_name, HF_TOKEN, CIVITAI_API_KEY)
            if not model_name:
                raise ValueError("Error retrieving model information from URL")

        if IS_ZERO_GPU:
            self.update_storage_models()

        vae_model = vae_model if vae_model != "None" else None
        model_type = get_model_type(model_name)
        dtype_model = torch.bfloat16 if model_type == "FLUX" else torch.float16

        if not os.path.exists(model_name):
            logger.debug(f"model_name={model_name}, vae_model={vae_model}, task={task}, controlnet_model={controlnet_model}")
            _ = download_diffuser_repo(
                repo_name=model_name,
                model_type=model_type,
                revision="main",
                token=True,
            )

        self.update_inventory(model_name)

        for i in range(68):
            if not self.status_loading:
                self.status_loading = True
                if i > 0:
                    time.sleep(self.sleep_loading)
                    print("Previous model ops...")
                break
            time.sleep(0.5)
            print(f"Waiting queue {i}")
            yield "Waiting queue"

        self.status_loading = True

        yield f"Loading model: {model_name}"

        if vae_model == "BakedVAE":
            vae_model = model_name
        elif vae_model:
            vae_type = "SDXL" if "sdxl" in vae_model.lower() else "SD 1.5"
            if model_type != vae_type:
                gr.Warning(WARNING_MSG_VAE)

        print("Loading model...")

        try:
            start_time = time.time()

            if self.model is None:
                self.model = Model_Diffusers(
                    base_model_id=model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    controlnet_model=controlnet_model,
                    device="cpu" if IS_ZERO_GPU else None,
                    env_components=components,
                )
                self.model.advanced_params(image_preprocessor_cuda_active=IS_GPU_MODE)
            else:
                if self.model.base_model_id != model_name:
                    load_now_time = datetime.now()
                    elapsed_time = max((load_now_time - self.last_load).total_seconds(), 0)

                    if elapsed_time <= 9:
                        print("Waiting for the previous model's time ops...")
                        time.sleep(9 - elapsed_time)

                if IS_ZERO_GPU:
                    self.model.device = torch.device("cpu")
                self.model.load_pipe(
                    model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    controlnet_model=controlnet_model,
                )

            end_time = time.time()
            self.sleep_loading = max(min(int(end_time - start_time), 10), 4)
        except Exception as e:
            self.last_load = datetime.now()
            self.status_loading = False
            self.sleep_loading = 4
            raise e

        self.last_load = datetime.now()
        self.status_loading = False

        yield f"Model loaded: {model_name}"

    @torch.inference_mode()
    def generate_pipeline(
        self,
        prompt,
        neg_prompt,
        num_images,
        steps,
        cfg,
        clip_skip,
        seed,
        lora1, lora_scale1,
        lora2, lora_scale2,
        lora3, lora_scale3,
        lora4, lora_scale4,
        lora5, lora_scale5,
        lora6, lora_scale6,
        lora7, lora_scale7,
        sampler,
        schedule_type,
        schedule_prediction_type,
        img_height,
        img_width,
        model_name,
        vae_model,
        task,
        image_control,
        preprocessor_name,
        preprocess_resolution,
        image_resolution,
        style_prompt,
        style_json_file,
        image_mask,
        strength,
        low_threshold,
        high_threshold,
        value_threshold,
        distance_threshold,
        recolor_gamma_correction,
        tile_blur_sigma,
        controlnet_output_scaling_in_unet,
        controlnet_start_threshold,
        controlnet_stop_threshold,
        textual_inversion,
        syntax_weights,
        upscaler_model_path,
        upscaler_increases_size,
        upscaler_tile_size,
        upscaler_tile_overlap,
        hires_steps,
        hires_denoising_strength,
        hires_sampler,
        hires_prompt,
        hires_negative_prompt,
        hires_before_adetailer,
        hires_after_adetailer,
        hires_schedule_type,
        hires_guidance_scale,
        controlnet_model,
        loop_generation,
        leave_progress_bar,
        disable_progress_bar,
        image_previews,
        display_images,
        save_generated_images,
        filename_pattern,
        image_storage_location,
        retain_compel_previous_load,
        retain_detailfix_model_previous_load,
        retain_hires_model_previous_load,
        t2i_adapter_preprocessor,
        t2i_adapter_conditioning_scale,
        t2i_adapter_conditioning_factor,
        xformers_memory_efficient_attention,
        freeu,
        generator_in_cpu,
        adetailer_inpaint_only,
        adetailer_verbose,
        adetailer_sampler,
        adetailer_active_a,
        prompt_ad_a,
        negative_prompt_ad_a,
        strength_ad_a,
        face_detector_ad_a,
        person_detector_ad_a,
        hand_detector_ad_a,
        mask_dilation_a,
        mask_blur_a,
        mask_padding_a,
        adetailer_active_b,
        prompt_ad_b,
        negative_prompt_ad_b,
        strength_ad_b,
        face_detector_ad_b,
        person_detector_ad_b,
        hand_detector_ad_b,
        mask_dilation_b,
        mask_blur_b,
        mask_padding_b,
        retain_task_cache_gui,
        guidance_rescale,
        image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1,
        image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2,
        pag_scale,
        face_restoration_model,
        face_restoration_visibility,
        face_restoration_weight,
    ):
        info_state = html_template_message("Navigating latent space...")
        yield info_state, gr.update(), gr.update()

        vae_model = vae_model if vae_model != "None" else None
        loras_list = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
        vae_msg = f"VAE: {vae_model}" if vae_model else ""
        msg_lora = ""

        logger.debug(f"Config model: {model_name}, {vae_model}, {loras_list}")

        task = TASK_STABLEPY[task]

        params_ip_img = []
        params_ip_msk = []
        params_ip_model = []
        params_ip_mode = []
        params_ip_scale = []

        all_adapters = [
            (image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1),
            (image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2),
        ]

        if not hasattr(self.model.pipe, "transformer"):
            for imgip, mskip, modelip, modeip, scaleip in all_adapters:
                if imgip:
                    params_ip_img.append(imgip)
                    if mskip:
                        params_ip_msk.append(mskip)
                    params_ip_model.append(modelip)
                    params_ip_mode.append(modeip)
                    params_ip_scale.append(scaleip)

        concurrency = 5
        self.model.stream_config(concurrency=concurrency, latent_resize_by=1, vae_decoding=False)

        if task != "txt2img" and not image_control:
            raise ValueError("Reference image is required. Please upload one in 'Image ControlNet/Inpaint/Img2img'.")

        if task in ["inpaint", "repaint"] and not image_mask:
            raise ValueError("Mask image not found. Upload one in 'Image Mask' to proceed.")

        if "https://" not in str(UPSCALER_DICT_GUI[upscaler_model_path]):
            upscaler_model = upscaler_model_path
        else:
            url_upscaler = UPSCALER_DICT_GUI[upscaler_model_path]
            if not os.path.exists(f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"):
                download_things(DIRECTORY_UPSCALERS, url_upscaler, HF_TOKEN)
            upscaler_model = f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"

        logging.getLogger("ultralytics").setLevel(logging.INFO if adetailer_verbose else logging.ERROR)

        adetailer_params_A = {
            "face_detector_ad": face_detector_ad_a,
            "person_detector_ad": person_detector_ad_a,
            "hand_detector_ad": hand_detector_ad_a,
            "prompt": prompt_ad_a,
            "negative_prompt": negative_prompt_ad_a,
            "strength": strength_ad_a,
            "mask_dilation": mask_dilation_a,
            "mask_blur": mask_blur_a,
            "mask_padding": mask_padding_a,
            "inpaint_only": adetailer_inpaint_only,
            "sampler": adetailer_sampler,
        }

        adetailer_params_B = {
            "face_detector_ad": face_detector_ad_b,
            "person_detector_ad": person_detector_ad_b,
            "hand_detector_ad": hand_detector_ad_b,
            "prompt": prompt_ad_b,
            "negative_prompt": negative_prompt_ad_b,
            "strength": strength_ad_b,
            "mask_dilation": mask_dilation_b,
            "mask_blur": mask_blur_b,
            "mask_padding": mask_padding_b,
        }

        pipe_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "img_height": img_height,
            "img_width": img_width,
            "num_images": num_images,
            "num_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "pag_scale": float(pag_scale),
            "seed": seed,
            "image": image_control,
            "preprocessor_name": preprocessor_name,
            "preprocess_resolution": preprocess_resolution,
            "image_resolution": image_resolution,
            "style_prompt": style_prompt if style_prompt else "",
            "style_json_file": "",
            "image_mask": image_mask,
            "strength": strength,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "value_threshold": value_threshold,
            "distance_threshold": distance_threshold,
            "recolor_gamma_correction": float(recolor_gamma_correction),
            "tile_blur_sigma": int(tile_blur_sigma),
            "lora_A": lora_chk(lora1), "lora_scale_A": lora_scale1,
            "lora_B": lora_chk(lora2), "lora_scale_B": lora_scale2,
            "lora_C": lora_chk(lora3), "lora_scale_C": lora_scale3,
            "lora_D": lora_chk(lora4), "lora_scale_D": lora_scale4,
            "lora_E": lora_chk(lora5), "lora_scale_E": lora_scale5,
            "lora_F": lora_chk(lora6), "lora_scale_F": lora_scale6,
            "lora_G": lora_chk(lora7), "lora_scale_G": lora_scale7,
            "textual_inversion": embed_list if textual_inversion else [],
            "syntax_weights": syntax_weights,
            "sampler": sampler,
            "schedule_type": schedule_type,
            "schedule_prediction_type": schedule_prediction_type,
            "xformers_memory_efficient_attention": xformers_memory_efficient_attention,
            "gui_active": True,
            "loop_generation": loop_generation,
            "controlnet_conditioning_scale": float(controlnet_output_scaling_in_unet),
            "control_guidance_start": float(controlnet_start_threshold),
            "control_guidance_end": float(controlnet_stop_threshold),
            "generator_in_cpu": generator_in_cpu,
            "FreeU": freeu,
            "adetailer_A": adetailer_active_a,
            "adetailer_A_params": adetailer_params_A,
            "adetailer_B": adetailer_active_b,
            "adetailer_B_params": adetailer_params_B,
            "leave_progress_bar": leave_progress_bar,
            "disable_progress_bar": disable_progress_bar,
            "image_previews": image_previews,
            "display_images": display_images,
            "save_generated_images": save_generated_images,
            "filename_pattern": filename_pattern,
            "image_storage_location": image_storage_location,
            "retain_compel_previous_load": retain_compel_previous_load,
            "retain_detailfix_model_previous_load": retain_detailfix_model_previous_load,
            "retain_hires_model_previous_load": retain_hires_model_previous_load,
            "t2i_adapter_preprocessor": t2i_adapter_preprocessor,
            "t2i_adapter_conditioning_scale": float(t2i_adapter_conditioning_scale),
            "t2i_adapter_conditioning_factor": float(t2i_adapter_conditioning_factor),
            "upscaler_model_path": upscaler_model,
            "upscaler_increases_size": upscaler_increases_size,
            "upscaler_tile_size": upscaler_tile_size,
            "upscaler_tile_overlap": upscaler_tile_overlap,
            "hires_steps": hires_steps,
            "hires_denoising_strength": hires_denoising_strength,
            "hires_prompt": h
