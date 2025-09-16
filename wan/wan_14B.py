import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    loop.run_until_complete(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS


def run_inpainting_wan_14(
    input_video: str,
    input_mask: str,
    output_prefix: str,
    output_frames: int,
    prompt: str = "A realistic video.",
    output_width: int = 320,
    output_height: int = 240,
    encode_batch_size: int = 1,
):
    import_custom_nodes()
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,过曝，",
            clip=get_value_at_index(cliploader_38, 0),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        loadvideo = NODE_CLASS_MAPPINGS["LoadVideo"]()
        loadvideo_209 = loadvideo.load_video(file=input_video)

        getvideocomponents = NODE_CLASS_MAPPINGS["GetVideoComponents"]()
        getvideocomponents_210 = getvideocomponents.get_components(
            video=get_value_at_index(loadvideo_209, 0)
        )

        loadvideo_229 = loadvideo.load_video(file=input_mask)

        orig_img = getvideocomponents.get_components(
            video=get_value_at_index(loadvideo_229, 0)
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        orig_mask = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(orig_img, 0)
        )

        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        inv_mask = invertmask.invert(mask=get_value_at_index(orig_mask, 0))

        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        inv_img = masktoimage.mask_to_image(
            mask=get_value_at_index(inv_mask, 0)
        )

        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        imagecompositemasked_236 = imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=True,
            destination=get_value_at_index(getvideocomponents_210, 0),
            source=get_value_at_index(orig_img, 0),
            mask=get_value_at_index(inv_mask, 0),
        )

        wanvacetovideo = NODE_CLASS_MAPPINGS["WanVaceToVideo"]()
        wanvacetovideo_49 = wanvacetovideo.encode(
            width=output_width,
            height=output_height,
            length=output_frames,
            batch_size=encode_batch_size,
            strength=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_39, 0),
            control_video=get_value_at_index(imagecompositemasked_236, 0),
            control_masks=get_value_at_index(inv_mask, 0),
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_140 = unetloader.load_unet(
            unet_name="wan2.1_vace_14B_fp16.safetensors", weight_dtype="fp8_e4m3fn_fast"
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_154 = loraloadermodelonly.load_lora_model_only(
            lora_name="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
            strength_model=0.30000000000000004,
            model=get_value_at_index(unetloader_140, 0),
        )

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        trimvideolatent = NODE_CLASS_MAPPINGS["TrimVideoLatent"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        savevideo = NODE_CLASS_MAPPINGS["SaveVideo"]()
        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()

        modelsamplingsd3_48 = modelsamplingsd3.patch(
            shift=5, model=get_value_at_index(loraloadermodelonly_154, 0)
        )

        ksampler_3 = ksampler.sample(
            seed=42,
            steps=4,
            cfg=1,
            sampler_name="uni_pc",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(modelsamplingsd3_48, 0),
            positive=get_value_at_index(wanvacetovideo_49, 0),
            negative=get_value_at_index(wanvacetovideo_49, 1),
            latent_image=get_value_at_index(wanvacetovideo_49, 2),
        )

        trimvideolatent_58 = trimvideolatent.op(
            trim_amount=get_value_at_index(wanvacetovideo_49, 3),
            samples=get_value_at_index(ksampler_3, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(trimvideolatent_58, 0),
            vae=get_value_at_index(vaeloader_39, 0),
        )

        createvideo_68 = createvideo.create_video(
            fps=get_value_at_index(getvideocomponents_210, 2),
            images=get_value_at_index(vaedecode_8, 0),
        )

        savevideo_69 = savevideo.save_video(
            filename_prefix=output_prefix,
            format="auto",
            codec="auto",
            video=get_value_at_index(createvideo_68, 0),
        )

        res = savevideo_69["ui"]["images"][0]
        return os.path.join(res["subfolder"], res["filename"])

        # getimagesize_211 = getimagesize.get_size(
        #     image=get_value_at_index(getvideocomponents_210, 0),
        #     unique_id=11767634729096250232,
        # )

        # getimagesize_231 = getimagesize.get_size(
        #     image=get_value_at_index(getvideocomponents_230, 0),
        #     unique_id=15437511462173506877,
        # )
