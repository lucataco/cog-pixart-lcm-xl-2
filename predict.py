# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import torch
from typing import List
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE

MODEL_NAME = "PixArt-alpha/PixArt-LCM-XL-2-1024-MS"
MODEL_CACHE = "model-cache"
VAE_NAME = "openai/consistency-decoder"
VAE_CACHE="vae-cache"

style_list = [
    {
        "name": "None",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel Art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy Art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

def apply_style(style, prompt, negative_prompt):
    if style == "None":
        return prompt, negative_prompt
    else:
        for style_dict in style_list:
            if style_dict["name"] == style:
                return style_dict["prompt"].format(prompt=prompt), style_dict["negative_prompt"]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Using DALL-E 3 Consistency Decoder")
        vae = ConsistencyDecoderVAE.from_pretrained(
            VAE_NAME,
            torch_dtype=torch.float16,
            cache_dir=VAE_CACHE
        )
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=MODEL_CACHE
        )
        # speed-up T5
        pipe.text_encoder.to_bettertransformer()
        self.pipe = pipe.to("cuda")


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A small cactus with a happy face in the Sahara desert"),
        negative_prompt: str = Input(description="Negative prompt", default=None),
        style: str = Input(
            description="Image style",
            choices=["None", "Cinematic", "Photographic", "Anime", "Manga", "Digital Art", "Pixel Art", "Fantasy Art", "Neonpunk", "3D Model"],
            default="None",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=20, default=4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
        print("Prompt:", prompt, " Negative Prompt:", negative_prompt)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=0.,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=num_outputs,
            output_type="pil",
        )

        output_paths = []
        for i, img in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            img.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths