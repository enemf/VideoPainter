from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from diffusers import (
    CogVideoXI2VDualInpaintAnyLPipeline,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import load_video, export_to_video
import torch
import tempfile
import os

app = FastAPI()

# Load model once at startup. Allow overriding component paths via environment
# variables so preinstalled weights can be used without re-downloading from
# Hugging Face.
MODEL_PATH = os.getenv("INPAINT_MODEL_PATH", "THUDM/CogVideoX-5b")
BRANCH_PATH = os.getenv("INPAINT_BRANCH_PATH")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if BRANCH_PATH:
    # Load a separately saved inpainting branch if provided.
    branch = CogvideoXBranchModel.from_pretrained(
        BRANCH_PATH, torch_dtype=DTYPE
    ).to(DEVICE, dtype=DTYPE)
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        MODEL_PATH, branch=branch, torch_dtype=DTYPE
    )
else:
    # Otherwise construct a default branch from the base transformer.
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=DTYPE
    ).to(DEVICE, dtype=DTYPE)
    branch = CogvideoXBranchModel.from_transformer(
        transformer=transformer,
        num_layers=1,
        attention_head_dim=transformer.config.attention_head_dim,
        num_attention_heads=transformer.config.num_attention_heads,
        load_weights_from_transformer=True,
    ).to(DEVICE, dtype=DTYPE)
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        MODEL_PATH, branch=branch, transformer=transformer, torch_dtype=DTYPE
    )

pipe.to(DEVICE)

@app.post("/inpaint")
async def inpaint(
    prompt: str = Form(...),
    video: UploadFile = File(...),
    mask: UploadFile = File(...),
    height: int = Form(720),
    width: int = Form(1280),
    output_fps: int = Form(24),
):
    """Inpaint a video using CogVideoX pipeline.

    Parameters
    ----------
    prompt: str
        Text prompt to guide inpainting.
    video: UploadFile
        Input video file.
    mask: UploadFile
        Video mask file where masked regions will be inpainted.
    height: int
        Output video height. Defaults to 720.
    width: int
        Output video width. Defaults to 1280.
    output_fps: int
        Frames per second for exported video. Defaults to 24.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, video.filename)
        mask_path = os.path.join(tmpdir, mask.filename)
        with open(video_path, "wb") as f:
            f.write(await video.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())

        frames = load_video(video_path)
        mask_frames = load_video(mask_path)
        num_frames = len(frames)
        if num_frames > 49:
            raise HTTPException(status_code=400, detail="Video too long (limit 49 frames)")

        result = pipe(
            prompt=prompt,
            image=frames[0],
            video=frames,
            masks=mask_frames,
            height=height,
            width=width,
            num_frames=num_frames,
            output_type="np",
        ).frames[0]

        out_path = os.path.join(tmpdir, "output.mp4")
        export_to_video(result, out_path, fps=output_fps)

        return FileResponse(out_path, media_type="video/mp4")
