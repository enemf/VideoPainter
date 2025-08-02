from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from diffusers import CogVideoXI2VDualInpaintAnyLPipeline
from diffusers.utils import load_video, export_to_video
import torch
import tempfile
import os

app = FastAPI()

# Load model once at startup
pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained("THUDM/CogVideoX-5b")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

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
