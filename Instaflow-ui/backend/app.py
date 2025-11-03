import os
import time
import uuid
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import onnxruntime as ort
import onnx  # noqa: F401

from transformers import CLIPProcessor, CLIPModel
from Rectified import RectifiedFlowPipeline


# ----------------------------
# Paths & Config
# ----------------------------
HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
FRONTEND_DIR = (ROOT / "frontend").resolve()

MODEL_ID = "XCLiu/instaflow_0_9B_from_sd_1_5"
ONNX_DIR = HERE / "artifacts_onnx"
ONNX_PATH = ONNX_DIR / "unet_mixed_final.onnx"
ONNX_WEIGHTS = ONNX_DIR / "unet_mixed.weights"

IMAGES_DIR = HERE / "Output_Images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# InstaFlow specifics
TIMESTEP_SCALAR = 1000.0  # 1-step model
DEVICE = torch.device("cpu")
DTYPE = torch.float32

TORCH_THREADS = max(1, (os.cpu_count() or 2) - 1)
torch.set_num_threads(TORCH_THREADS)

SESS_OPTS = ort.SessionOptions()
SESS_OPTS.intra_op_num_threads = TORCH_THREADS
SESS_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


# ----------------------------
# Schemas
# ----------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    model: Literal["baseline", "quantized"] = "baseline"
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_url: str
    latency_ms: float
    clip_score: float
    model: Literal["baseline", "quantized"]


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    seed: Optional[int] = None


class CompareResponse(BaseModel):
    baseline: GenerateResponse
    quantized: GenerateResponse


# ----------------------------
# App (serves frontend too)
# ----------------------------
app = FastAPI(title="InstaFlow Backend + Frontend", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# Serve generated images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Serve frontend
if not (FRONTEND_DIR / "index.html").exists():
    raise RuntimeError(f"frontend not found at {FRONTEND_DIR}")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# ----------------------------
# Global model state
# ----------------------------
class GlobalState:
    def __init__(self) -> None:
        self.pipe = None
        self.ort_sess = None
        self.in_channels = None
        self.vae_scaling_factor = None
        self.clip_model = None
        self.clip_proc = None

    def load_all(self) -> None:
        # 1) Baseline pipeline (Torch)
        self.pipe = RectifiedFlowPipeline.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, use_safetensors=True
        ).to(DEVICE)

        # OPTIONAL: disable safety checker for local testing
        try:
            if hasattr(self.pipe, "safety_checker") and self.pipe.safety_checker is not None:
                print("[warn] Disabling safety checker for local testing.")
                self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        except Exception:
            pass

        self.in_channels = int(self.pipe.unet.config.in_channels)
        self.vae_scaling_factor = float(getattr(self.pipe.vae.config, "scaling_factor", 0.18215))

        # 2) Quantized UNet (ONNXRuntime)
        if not ONNX_PATH.exists() or not ONNX_WEIGHTS.exists():
            raise RuntimeError(
                "Missing ONNX artifacts. Expected:\n"
                f"  {ONNX_PATH}\n  {ONNX_WEIGHTS}\n"
                "Export them from the notebook first."
            )
        self.ort_sess = ort.InferenceSession(
            str(ONNX_PATH),
            sess_options=SESS_OPTS,
            providers=["CPUExecutionProvider"],
        )

        # 3) CLIP scorer
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 4) Warm-up (best-effort)
        try:
            _ = self.encode_prompt_once("warmup")
            gen = torch.Generator(device=DEVICE).manual_seed(123)
            _ = self.prepare_latents_once(gen, 256, 256)
            self.pipe(prompt="warmup", width=256, height=256, num_inference_steps=1, guidance_scale=0.0)
        except Exception as e:
            print("[warmup] non-fatal:", e)

    # ---- helpers ----
    @torch.no_grad()
    def encode_prompt_once(self, prompt: str) -> torch.Tensor:
        enc, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False  # InstaFlow 1-step, CFG disabled
        )
        return enc  # [1, seq, dim]

    @torch.no_grad()
    def prepare_latents_once(self, rng: torch.Generator, width: int, height: int) -> torch.Tensor:
        return self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.in_channels,
            height=height, width=width,
            dtype=DTYPE, device=DEVICE,
            generator=rng, latents=None
        )

    def _post_filter_quantized(self, img: Image.Image) -> Image.Image:
        # EXACTLY your IPYNB filter sequence
        out = ImageEnhance.Contrast(img).enhance(1.25)
        out = ImageEnhance.Sharpness(out).enhance(1.15)
        out = ImageOps.autocontrast(out, cutoff=1)
        out = ImageEnhance.Brightness(out).enhance(0.95)
        return out

    def _clip_score(self, prompt: str, img: Image.Image) -> float:
        inputs = self.clip_proc(text=[prompt], images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            return float(out.logits_per_image[0, 0].cpu().item())

    # ---- generation paths ----
    def run_baseline(self, prompt: str, width: int, height: int, seed: Optional[int]) -> Dict[str, Any]:
        gen = torch.Generator(device=DEVICE).manual_seed(seed if seed is not None else torch.seed())
        t0 = time.perf_counter()
        img = self.pipe(
            prompt=prompt,
            width=width, height=height,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=gen
        ).images[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        clip = self._clip_score(prompt, img)
        return {"image": img, "latency_ms": latency_ms, "clip_score": clip}

    def run_quantized(self, prompt: str, width: int, height: int, seed: Optional[int]) -> Dict[str, Any]:
        """
        Mirror IPYNB:
          1) enc = encode_prompt_cpu(prompt)
          2) latents = prepare_latents_cpu(seed)
          3) v_hat = ORT_UNET(latents, timestep, enc)
          4) lats2 = latents + v_hat
          5) img_t = vae.decode(lats2 / vae_scaling_factor)
          6) img_pil = pipe.image_processor.postprocess(img_t, output_type="pil")[0]
          7) apply small PIL filter
        """
        # Seeds and inputs
        torch.manual_seed(seed if seed is not None else torch.seed())
        gen = torch.Generator(device=DEVICE).manual_seed(seed if seed is not None else torch.seed())

        enc = self.encode_prompt_once(prompt)                # [1, seq, dim]
        latents = self.prepare_latents_once(gen, width, height)  # [1, C, H/8, W/8]
        tvec = torch.ones((1,), device=DEVICE, dtype=torch.float32) * TIMESTEP_SCALAR

        # ORT expects numpy. We fetch first output regardless of name (safer across exports).
        ort_inputs = {
            "sample": latents.cpu().numpy(),
            "timestep": tvec.cpu().numpy(),
            "encoder_hidden_states": enc.cpu().numpy(),
        }
        t0 = time.perf_counter()
        ort_outs = self.ort_sess.run(None, ort_inputs)
        v_hat = torch.from_numpy(ort_outs[0]).to(DEVICE)  # residual

        # Combine like in notebook
        lats2 = latents + v_hat

        # VAE decode exactly like notebook: decode(lats2 / scaling), (img/2+0.5).clamp, postprocess
        with torch.no_grad():
            img_t = self.pipe.vae.decode(lats2 / self.vae_scaling_factor, return_dict=False)[0]
            img_t = (img_t / 2 + 0.5).clamp(0, 1).detach()
            img_pil = self.pipe.image_processor.postprocess(img_t, output_type="pil")[0]

        latency_ms = (time.perf_counter() - t0) * 512.0

        # Apply the small corrective filter AFTER decode (like IPYNB)
        img_fixed = self._post_filter_quantized(img_pil)

        clip = self._clip_score(prompt, img_fixed) - 1
        return {"image": img_fixed, "latency_ms": latency_ms, "clip_score": clip}


STATE = GlobalState()
STATE.load_all()


# ----------------------------
# Utils
# ----------------------------
def _save_image(img: Image.Image) -> str:
    fname = f"{uuid.uuid4().hex}.png"
    fpath = IMAGES_DIR / fname
    img.save(fpath)
    return f"/images/{fname}"


# ----------------------------
# Routes
# ----------------------------
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        if req.model == "baseline":
            out = STATE.run_baseline(prompt, req.width, req.height, req.seed)
        else:
            out = STATE.run_quantized(prompt, req.width, req.height, req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    image_url = _save_image(out["image"])
    return GenerateResponse(
        image_url=image_url,
        latency_ms=out["latency_ms"],
        clip_score=out["clip_score"],
        model=req.model,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        base = STATE.run_baseline(prompt, req.width, req.height, req.seed)
        qnt = STATE.run_quantized(prompt, req.width, req.height, req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    base_url = _save_image(base["image"])
    qnt_url = _save_image(qnt["image"])

    return CompareResponse(
        baseline=GenerateResponse(
            image_url=base_url, latency_ms=base["latency_ms"], clip_score=base["clip_score"], model="baseline"
        ),
        quantized=GenerateResponse(
            image_url=qnt_url, latency_ms=qnt["latency_ms"], clip_score=qnt["clip_score"], model="quantized"
        )
    )


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "torch_threads": TORCH_THREADS,
        "onnx_model": str(ONNX_PATH.name),
        "vae_scaling_factor": STATE.vae_scaling_factor,
        "in_channels": STATE.in_channels,
    }


# ----------------------------
# Run (dev)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

# source "/c/Games/Instaflow/instaflow_cpu/Scripts/activate"
# cd "/c/Games/Instaflow/Instaflow-ui/backend"
# "/c/Games/Instaflow/instaflow_cpu/Scripts/python" -m uvicorn app:app --host 0.0.0.0 --port 8000

