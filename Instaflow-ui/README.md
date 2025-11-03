# InstaFlow – Baseline vs Quantized (CPU)  
**IE 643 Course Project — “Quantization of InstaFlow Model”**  
Team: **Sota** • Members: **Akshat Kumar (22B4513), Anirudh Shinde (22B2181)**

This repo ships a **single-origin app**: a FastAPI backend that **also serves** the comparison-first frontend.  
You open one URL — `http://localhost:8000/` — type a prompt, and get **side-by-side 512×512 images** from:

- **Baseline**: Torch `RectifiedFlowPipeline` (original model on CPU)
- **Quantized**: Mixed-precision ONNXRuntime UNet (Conv static INT8 + MatMul dynamic) with Torch VAE decode  
  (exactly mirroring the notebook path: predict residual `v̂` → add to latents → VAE decode → small tone/sharpness fix)

Both runs return **latency** and a **CLIP score** (prompt–image alignment).

---

## ✨ Features

- One-click **Baseline vs Quantized** comparison at **512×512** (seed optional)
- No flicker front-end; images swap in at full size
- Images saved under `/backend/Output_Images/` and served at `/images/<uuid>.png`
- Health probe at `/healthz`
- All CPU; easy to run on any Windows machine with a Python venv

---

## 📦 Folder Structure

```

Instaflow-ui/
├─ backend/
│  ├─ app.py                        # FastAPI backend + serves the frontend
│  ├─ Rectified.py                  # Provides RectifiedFlowPipeline (imported by app.py)
│  ├─ artifacts_onnx/               # ONNX artifacts (from your notebook export)
│  │  ├─ unet_mixed_final.onnx
│  │  └─ unet_mixed.weights
│  ├─ Output_Images/                # Generated images (auto-created)
│  └─ tests_out/                    # (optional) from test script, if you use one
└─ frontend/
├─ index.html                    # Comparison-first UI
├─ styles.css                    # Minimal, responsive styling
└─ app.js                        # Calls POST /compare and renders both images

````

> **Important:** The two ONNX files **must** exist in `backend/artifacts_onnx/` with **exact** filenames:
> - `unet_mixed_final.onnx`
> - `unet_mixed.weights`

---

## 🧰 Requirements

- **Windows** with Python **3.11 or 3.12** (PyTorch CPU wheels are available; 3.13 is **not** supported yet)
- Virtual environment (venv) that you **used for your notebook** (already has torch/onnxruntime/etc.)
- Internet access the first time (to download HF models like CLIP and InstaFlow weights)

### Python packages (typical)
If you need to recreate the environment:
```powershell
pip install fastapi "uvicorn[standard]" onnx onnxruntime Pillow numpy transformers
pip install --index-url https://download.pytorch.org/whl/cpu torch
````

---

## 🚀 Run (single origin: backend serves frontend)

> Assumes your venv is at `C:\Games\Instaflow\instaflow_cpu` (as shared) and you unzipped the project to `C:\Games\Instaflow\Instaflow-ui\`.

### PowerShell

```powershell
# 0) Activate your venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Games\Instaflow\instaflow_cpu\Scripts\Activate.ps1"

# 1) Start the server (from backend folder)
cd "C:\Games\Instaflow\Instaflow-ui\backend"
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Git Bash (no activation needed — call venv python directly)

```bash
cd "/c/Games/Instaflow/Instaflow-ui/backend"
/c/Games/Instaflow/instaflow_cpu/Scripts/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Now open: **[http://localhost:8000/](http://localhost:8000/)**

* Enter a prompt (e.g., *“a scenic mountain lake with pine trees, photorealistic, ultra-detailed”*)
* (Optional) enter a seed (number)
* Click **Generate Both**
  → You’ll see **Baseline** and **Quantized** images at **512×512**, with **latency** and **CLIP score** for each.

---

## 🧪 Health Check

* **Service:** [http://localhost:8000/healthz](http://localhost:8000/healthz)
  Returns JSON like:

  ```json
  {
    "ok": true,
    "torch_threads": 15,
    "onnx_model": "unet_mixed_final.onnx",
    "vae_scaling_factor": 0.18215,
    "in_channels": 4
  }
  ```

---

## 🔧 What Each File Does

### `backend/app.py`

* Loads **Baseline** model (Torch `RectifiedFlowPipeline`) and **Quantized** UNet (ONNXRuntime) **once** at startup.
* Provides helpers to:

  * encode prompt once,
  * prepare latents for 512×512,
  * run ORT UNet to get residual `v̂`,
  * **add `v̂` to latents** (matching notebook),
  * decode with the Torch VAE, and
  * apply a **small post-filter** (contrast/sharpness/autocontrast/brightness) **after decode** (exactly like the notebook).
* Computes **CLIP** score (`openai/clip-vit-base-patch32`) for each image.
* Serves endpoints:

  * `POST /compare` → always generates **both** baseline and quantized images and returns:

    ```json
    {
      "baseline": { "image_url": "/images/<uuid>.png", "latency_ms": ..., "clip_score": ..., "model": "baseline" },
      "quantized": { "image_url": "/images/<uuid>.png", "latency_ms": ..., "clip_score": ..., "model": "quantized" }
    }
    ```
  * `POST /generate` → generate with a single model (kept for completeness)
  * `GET /images/<file>` → returns the saved PNGs
  * `GET /` → sends the frontend `index.html`
  * `GET /static/...` → serves the frontend assets

> **Safety checker:** temporarily disabled in local mode to avoid black images for benign prompts.

### `backend/Rectified.py`

* Exposes `RectifiedFlowPipeline` used by both baseline generation and quantized path:

  * Baseline: uses the Torch UNet
  * Quantized: uses ONNXRuntime for the UNet **but** reuses the pipeline’s **tokenizer/text encoder** and **VAE** decode

### `backend/artifacts_onnx/`

* `unet_mixed_final.onnx` and `unet_mixed.weights` from your notebook’s **mixed-precision export**.
* The model MUST reference the external weights file with that exact name.

### `frontend/index.html`

* Minimal UI with **Prompt** and optional **Seed** input.
* Big **Generate Both** button.
* Two fixed **512×512** panes showing Baseline and Quantized results.
* Metrics (Latency & CLIP) and **Open/Download** links for each image.

### `frontend/styles.css`

* Clean dark theme, responsive layout.
* The two 512×512 cards are **original size**; on small screens they scroll horizontally.

### `frontend/app.js`

* Posts to `POST /compare` with `{ prompt, width: 512, height: 512, seed }`.
* Replaces the two `<img>` sources with returned `/images/<uuid>.png`.
* Shows metrics and enables Open/Download links.

---

## 🧩 How the Quantized Path Mirrors the Notebook

The quantized path in `app.py` is **line-for-line equivalent** to your IPYNB:

1. `enc = encode_prompt_once(prompt)`  *(Torch; CFG off)*
2. `latents = prepare_latents_once(generator, W, H)` *(Torch)*
3. ORT UNet forward: **`v_hat = ort_sess.run(...)[0]`**
4. Combine: **`lats2 = latents + v_hat`**
5. VAE Decode: **`img_t = vae.decode(lats2 / scaling)`**
6. Postprocess to PIL: `image_processor.postprocess(img_t)[0]`
7. Apply the small PIL filter (contrast↑, sharpness↑, autocontrast, brightness↓ slightly)

This ensures the quantized image **matches your reference notebook** (and fixes the earlier mismatch).

---

## 🐛 Troubleshooting

* **Nothing loads / red errors in terminal**
  Check the ONNX files exist:

  ```
  Instaflow-ui/backend/artifacts_onnx/unet_mixed_final.onnx
  Instaflow-ui/backend/artifacts_onnx/unet_mixed.weights
  ```

* **ImportError for `Rectified`**
  Run Uvicorn **from the backend folder**, so Python can find `Rectified.py`.

* **DLL / torch errors**
  Ensure your Python is **3.11 or 3.12** and install **CPU** torch wheels:

  ```
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  ```

* **First request is slow**
  Normal. The server does a small warm-up; the first real prompt is still the slowest.

* **Images 404**
  Confirm the backend is running and that the response’s `image_url` looks like `/images/<uuid>.png`.

---

## ⚙️ Useful Commands

### Start (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Games\Instaflow\instaflow_cpu\Scripts\Activate.ps1"
cd "C:\Games\Instaflow\Instaflow-ui\backend"
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Start (Git Bash)

```bash
cd "/c/Games/Instaflow/Instaflow-ui/backend"
/c/Games/Instaflow/instaflow_cpu/Scripts/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl http://localhost:8000/healthz
```

---

## 📝 Notes

* This is a **CPU** demo by design; if you have a CUDA machine you can port VAE/CLIP to GPU, but ONNXRuntime graph here targets CPU.
* The UI is fixed to **512×512** to match the evaluation setting and guarantee performance predictability.
* Images are stored (and grow) under `backend/Output_Images/`. Clean up periodically.

---

## 📣 Credits

* **Model:** `XCLiu/instaflow_0_9B_from_sd_1_5`
* **Quantization:** ONNXRuntime Quantization (static Conv INT8 + dynamic MatMul/Gemm)
* **Scoring:** `openai/clip-vit-base-patch32`
* **Team:** **Sota** — Akshat Kumar (22B4513), Anirudh Shinde (22B2181)

