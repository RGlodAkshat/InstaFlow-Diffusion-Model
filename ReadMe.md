# InstaFlow – Baseline vs Quantized (CPU)
**IE 643 Course Project — “Quantization of InstaFlow Model”**  
Team: **Sota** • Members: **Akshat Kumar (22B4513), Anirudh Shinde (22B2181)**

This repository contains:

1) A **research notebook pipeline** that exports InstaFlow’s UNet to ONNX and performs **mixed-precision quantization** (static INT8 for Conv, dynamic for MatMul/Gemm), then evaluates image quality & CLIP alignment.

2) A **single-origin web app** (FastAPI) that **serves the frontend** and compares **Baseline (Torch)** vs **Quantized (ONNXRuntime)** outputs **side-by-side at 512×512**, with latency and CLIP scores.

---

## 🗂 Folder Structure (top-level)

```

.
├─ artifacts_onnx/                 # ONNX models exported from the notebook
│  ├─ unet_mixed_final.onnx
│  └─ unet_mixed.weights
├─ instaflow_cpu/                  # Python venv (Windows) with torch/onnxruntime/etc.
├─ Instaflow-ui/                   # Web app (backend + frontend)
│  ├─ backend/
│  │  ├─ app.py                    # FastAPI server + serves frontend + inference
│  │  ├─ Rectified.py              # Provides RectifiedFlowPipeline
│  │  ├─ artifacts_onnx/ -> ../../artifacts_onnx (copy or keep in sync)
│  │  └─ Output_Images/            # Generated images (auto)
│  └─ frontend/
│     ├─ index.html                # 512×512 comparison-first UI
│     ├─ styles.css
│     └─ app.js
├─ Output_Images/                  # Notebook-generated images (if running notebook)
├─ quant_scout_results/            # (optional) exploration/metrics from experiments
├─ 8bit_quantized_clip_score.csv   # CLIP results snapshot (optional artifact)
├─ Mixed_quantized_clip_score.csv  # CLIP results snapshot (optional artifact)
├─ quantized_clip_score.csv        # CLIP results snapshot (optional artifact)
├─ baseline_clip_score.csv         # CLIP results snapshot (optional artifact)
├─ InstaFlow.ipynb                 # Main research notebook (export + quantize + eval)
├─ Rectified.py                    # Pipeline module (duplicated here for notebook runs)
└─ .gitignore

````

> **Important:** The web app expects the ONNX artifacts under `Instaflow-ui/backend/artifacts_onnx/`.  
> You can either **copy** `artifacts_onnx/*` there or **symlink** the folder.

---

## 🔬 Part A — Research Notebook (InstaFlow.ipynb)

### What it does (high-level)
- Loads `RectifiedFlowPipeline` on CPU.
- Exports the **UNet** to **ONNX** (packed weights).
- Builds a tiny calibration dataset (10 frames) from two prompts.
- **Quantizes**:
  - **Static INT8** for `Conv` ops (QDQ format; per-channel weights; `Percentile` → fallback `MinMax`).
  - **Dynamic** for `MatMul/Gemm` (weight-only INT8).
- Saves `unet_mixed_final.onnx` with a single **external weights file** `unet_mixed.weights`.
- **Inference (quantized path)** mirrors the intended deployment:
  1. Encode prompt → `encoder_hidden_states`
  2. Prepare latents at 512×512
  3. ONNXRuntime UNet → residual **v̂**
  4. Add residual: **`latents + v̂`**
  5. VAE decode: **`vae.decode(lats2 / scaling_factor)`**
  6. `(img/2 + 0.5).clamp(0,1)` then **postprocess** to PIL
  7. Apply small **post-filter** (contrast ↑, sharpness ↑, autocontrast, brightness slight ↓)
- Computes **CLIP score** for prompt–image alignment
- Writes sample images + optional CSV metrics

### Running the notebook (Windows)
> Use the same venv you ran before (Python **3.11/3.12**; not 3.13).

1. **Activate venv**
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   & "C:\Games\Instaflow\instaflow_cpu\Scripts\Activate.ps1"
   ````

2. **Launch Jupyter**

   ```powershell
   jupyter lab
   ```

   Open `InstaFlow.ipynb`.

3. **Run cells sequentially**:

   * Export → Quantize → Save `artifacts_onnx/unet_mixed_final.onnx` + `unet_mixed.weights`
   * Inference: verify `img_q` (raw) and `img_q_fixed` (filtered)
   * CLIP: verify reported `logits_per_image` or your normalized score

4. **Copy artifacts for the web app** (if needed)

   ```
   copy .\artifacts_onnx\unet_mixed_final.onnx .\Instaflow-ui\backend\artifacts_onnx\
   copy .\artifacts_onnx\unet_mixed.weights   .\Instaflow-ui\backend\artifacts_onnx\
   ```

---

## 🌐 Part B — Web App (FastAPI + Frontend)

### What it does

* **Baseline** path uses Torch UNet via `RectifiedFlowPipeline`.
* **Quantized** path uses **ONNXRuntime** UNet (+ Torch VAE decode), exactly matching the notebook:

  * ONNXRuntime returns **v̂**, we do **`latents + v̂`**, decode, then **apply the same post-filter**.
* Computes **CLIP** score using `openai/clip-vit-base-patch32`.
* **Serves**:

  * `GET /` → Frontend (`index.html`)
  * `GET /static/*` → Frontend assets (`styles.css`, `app.js`)
  * `GET /images/<uuid>.png` → Generated images
  * `POST /compare` → Generates **both** 512×512 images (Baseline + Quantized), returns URLs and metrics
  * `POST /generate` → Single model (kept for completeness)
  * `GET /healthz` → Health info (threads, onnx filename, VAE scaling, etc.)

### Start the server (single origin)

Use your working venv. Examples:

**PowerShell**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Games\Instaflow\instaflow_cpu\Scripts\Activate.ps1"

cd "C:\Games\Instaflow\Instaflow-ui\backend"
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Git Bash**

```bash
cd "/c/Games/Instaflow/Instaflow-ui/backend"
/c/Games/Instaflow/instaflow_cpu/Scripts/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000/](http://localhost:8000/)**

* Type your prompt, optional seed, **Generate Both**.
* You’ll see **two 512×512** panes with **Latency** and **CLIP** for Baseline & Quantized.
* Images also saved to `Instaflow-ui/backend/Output_Images/`.

> **Note:** The service disables the diffusers safety checker for local testing to avoid false black frames for benign prompts.

---

## 🧩 How the Quantized Inference Matches the Notebook

Inside `backend/app.py → GlobalState.run_quantized()`:

```python
enc     = encode_prompt_once(prompt)
latents = prepare_latents_once(gen, W, H)
tvec    = torch.ones((1,), device=cpu, dtype=torch.float32) * 1000.0

v_hat   = ort_sess.run(None, {"sample":latents.numpy(), "timestep":tvec.numpy(), "encoder_hidden_states":enc.numpy()})[0]
lats2   = latents + torch.from_numpy(v_hat)

img_t   = vae.decode(lats2 / vae_scaling_factor)[0]
img_t   = (img_t / 2 + 0.5).clamp(0, 1).detach()
img_pil = image_processor.postprocess(img_t, output_type="pil")[0]

# small tone/sharpness correction (same as notebook)
img_fix = brighten(sharpen(autocontrast(contrast(img_pil))))
```

This is the same sequence you validated in the IPYNB.

---

## 🧰 Dependencies (typical)

If you need to recreate the env:

```powershell
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" onnx onnxruntime Pillow numpy transformers
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

* Python **3.11 or 3.12** recommended (PyTorch CPU wheels available)
* First run will download model weights (Hugging Face)

---

## 🧪 Health & Testing

* **Health**: `http://localhost:8000/healthz`
* **Images folder**: `Instaflow-ui/backend/Output_Images/` (auto-created)

If you want a quick programmatic smoke test (optional), you can hit `/compare` with curl:

```bash
curl -X POST "http://localhost:8000/compare" ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"a scenic mountain lake with pine trees, photorealistic\",\"width\":512,\"height\":512}"
```

---

## 🐛 Troubleshooting

* **ImportError: Rectified**
  Run Uvicorn **from the `Instaflow-ui/backend` folder** so Python can see `Rectified.py`.

* **ONNX artifacts missing**
  Ensure:

  ```
  Instaflow-ui/backend/artifacts_onnx/unet_mixed_final.onnx
  Instaflow-ui/backend/artifacts_onnx/unet_mixed.weights
  ```

  Filenames must match exactly.

* **Torch DLL error / Python 3.13**
  Use Python **3.11 or 3.12** and install the **CPU torch wheel** from the PyTorch index shown above.

* **First request slow**
  Normal on CPU. The server does a warm-up, but the first actual prompt incurs additional load times.

* **Images not showing in UI**
  Check DevTools → Network: verify `<img>` requests go to `/images/<uuid>.png` and return HTTP 200.
  Hard refresh (**Ctrl+Shift+R**) to clear cached JS.

---

## 📈 Optional CSVs

* `baseline_clip_score.csv`, `8bit_quantized_clip_score.csv`, `Mixed_quantized_clip_score.csv`, `quantized_clip_score.csv`
  Quick logs captured during experiments—useful for plotting CLIP distributions or deltas.

---

## 📣 Credits

* **Model**: `XCLiu/instaflow_0_9B_from_sd_1_5`
* **Quantization**: ONNXRuntime (QDQ format; static INT8 Conv + dynamic MatMul)
* **Scoring**: `openai/clip-vit-base-patch32`
* **Team**: **Sota** — *Akshat Kumar (22B4513), Anirudh Shinde (22B2181)*

---

## 🔐 License / Use

Academic project usage. Check model licenses for downstream redistribution limits.

