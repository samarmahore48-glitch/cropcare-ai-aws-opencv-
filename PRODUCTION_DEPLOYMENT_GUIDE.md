# CropCare AI — Production Architecture & Zero-Crash Deployment Guide

> **Production Guarantee ("Website Par Phate Na"):** This system is built with an **Auto-Adaptive Dual Engine**. If deployed on any static web host (Vercel, Netlify, GitHub Pages, Render, AWS S3 Static Web) without a live Python backend, it automatically executes its client-side OpenCV 5 simulation pipeline using HTML5 Canvas SIMD. If connected to AWS EC2 Graviton, it seamlessly orchestrates the full cloud pipeline. It **never crashes, never throws unhandled errors, and never stays stuck on a blank screen.**

---

## 1. Core Vision & AI Engine (OpenCV 5.0)

### Preprocessing Pipeline:
1. **Luminance Equalization (CLAHE):** Converts input image to LAB color space and applies `cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))` on the L-channel to normalize harsh Indian field sunlight and shadows.
2. **Noise Suppression:** Applies `cv2.GaussianBlur(..., (5, 5), 0)` to eliminate CMOS sensor noise.
3. **HSV Color Space Segmentation:** Converts image to HSV to separate:
   - **Chlorophyll Green (Healthy Foliage):** Hue: `[25, 88]`, Saturation: `[35, 255]`, Value: `[35, 255]`.
   - **Necrotic Lesions & Blights:** Hue ranges for chlorotic yellow halos `[12, 24]`, necrotic brown spots `[0, 11]`, and dark sporulation `[95, 135]`.
4. **Severity Calculation Algorithm:**
   $$\text{Severity \%} = \left(\frac{\text{Infected Area } (A_{\text{inf}})}{\text{Total Leaf Area } (A_{\text{leaf}})}\right) \times 100$$
   - Uses `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` and `cv2.contourArea`.
   - Extracts bounding boxes (`cv2.boundingRect`) and generates HUD overlays.
5. **OpenCV 5 DNN Module Inference:**
   - Uses `cv2.dnn.readNetFromONNX` and `cv2.dnn.blobFromImage` for fast YOLOv8 / MobileNetV3 inference on CPU/Graviton.
   - Built-in analytical heuristic fallback if ONNX weights are not mounted.

---

## 2. AWS Cloud Architecture (AWS Sponsored Track)

- **AWS EC2 (Graviton Instances):**
  - Architecture: ARM64 (Neoverse V1/V2 cores).
  - Acceleration: ARM NEON SIMD vectorization enabled in OpenCV 5.
  - Price-Performance: Up to 40% cost reduction compared to legacy x86 instances.
- **Amazon S3 Bucket (`cropcare-farmer-leaf-uploads-prod`):**
  - Stores high-resolution leaf images with pre-signed URLs (15-min TTL) and S3 Intelligent-Tiering.
- **Amazon DynamoDB (`CropCare_FarmScans`):**
  - Partition Key: `farm_id` (String)
  - Sort Key: `scan_id` (String: `SCAN#timestamp`)
  - Attributes: `crop_type`, `disease_detected`, `severity_percentage`, `lesion_count`, `risk_score_7day`, `spray_window_hours`, `s3_image_uri`, `processed_by_instance`.
- **AWS Bedrock / Lambda (Agentic Loop):**
  - Invokes Anthropic Claude 3 Haiku or Amazon Titan via `bedrock-runtime` for contextual agronomic advice.
  - Automatic fallback to deterministic CIBRC / ICAR agronomic rule engine when running offline.

---

## 3. Agentic Decision & Risk Prediction Layer

### Perception-Decision Loop:
- Correlates vision outputs (Disease + Severity %) with live weather data and crop growth stage:
  - **Humidity Factor:** Relative Humidity $> 80\% \to 1.35\times$ risk multiplier.
  - **Rainfall Factor:** Rain probability $> 60\% \to +15$ risk points (wash-off & splash danger).
  - **Crop Phenology Stage:** Flowering/Fruiting $\to 1.3\times$ vulnerability; Vegetative $\to 1.0\times$.
- Generates near-term 7-day risk curve and actionable recommendations:
  - *"Fungicide spray required within 24–48 hours."*
  - Specific approved chemicals: Copper Oxychloride 50 WP @ 2.5 g/L, Mancozeb, Metalaxyl, or bio-alternatives (Trichoderma, Neem oil).
  - Bilingual voice/text outputs (English + Hindi).

---

## 4. How to Run & Publish Without Crashing

### Option A: Static Web Hosting (Vercel, Netlify, GitHub Pages)
1. Simply upload `cropcare_ai/index.html` as the root `index.html`.
2. The application will boot immediately into **Edge Vision Mode**.
3. All camera scans, sample tests, CLAHE processing, and HSV contour analysis execute in the browser's HTML5 Canvas without needing a running server.

### Option B: Local Full-Stack Launch
```bash
# 1. Start Python FastAPI backend (Port 8000)
cd cropcare_ai/backend
python main.py

# 2. Open frontend
Open cropcare_ai/index.html in any browser.
The app will auto-detect: "🟢 AWS Graviton (API) Connected".
```

### Option C: AWS EC2 Graviton Deployment
```bash
# On your AWS Graviton EC2 instance (c7g.xlarge / t4g.xlarge):
git clone <your-repo>
cd cropcare_ai/backend
chmod +x deploy_aws_ec2.sh
./deploy_aws_ec2.sh
```
