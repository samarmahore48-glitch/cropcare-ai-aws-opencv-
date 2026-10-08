"""
CropCare AI - Production FastAPI Backend
Orchestrates OpenCV 5 Core Vision Pipeline, AWS Cloud Architecture (EC2 Graviton, S3, DynamoDB),
and the Agentic Decision & Risk Prediction Layer.
Author: CropCare AI Core Engineering Team
"""

import os
import time
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Local core modules
from cropcare_ai.backend.vision_engine import OpenCV5VisionEngine
from cropcare_ai.backend.aws_services import AWSCloudManager
from cropcare_ai.backend.decision_agent import AgenticDecisionEngine

logger = logging.getLogger("CropCareApp")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="CropCare AI - Crop Intelligence & Vision Engine API",
    description="Production-grade AI pipeline with OpenCV 5, AWS Graviton, S3, DynamoDB, and Bedrock.",
    version="2.0.0"
)

# Enable CORS for all frontend origins (Vercel, Netlify, localhost, S3 Static Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines with defensive error guards
vision_engine = OpenCV5VisionEngine()
aws_manager = AWSCloudManager()
decision_agent = AgenticDecisionEngine(aws_manager=aws_manager)

@app.get("/api/v1/health")
async def health_check():
    """Liveness probe for AWS Application Load Balancer / ECS / Kubernetes."""
    return {
        "status": "healthy",
        "service": "CropCare AI Core Vision & Cloud Service",
        "opencv_version": vision_engine.opencv_version,
        "instance_arch": aws_manager.instance_type,
        "timestamp": time.time()
    }

@app.get("/api/v1/aws-status")
async def get_aws_status():
    """Returns AWS Cloud Architecture telemetry (AWS Sponsored Track)."""
    return aws_manager.get_cloud_status()

@app.get("/api/v1/farm-logs")
async def get_farm_logs(farm_id: Optional[str] = None, limit: int = 20):
    """Fetches historical scan logs from Amazon DynamoDB."""
    logs = aws_manager.query_farm_scans(farm_id=farm_id, limit=limit)
    return {
        "status": "success",
        "count": len(logs),
        "table": aws_manager.dynamodb_table_name,
        "records": logs
    }

@app.post("/api/v1/agent-decision")
async def compute_agent_decision(
    disease_name: str = Form("Early Blight"),
    severity_pct: float = Form(42.0),
    crop_name: str = Form("Tomato"),
    growth_stage: str = Form("Flowering"),
    humidity_pct: float = Form(86.0),
    temperature_c: float = Form(29.0),
    rainfall_prob_pct: float = Form(65.0),
    wind_speed_kmh: float = Form(8.5)
):
    """Evaluates the Perception-Decision loop on given parameters."""
    decision = decision_agent.evaluate_perception_decision_loop(
        disease_name=disease_name,
        severity_pct=severity_pct,
        crop_name=crop_name,
        growth_stage=growth_stage,
        humidity_pct=humidity_pct,
        temperature_c=temperature_c,
        rainfall_prob_pct=rainfall_prob_pct,
        wind_speed_kmh=wind_speed_kmh
    )
    return {"status": "success", "decision": decision}

@app.post("/api/v1/scan")
async def scan_crop_leaf(
    file: UploadFile = File(...),
    crop: str = Form("Tomato"),
    farm_id: str = Form("FARM-INDORE-01"),
    growth_stage: str = Form("Flowering"),
    humidity: float = Form(86.0),
    temperature: float = Form(29.0),
    rainfall_prob: float = Form(65.0)
):
    """
    Primary Production Scan Endpoint:
    1. Receives leaf image byte buffer
    2. Runs OpenCV 5 Vision Pipeline (CLAHE, Gaussian Blur, HSV Segmentation, Contours, Severity %)
    3. Runs Agentic Decision Loop (Correlates Vision + Weather + Stage)
    4. Persists high-res image to Amazon S3
    5. Records audit and metrics log into Amazon DynamoDB
    """
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # 1. Execute OpenCV 5 Core Vision Pipeline
        start_time = time.time()
        vision_result = vision_engine.process_image_bytes(file_bytes, crop_hint=crop)
        processing_time_ms = round((time.time() - start_time) * 1000, 1)

        # 2. Execute Perception-Decision Loop (Correlate with Weather)
        decision = decision_agent.evaluate_perception_decision_loop(
            disease_name=vision_result["disease_name"],
            severity_pct=vision_result["severity_percentage"],
            crop_name=crop,
            growth_stage=growth_stage,
            humidity_pct=humidity,
            temperature_c=temperature,
            rainfall_prob_pct=rainfall_prob
        )

        # 3. Upload raw leaf image to AWS S3 Bucket
        s3_raw = aws_manager.upload_to_s3(file_bytes, file.filename or "leaf_scan.jpg", folder="raw")

        # 4. Log to Amazon DynamoDB
        scan_record = {
            "farm_id": farm_id,
            "scan_id": f"SCAN#{int(time.time())}_{file.filename[:8] if file.filename else 'leaf'}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "crop_type": crop,
            "growth_stage": growth_stage,
            "disease_detected": vision_result["disease_name"],
            "pathogen": vision_result["pathogen"],
            "severity_percentage": vision_result["severity_percentage"],
            "lesion_count": vision_result["lesion_count"],
            "total_leaf_area_px": vision_result["total_leaf_area_px"],
            "total_lesion_area_px": vision_result["total_lesion_area_px"],
            "weather_humidity": humidity,
            "weather_temp": temperature,
            "risk_score_7day": decision["risk_score_7day"],
            "urgency": decision["urgency"],
            "spray_window_hours": decision["spray_window_hours"],
            "action_recommendation": decision["prescription"]["action_summary_en"],
            "s3_image_uri": s3_raw["s3_uri"],
            "processed_by_instance": aws_manager.instance_type,
            "latency_ms": processing_time_ms
        }
        dynamo_res = aws_manager.log_scan_dynamodb(scan_record)

        return JSONResponse(content={
            "status": "success",
            "opencv_version": vision_engine.opencv_version,
            "latency_ms": processing_time_ms,
            "farm_id": farm_id,
            "crop": crop,
            "growth_stage": growth_stage,
            # Vision metrics
            "vision": {
                "disease": vision_result["disease_name"],
                "pathogen": vision_result["pathogen"],
                "secondary_pest": vision_result["secondary_pest"],
                "confidence": vision_result["confidence_score"],
                "severity_percentage": vision_result["severity_percentage"],
                "lesion_count": vision_result["lesion_count"],
                "total_leaf_area_px": vision_result["total_leaf_area_px"],
                "total_lesion_area_px": vision_result["total_lesion_area_px"],
                "bounding_boxes": vision_result["bounding_boxes"],
                "inference_engine": vision_result["inference_engine"],
            },
            # 4 Pipeline Stage Images (Base64 JPEG for instant rendering)
            "pipeline_stages": {
                "stage1_original": vision_result["stage_original_b64"],
                "stage2_clahe": vision_result["stage_clahe_b64"],
                "stage3_hsv_mask": vision_result["stage_hsv_mask_b64"],
                "stage4_contours": vision_result["stage_contour_overlay_b64"]
            },
            # Agentic Decision & Weather Correlation
            "decision": decision,
            # AWS Cloud Architecture Receipts
            "aws": {
                "s3_uri": s3_raw["s3_uri"],
                "dynamodb_record": dynamo_res,
                "instance": aws_manager.instance_type
            }
        })

    except Exception as e:
        logger.error(f"Error during scan processing: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Processing error: {str(e)}"}
        )

# Optional static frontend serving if hosted as a monolith
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
if os.path.exists(frontend_path):
    from fastapi.responses import FileResponse
    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CropCare AI on port {port} with OpenCV {vision_engine.opencv_version}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
