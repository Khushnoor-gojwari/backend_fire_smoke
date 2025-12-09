from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io, cv2
import numpy as np
import base64
from app.utils.model_loader import get_model
from app.utils.draw import draw_boxes

router = APIRouter()

def run_inference(img_bytes):
    """
    Run YOLO inference on image bytes
    """
    model = get_model()
    model.model.names = {0: "Smoke", 1: "Fire"}

    # Decode bytes to image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(400, "Invalid image format")

    # Run YOLO inference
    results = model(
        img,
        conf=0.17,
        iou=0.45,
        verbose=False
    )[0]

    detections = []
    fire = smoke = False

    for box in results.boxes:
        cls = int(box.cls.cpu())
        conf = float(box.conf.cpu())
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        label = model.model.names[cls]

        detections.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })

        if label == "Fire":
            fire = True
        if label == "Smoke":
            smoke = True

    return detections, fire, smoke, img

@router.post("/upload/combined")
async def predict_image_combined(file: UploadFile = File(...)):
    """
    Single endpoint that returns both original and annotated images with detections
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Read image
        content = await file.read()
        
        # Convert original image to base64
        original_base64 = base64.b64encode(content).decode('utf-8')
        
        # Run inference
        detections, fire, smoke, img = run_inference(content)
        
        # Create annotated image with bounding boxes
        annotated_img = draw_boxes(img.copy(), detections)
        _, encoded = cv2.imencode('.jpg', annotated_img)
        annotated_bytes = encoded.tobytes()
        annotated_base64 = base64.b64encode(annotated_bytes).decode('utf-8')


        # Fix filename - remove extension if it already has one
        filename = file.filename
        # Remove existing extension
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        # Add proper extension
        filename = filename + '.jpg'
        
        return {
            "original_image": f"data:image/jpeg;base64,{original_base64}",
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections": detections,
            "fire_detected": fire,
            "smoke_detected": smoke,
            "filename": file.filename,
            "detection_count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

# Optional: Separate endpoints if needed


@router.post("/upload/json")
async def predict_image_json(file: UploadFile = File(...)):
    """
    Returns only detection data (for backward compatibility)
    """
    try:
        content = await file.read()
        detections, fire, smoke, img = run_inference(content)
        return {
            "detections": detections,
            "fire_detected": fire,
            "smoke_detected": smoke,
            "detection_count": len(detections)
        }
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")