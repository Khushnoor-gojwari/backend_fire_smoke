from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import uuid, cv2
import os
from pathlib import Path
import base64
from typing import Optional

router = APIRouter()

# Create temp directory
TEMP_DIR = Path("temp_videos")
TEMP_DIR.mkdir(exist_ok=True)

# Store processing tasks
processing_tasks = {}

def get_video_preview(video_path: Path):
    """Get first frame as preview"""
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    return None

def get_processed_preview(video_path: Path):
    """Get first frame from processed video as preview"""
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Resize for better display
            frame = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    return None

def process_video_task(task_id: str, video_path: Path):
    """Process video in background"""
    try:
        from app.utils.model_loader import get_model
        from app.utils.draw import draw_boxes
        
        model = get_model()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output path
        output_path = TEMP_DIR / f"{task_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        fire_detected = False
        smoke_detected = False
        detection_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store first processed frame with detections
        first_processed_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame)[0]
            
            detections = []
            for box in results.boxes:
                cls = int(box.cls[0])
                label = results.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })
                
                if label.lower() == "fire":
                    fire_detected = True
                elif label.lower() == "smoke":
                    smoke_detected = True
            
            # Count detections in this frame
            detection_count += len(detections)
            
            # Draw boxes
            if detections:
                frame = draw_boxes(frame, detections)
                # Save first frame with detections for preview
                if first_processed_frame is None:
                    first_processed_frame = frame.copy()
            
            out.write(frame)
            frame_count += 1
            
            # Update progress
            if total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                processing_tasks[task_id]['progress'] = progress
        
        cap.release()
        out.release()
        
        # Get processed preview
        processed_preview = None
        if first_processed_frame is not None:
            # Use the first frame with detections
            frame_resized = cv2.resize(first_processed_frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_preview = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        else:
            # Fallback to first frame of processed video
            processed_preview = get_processed_preview(output_path)
        
        # Update task
        processing_tasks[task_id].update({
            'status': 'completed',
            'output_path': str(output_path),
            'processed_preview': processed_preview,  # ADDED: Processed preview image
            'fire_detected': fire_detected,
            'smoke_detected': smoke_detected,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0,
            'detection_count': detection_count  # ADDED: Total detections count
        })
        
    except Exception as e:
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['error'] = str(e)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video and return preview"""
    # Validate file
    if not file.content_type or 'video' not in file.content_type:
        raise HTTPException(400, "Please upload a video file")
    
    # Save file
    task_id = uuid.uuid4().hex[:8]
    video_path = TEMP_DIR / f"{task_id}_original.mp4"
    
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)
    
    # Get preview
    preview = get_video_preview(video_path)
    
    # Store task
    processing_tasks[task_id] = {
        'task_id': task_id,
        'status': 'uploaded',
        'original_path': str(video_path),
        'filename': file.filename,
        'preview': preview,
        'progress': 0
    }
    
    return {
        "task_id": task_id,
        "preview": preview,
        "filename": file.filename,
        "message": "Video uploaded successfully"
    }

@router.post("/process/{task_id}")
async def start_processing(task_id: str, background_tasks: BackgroundTasks):
    """Start processing video"""
    if task_id not in processing_tasks:
        raise HTTPException(404, "Video not found")
    
    task = processing_tasks[task_id]
    if task['status'] == 'processing':
        raise HTTPException(400, "Already processing")
    
    # Start processing
    task['status'] = 'processing'
    video_path = Path(task['original_path'])
    background_tasks.add_task(process_video_task, task_id, video_path)
    
    return {"message": "Processing started", "task_id": task_id}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get processing status"""
    if task_id not in processing_tasks:
        raise HTTPException(404, "Task not found")
    
    task = processing_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task['status'],
        "progress": task.get('progress', 0),
        "preview": task.get('preview'),
        "filename": task.get('filename', '')
    }
    
    if task['status'] == 'completed':
        response.update({
            "output_path": task['output_path'],
            "stream_url": f"http://localhost:8000/api/video/stream/{task_id}",
            "processed_preview": task.get('processed_preview'),  # ADDED: Processed preview
            "fire_detected": task.get('fire_detected', False),
            "smoke_detected": task.get('smoke_detected', False),
            "download_url": f"http://localhost:8000/api/video/download/{task_id}",
            "frame_count": task.get('frame_count', 0),
            "duration": task.get('duration', 0),
            "detection_count": task.get('detection_count', 0)  # ADDED: Detection count
        })
    elif task['status'] == 'error':
        response['error'] = task.get('error')
    
    return response

@router.get("/stream/{task_id}")
async def stream_video(task_id: str):
    """Stream video for playback (no download)"""
    if task_id not in processing_tasks:
        raise HTTPException(404, "Task not found")
    
    task = processing_tasks[task_id]
    if task['status'] != 'completed':
        raise HTTPException(400, "Video not ready for streaming")
    
    output_path = Path(task['output_path'])
    if not output_path.exists():
        raise HTTPException(404, "Video file not found")
    
    # Simple streaming response for playback
    def iterfile():
        with open(output_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{task["filename"]}"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.get("/download/{task_id}")
async def download_video(task_id: str):
    """Download processed video (forces download)"""
    if task_id not in processing_tasks:
        raise HTTPException(404, "Task not found")
    
    task = processing_tasks[task_id]
    if task['status'] != 'completed':
        raise HTTPException(400, "Processing not complete")
    
    output_path = Path(task['output_path'])
    if not output_path.exists():
        raise HTTPException(404, "Result not found")
    
    return FileResponse(
        path=output_path,
        filename=f"processed_{task['filename']}",
        media_type='video/mp4',
        headers={
            "Content-Disposition": f'attachment; filename="processed_{task["filename"]}"'
        }
    )