# app/api/realtime.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import base64
import cv2
import numpy as np
from typing import List, Dict, Optional
import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from collections import deque

router = APIRouter()

# Load YOLO model
from app.utils.model_loader import get_model
model = get_model()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.frame_queues = {}  # Store frame queues per client
        self.processing_flags = {}  # Track if client is being processed
        self.last_processed = {}  # Track last processed timestamp per client
        self.frame_counter = {}  # Frame counter for skipping
        self.executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for processing

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        websocket.client_id = client_id
        
        self.active_connections.append(websocket)
        self.frame_queues[client_id] = deque(maxlen=2)  # Only keep latest 2 frames
        self.processing_flags[client_id] = False
        self.last_processed[client_id] = 0
        self.frame_counter[client_id] = 0
        
        print(f"New client connected: {client_id}. Total: {len(self.active_connections)}")
        return client_id

    def disconnect(self, websocket: WebSocket):
        if hasattr(websocket, 'client_id'):
            client_id = websocket.client_id
            if client_id in self.frame_queues:
                del self.frame_queues[client_id]
            if client_id in self.processing_flags:
                del self.processing_flags[client_id]
            if client_id in self.last_processed:
                del self.last_processed[client_id]
            if client_id in self.frame_counter:
                del self.frame_counter[client_id]
        
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")

manager = ConnectionManager()

def process_frame_async(frame, client_id, timestamp):
    """Process frame in a separate thread"""
    try:
        # Run YOLO inference with confidence threshold
        results = model(frame, conf=0.3, verbose=False)[0]
        
        detections = []
        fire_detected = False
        smoke_detected = False
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls_id = classes[i]
                confidence = float(confidences[i])
                label = results.names[cls_id]
                
                # Store detection info
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "confidence": confidence,
                    "timestamp": timestamp
                })
                
                # Check for fire/smoke
                label_lower = label.lower()
                if "fire" in label_lower:
                    fire_detected = True
                if "smoke" in label_lower:
                    smoke_detected = True
        
        return {
            "detections": detections,
            "fire_detected": fire_detected,
            "smoke_detected": smoke_detected,
            "timestamp": timestamp,
            "client_id": client_id
        }
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    if not detections:
        return frame
    
    annotated_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        confidence = det["confidence"]
        
        # Colors for different classes
        if "fire" in label.lower():
            color = (0, 0, 255)  # Red for fire
            thickness = 3
        elif "smoke" in label.lower():
            color = (0, 165, 255)  # Orange for smoke
            thickness = 2
        else:
            color = (0, 255, 0)  # Green for others
            thickness = 1
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_text = f"{label} {confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
    
    return annotated_frame

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    # Encode frame as JPEG with low quality for speed
    success, buffer = cv2.imencode('.jpg', frame, [
        cv2.IMWRITE_JPEG_QUALITY, 60  # Lower quality for faster transfer
    ])
    if not success:
        return None
    
    # Convert to base64
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

@router.websocket("/stream")
async def ws_stream(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    
    try:
        while True:
            try:
                # Receive frame from client with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                print(f"Client {client_id} timeout")
                break
            except WebSocketDisconnect:
                break
                
            try:
                payload = json.loads(data)
                frame_data = payload.get("frame", "")
                timestamp = payload.get("timestamp", time.time())
                
                if not frame_data:
                    continue
                
                # Extract base64 data
                if "," in frame_data:
                    frame_data = frame_data.split(",")[-1]
                
                # Decode base64 to image
                try:
                    image_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                        
                except Exception as e:
                    print(f"Image decode error: {e}")
                    continue
                
                # Store frame in queue (only keep latest)
                manager.frame_queues[client_id].append({
                    "frame": frame,
                    "timestamp": timestamp
                })
                
                current_time = time.time()
                
                # Apply frame skipping: Process only 2 frames per second
                manager.frame_counter[client_id] += 1
                
                # Skip frames to achieve ~2 FPS processing
                skip_frames = 15  # Assuming client sends 30 FPS, skip 15 to get ~2 FPS
                if manager.frame_counter[client_id] % skip_frames != 0:
                    # Send quick acknowledgement without processing
                    await manager.send_personal_message({
                        "status": "frame_received",
                        "timestamp": timestamp
                    }, websocket)
                    continue
                
                # Check if last processing was too recent (minimum 0.5 seconds between processing)
                time_since_last = current_time - manager.last_processed.get(client_id, 0)
                if time_since_last < 0.5:  # 500ms minimum between processing
                    continue
                
                # Check if already processing
                if manager.processing_flags[client_id]:
                    continue
                
                # Get latest frame from queue
                if not manager.frame_queues[client_id]:
                    continue
                
                latest_data = manager.frame_queues[client_id][-1]
                process_frame = latest_data["frame"]
                process_timestamp = latest_data["timestamp"]
                
                # Mark as processing
                manager.processing_flags[client_id] = True
                manager.last_processed[client_id] = current_time
                
                # Process frame in thread pool
                future = manager.executor.submit(
                    process_frame_async,
                    process_frame,
                    client_id,
                    process_timestamp
                )
                
                # Get result (blocking, but in thread)
                result = future.result()
                
                if result:
                    # Create alert message
                    alert = ""
                    if result["fire_detected"] and result["smoke_detected"]:
                        alert = "🚨 FIRE & SMOKE DETECTED! 🚨"
                    elif result["fire_detected"]:
                        alert = "🔥 FIRE DETECTED! 🔥"
                    elif result["smoke_detected"]:
                        alert = "💨 SMOKE DETECTED! 💨"
                    
                    # Draw detections on original frame
                    annotated_frame = None
                    if len(result["detections"]) > 0:
                        annotated_frame = draw_detections(process_frame, result["detections"])
                    
                    # Prepare response
                    response = {
                        "detections": result["detections"],
                        "fire_detected": result["fire_detected"],
                        "smoke_detected": result["smoke_detected"],
                        "alert": alert,
                        "detection_count": len(result["detections"]),
                        "timestamp": result["timestamp"],
                        "processing_time": time.time() - current_time
                    }
                    
                    # Add voice alert if needed
                    if result["fire_detected"] or result["smoke_detected"]:
                        if result["fire_detected"] and result["smoke_detected"]:
                            response["voice_alert"] = "Warning! Fire and smoke detected!"
                        elif result["fire_detected"]:
                            response["voice_alert"] = "Warning! Fire detected!"
                        elif result["smoke_detected"]:
                            response["voice_alert"] = "Warning! Smoke detected!"
                    
                    # Add annotated frame if available
                    if annotated_frame is not None:
                        response["annotated_frame"] = frame_to_base64(annotated_frame)
                    else:
                        # Send original frame with detections data
                        response["frame"] = frame_to_base64(process_frame)
                    
                    # Send response to client
                    await manager.send_personal_message(response, websocket)
                
                # Reset processing flag
                manager.processing_flags[client_id] = False
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                if client_id in manager.processing_flags:
                    manager.processing_flags[client_id] = False
                
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected normally")
    except Exception as e:
        print(f"WebSocket error for {client_id}: {e}")
    finally: 
        manager.disconnect(websocket)

# Cleanup on shutdown
@router.on_event("shutdown")
def shutdown_event():
    manager.executor.shutdown(wait=False)
    print("Thread pool shutdown")