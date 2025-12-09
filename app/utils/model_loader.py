from ultralytics import YOLO
import os

# model_loader.py is inside app/utils/
# We go one directory up: app/
# Another directory up: FireSmokeBackend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

print("Loading YOLO model from:", MODEL_PATH)
print("File exists:", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
model.model.names = {0: "Smoke", 1: "Fire"}


def get_model():
    return model
