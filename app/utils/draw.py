import cv2
def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']

        # Different colors for fire and smoke
        if label.lower() == "fire":
            color = (0, 0, 255)  # Red for fire
            thickness = 4  # Increased from 2 to 4
        else:
            color = (0, 255, 0)  # Green for smoke
            thickness = 3  # Increased from 2 to 3

        # Draw thicker rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased from 0.6
        font_thickness = 2  # Increased font thickness
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            f"{label} {conf:.2f}", font, font_scale, font_thickness
        )
        
        # Draw filled rectangle for text background
        cv2.rectangle(image, 
                     (x1, y1 - text_height - 10),
                     (x1 + text_width, y1),
                     color, -1)  # -1 means filled rectangle
        
        # Draw text on top of background
        cv2.putText(image, f"{label} {conf:.2f}", 
                   (x1, y1 - 5),  # Position adjusted
                   font, font_scale, 
                   (255, 255, 255),  # White text
                   font_thickness)

    return image
