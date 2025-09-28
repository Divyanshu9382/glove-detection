# detection_script.py

import os
import json
import argparse
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def detect_hands(input_folder, output_folder, model_path, confidence_threshold):
    """
    Detects objects in images using a custom YOLOv8 model,
    saves annotated images, and logs detections to JSON files.
    """
    #1. Load Custom Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get class names from the model
    class_names = model.names
    print(f"Model loaded. Detecting classes: {list(class_names.values())}")

    # Define colors for specific classes
    colors = {
        'glove': (0, 255, 0),  # Green
        'no-glove': (0, 0, 255)  # Red
    }

    #2. Create Output Directories
    annotated_img_folder = os.path.join(output_folder, 'images')
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(annotated_img_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    #3. Process Images
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png','.webp'))]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images. Starting detection...")

    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_folder, filename)

        results = model(image_path, verbose=False)
        result = results[0]

        img = result.orig_img.copy()

        detection_log = {
            "filename": filename,
            "detections": []
        }

        #4. Parse Detections and Annotate
        for box in result.boxes:
            conf = float(box.conf[0])

            if conf >= confidence_threshold:
                class_id = int(box.cls[0])
                label = class_names[class_id]

                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                detection_log["detections"].append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [int(x) for x in [x1, y1, x2, y2]]
                })

                color = colors.get(label, (255, 255, 255))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label_text = f"{label}: {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                cv2.putText(img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        #5. Save Outputs
        output_image_path = os.path.join(annotated_img_folder, filename)
        cv2.imwrite(output_image_path, img)

        log_path = os.path.join(log_folder, os.path.splitext(filename)[0] + '.json')
        with open(log_path, 'w') as f:
            json.dump(detection_log, f, indent=2)

    print(f"\n✅ Done! Annotated images saved to: {annotated_img_folder}")
    print(f"✅ JSON logs saved to: {log_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects using a custom YOLOv8 model.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input folder with images.")
    parser.add_argument('--output', type=str, default='output', help="Path to the root output folder.")
    parser.add_argument('--model', type=str, default='best.pt', help="Path to the custom-trained .pt model file.")
    parser.add_argument('--confidence', type=float, default=0.5, help="Confidence threshold for detections.")

    args = parser.parse_args()

    detect_hands(args.input, args.output, args.model, args.confidence)