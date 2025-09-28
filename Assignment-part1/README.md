# Part 1: Gloved vs. Ungloved Hand Detection

This project uses a custom-trained YOLOv8 object detection model to identify and locate gloved and ungloved hands in images. The system takes a folder of images as input, runs detection, and outputs annotated images along with JSON logs detailing each detection.

---
## Dataset and Model

* **Dataset:** Gloves Detection
* **Source:** Roboflow Universe
* **Link:** `https://universe.roboflow.com/ds/8I5w4aY00h?ref=ultralytics`
* **Details:** The dataset contains over 3,000 images with two classes: `glove` and `no-glove`.

* **Model Used:** YOLOv8n (the 'nano' version from Ultralytics).

---
## Preprocessing and Training

The model was trained in Google Colab using a T4 GPU. The data was prepared using Roboflow with the following steps:

* **Preprocessing:**
    * **Auto-Orient:** All images were standardized to be upright.
    * **Resize:** All images were stretched to a 640x640 pixel square to match the model's input size.

* **Augmentation:**
    * **Flip:** Horizontal flip.
    * **Brightness:** +/- 25%.
    * **Rotation:** +/- 15°.

* **Training:**
    * The model was trained for **25 epochs**.
    * The final trained model achieved a **mAP50 score of 87.7%** on the validation set.

---
## What Worked and What Didn’t

* **What Worked Well:**
    * The YOLOv8 model trained very quickly and achieved high accuracy, especially for the `glove` class (90.6% mAP50).
    * Using data augmentation was very effective in creating a robust model that can handle variations in images.
    * Roboflow's platform made the data preparation and augmentation process straightforward.

* **What Was Challenging:**
    * The model was less sensitive to the `no-glove` class (bare hands), with a lower recall score. This required lowering the confidence threshold during inference to detect them more reliably.
    * An initial run was mistakenly performed with an incorrect "palm detection" dataset, which highlighted the critical importance of verifying the dataset and its classes before training.

---
## How to Run the Script

1.  **Setup Environment:**
    * Create a Python virtual environment.
    * Install the required packages: `pip install ultralytics opencv-python tqdm`

2.  **Prepare Files:**
    * Place the trained model file, `best.pt`, in the main project folder.
    * Create a folder named `input_images` and place your test images inside it.

3.  **Run Detection:**
    * Execute the script from your terminal. It is recommended to use a lower confidence to better detect the `no-glove` class.
    * `python detection_script.py --input input_images --output output --confidence 0.3`

4.  **Check Results:**
    * Annotated images will be saved in `output/images/`.
    * JSON logs will be saved in `output/logs/`.