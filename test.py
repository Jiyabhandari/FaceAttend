import os
import re
import cv2

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = "Trainner.yml"

# Check if the trained model exists
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()

recognizer.read(model_path)  # Load trained model

# Path to test images
test_images_path = r"C:\Users\uniks\Downloads\Face-Recognition-Based-Attendance-Monitoring-System\Face Recognition Based Attendance Monitoring System\TrainingImage"

# Check if the folder exists
if not os.path.exists(test_images_path):
    print(f"Error: Test images folder '{test_images_path}' not found!")
    exit()

# Process each image in the test folder
for filename in os.listdir(test_images_path):
    filepath = os.path.join(test_images_path, filename)

    # Ensure it's an image file
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # Skip non-image files

    print(f"\nProcessing: {filename}")

    # Extract numeric label from filename
    match = re.search(r'\d+', filename)
    
    if match:
        actual_label = int(match.group())  # Get the first number found
    else:
        print(f"Warning: No numeric label found in '{filename}', skipping...")
        continue

    # Load image in grayscale
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: Failed to load '{filename}', skipping...")
        continue

    # Predict using the trained model
    predicted_label, confidence = recognizer.predict(image)

    print(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}, Confidence: {confidence}")
