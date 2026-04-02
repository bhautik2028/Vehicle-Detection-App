from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load trained YOLO model
# -----------------------------
model = YOLO(r"runs\detect\train8\weights\best.pt")

# -----------------------------
# Take input image path from user
# -----------------------------
image_path = input("Enter image path: ").strip()

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# -----------------------------
# Run prediction
# -----------------------------
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# -----------------------------
# Count vehicles
# -----------------------------
vehicle_count = len(results[0].boxes)

# -----------------------------
# Plot result
# -----------------------------
boxed_img = results[0].plot()   # BGR image
boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(boxed_img)
plt.text(
    20, 40,
    f"Vehicle Count: {vehicle_count}",
    color="lime",
    fontsize=16,
    bbox=dict(facecolor="black", alpha=0.6)
)
plt.axis("off")
plt.show()

print(f"✅ Total Vehicles Detected: {vehicle_count}")
