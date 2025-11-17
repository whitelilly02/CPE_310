from ultralytics import YOLO

# โหลด model yolov11s (small)
model = YOLO("C:\CPE_310\Project_helmet\yolo11n.pt")  

# Train
model.train(
    data=r"C:\CPE_310\Project_helmet\dataset_motorcycle\data.yaml",  # path ไปยัง data.yaml
    epochs=125,
    imgsz=800,
    batch=16,   # กำหนดขนาด batch
    plots=True,
    device="0",
    workers = 0
)
