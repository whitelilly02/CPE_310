from ultralytics import YOLO

# โหลด model yolov11s (small)
model = YOLO(r"C:\CPE_310\Project_helmet\yolo11n.pt")  


results = model(r"C:\CPE_310\Project_helmet\test\images\BikesHelmets380_png_jpg.rf.1bb24eb3452ab4ff6116987c0ce0fa99.jpg")
results[0].show()