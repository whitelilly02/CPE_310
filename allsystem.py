import streamlit as st  # pyright: ignore[reportMissingImports]
import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import os
from datetime import datetime


def video_stream(model, confidence_threshold):
    # 🔹 ใช้กล้องจริง หรือไฟล์วิดีโอ
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(r"C:\CPE_310\Project_helmet\dataset_motorcycle\test\images\14571126_3840_2160_60fps.mp4")

    if not cap.isOpened():
        st.error("Cannot access the webcam or video file.")
        return

    frame_placeholder = st.empty()
    label_count_placeholder = st.empty()
    stop_button = st.button("Stop Webcam")

    # 🔹 โฟลเดอร์เก็บภาพถ่าย
    save_dir = "captured_no_helmet"
    os.makedirs(save_dir, exist_ok=True)

    last_capture_time = 0
    capture_interval = 5  # วินาที

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read frame from webcam/video.")
            break

        results = model(frame, conf=confidence_threshold)
        label_counts = defaultdict(int)

        
        motorcycle_detected = False
        no_helmet_detected = False
        helmet_detected = False

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                cls_id = int(cls)
                label = model.names[cls_id].strip().lower().replace("_", " ")
                # st.write(f"Detected: cls_id={cls_id}, label={label}, conf={conf:.2f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

                # label_counts[label] += 1

                if cls_id == 0:  # No helmet
                    color = (255, 0, 0)
                    no_helmet_detected = True
                elif cls_id == 1:  # Helmet
                    color = (0, 255, 255)
                    helmet_detected = True
                elif cls_id == 2:  # Motorcycle
                    color = (0, 0, 255)
                    motorcycle_detected = True
                else:
                    color = (255, 255, 255)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        
        current_time = time.time()
        if motorcycle_detected:
            if no_helmet_detected:
                # มีมอเตอร์ไซค์และไม่มีหมวก → บันทึกภาพ
                if current_time - last_capture_time > capture_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{save_dir}/motorcycle_no_helmet_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.warning(f"บันทึกภาพผู้ขับขี่มอเตอร์ไซค์ไม่สวมหมวก: {filename}")
                    last_capture_time = current_time
            elif helmet_detected:
                st.info("ผู้ขับขี่สวมหมวกกันน็อค — ไม่บันทึกภาพ")

        # แสดงผลบน Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        label_count_placeholder.markdown("### Object Counts:")
        for label, count in label_counts.items():
            label_count_placeholder.write(f"- **{label}**: {count}")

        if stop_button:
            break

        time.sleep(0.03)

    cap.release()


def main():
    st.title("Helmet Detection System with Auto Capture")
    st.write("ระบบตรวจจับหมวกกันน็อคและถ่ายรูปผู้ไม่สวมหมวกอัตโนมัติ")

    model_path = r"C:\CPE_310\Project_helmet\runs\detect\train8\weights\best.pt"

    st.info("Loading model...")
    model = YOLO(model_path)
    st.success("Model loaded successfully...")
    st.write(" Class names from model:", model.names)
    


    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

    start_button = st.button("Start Detection")
    if start_button:
        video_stream(model, confidence_threshold)


main()
