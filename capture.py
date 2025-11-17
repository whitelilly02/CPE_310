import streamlit as st # pyright: ignore[reportMissingImports]
import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import os
from datetime import datetime

def video_stream(model, confidence_threshold):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access the webcam. Please check your camera connection.")
        return

    frame_placeholder = st.empty()
    label_count_placeholder = st.empty()
    stop_button = st.button("Stop Webcam")

    #  สร้างโฟลเดอร์เก็บภาพถ่าย
    save_dir = "captured_no_helmet"
    os.makedirs(save_dir, exist_ok=True)

    last_capture_time = 0  # ป้องกันการถ่ายรัว
    capture_interval = 5   # หน่วงเวลาการถ่าย (วินาที)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read frame from webcam.")
            break

        results = model(frame, conf=confidence_threshold)
        label_counts = defaultdict(int)
        
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box

                if conf >= confidence_threshold:
                    label = f"{model.names[int(cls)]}"
                    label_counts[label] += 1

                    # ตรวจจับหมวก
                    if label.lower() in ["with helmet", "helmet", "helmet_on"]:
                        color = (0, 255, 0)  # เขียว = ใส่หมวก
                    else:
                        color = (0, 0, 255)  # แดง = ไม่ใส่หมวก

                        # ถ่ายภาพเมื่อเจอ "No Helmet"
                        current_time = time.time()
                        if current_time - last_capture_time > capture_interval:
                            #  ตัดเฉพาะบริเวณคนนั้น
                            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{save_dir}/no_helmet_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)  # บันทึกทั้งภาพ
                            st.warning(f"บันทึกภาพผู้ไม่สวมหมวก: {filename}")
                            last_capture_time = current_time

                    # วาดกรอบ
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # แปลงสีและแสดงผล
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # แสดงจำนวนการตรวจจับ
        label_count_placeholder.markdown("### Object Counts:")
        for label, count in label_counts.items():
            label_count_placeholder.write(f"- **{label}**: {count}")

        if stop_button:
            break

        time.sleep(0.03)

    cap.release()


# ฟังก์ชันหลัก
def main():
    st.title("Helmet Detection System with Auto Capture")
    st.write("ระบบตรวจจับหมวกกันน็อคและถ่ายรูปผู้ไม่สวมหมวกอัตโนมัติ")

    model_path = r"C:\CPE_310\Project_helmet\runs\detect\train2\weights\best.pt"

    st.info("Loading model...")
    model = YOLO(model_path)
    st.success("Model loaded successfully.....")

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

    start_button = st.button("Start Webcam")
    if start_button:
        video_stream(model, confidence_threshold)

main()
