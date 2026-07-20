from ultralytics import YOLO
import cv2
import time
import os
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage


SENDER_EMAIL = "darunsureshpdy@gmail.com"
APP_PASSWORD = "**********************"
RECEIVER_EMAIL = "darunsuresh07@gmail.com"


model = YOLO("runs/detect/train-3/weights/best.pt")

CAMERA_NAME = "Main Entrance Webcam"
MODEL_NAME = "YOLO11n"

alert_count = 0
last_sent = 0


ALERT_FOLDER = "alerts"
os.makedirs(ALERT_FOLDER, exist_ok=True)


def send_email(image_path, confidence):
    global alert_count

    alert_count += 1

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    msg = EmailMessage()
    msg["Subject"] = f"🚨 Face Mask Alert #{alert_count}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.set_content(f"""
Face Mask Violation Detected

Status       : No Mask
Confidence   : {confidence*100:.2f}%
Time         : {current_time}
Camera       : {CAMERA_NAME}
AI Model     : {MODEL_NAME}
Alert ID     : {alert_count}
""")

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(image_path)
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        print("✅ Email Sent")

    except Exception as e:
        print("❌ Email Error:", e)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw detections (keep original YOLO boxes)
    annotated_frame = results[0].plot()

    # Check detections for No Mask
    for box in results[0].boxes:

        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[cls].lower()

        if class_name in [
            "no mask",
            "no_mask",
            "no-mask",
            "nomask",
            "without mask",
            "without_mask"
        ]:

            current_time = time.time()

            # Send alert every 7 seconds
            if current_time - last_sent > 7:

                filename = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(ALERT_FOLDER, filename)

                cv2.imwrite(image_path, frame)

                threading.Thread(
                    target=send_email,
                    args=(image_path, confidence),
                    daemon=True
                ).start()

                last_sent = current_time

    # Show result
    cv2.imshow("Face Mask Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("d"):
        break

cap.release()
cv2.destroyAllWindows()
