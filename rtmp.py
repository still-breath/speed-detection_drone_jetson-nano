import cv2

# Ganti IP dengan alamat server RTMP (bisa dari drone, laptop, OBS, dll)
stream_url = "rtmp://192.168.201.150:1935/streams"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Gagal membuka stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Tidak ada frame, atau stream berakhir")
        break

    # 👉 Di sini kamu bisa pasang YOLOv8, speed estimator, dll
    cv2.imshow("RTMP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

