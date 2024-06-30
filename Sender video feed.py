import socket
import cv2
import struct
import pickle

# Define your server's IP address and port
server_ip = "127.0.0.1"
server_port = 1234

# Create a TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Try to connect to the server
while True:
    try:
        s.connect((server_ip, server_port))
        print("Socket successfully created and connected to the server.")
        break
    except ConnectionRefusedError:
        print("Failed to connect to the server. Retrying...")
        continue

cap = cv2.VideoCapture(0)  # Start video capture. You can replace '0' with your video file path.

if not cap.isOpened():
    print("Failed to open video capture.")
else:
    print("Video capture opened successfully.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    result, frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data = pickle.dumps(frame, 0)
    size = len(data)

    # Send size of data and then data
    s.sendall(struct.pack(">L", size) + data)

cap.release()
