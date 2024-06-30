import pyautogui
import cv2
import numpy as np
import pickle
import socket
import struct

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

while True:
    # Capture screenshot of a specific screen
    screenshot = pyautogui.screenshot(region=(100, 300, 3500, 1500))  # Adjust the region according to your screen setup

    # Convert the screenshot to a numpy array and resize it to 720p
    frame = np.array(screenshot)
    frame = cv2.resize(frame, (1280, 720))

    # Compress the frame
    result, frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Prepare data for sending
    data = pickle.dumps(frame, 0)
    size = len(data)

    # Send size of data and then data
    s.sendall(struct.pack(">L", size) + data)
