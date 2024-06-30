import socket
import cv2
import struct
import pickle
import torch
import numpy as np
import threading
import tensorflow as tf

# Load the YOLOv5 model
yolo_model = torch.hub.load('C:/Users/sarpa/Desktop/yolov5', 'custom',
                            path='C:/Users/sarpa/Desktop/best.pt',
                            source='local')

# Load the TensorFlow model
tf_model = tf.keras.models.load_model('C:/Users/sarpa/Desktop/modelEpoch50.h5')

# Labels for the YOLOv5 model
labels = {0: 'boat', 1: 'camping car', 2: 'car', 3: 'motorcycle', 4: 'other',
          5: 'pickup', 6: 'plane', 7: 'tractor', 8: 'truck', 9: 'van'}

# Define your server's IP address and port
server_ip = "127.0.0.1"
server_port = 1234

# Create a TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the server's IP address and port
s.bind((server_ip, server_port))
s.listen(10)

print("Socket successfully created. Listening for connections...")

conn, addr = s.accept()

print(f"Accepted connection from {addr}.")

data = b""
payload_size = struct.calcsize(">L")
frame_data = b""
new_frame_data = None
frame_count = 0
vehicle_count = 0
alert_count = 0

def recv_all():
    global data, frame_data, new_frame_data
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)

        new_frame_data = data[:msg_size]
        data = data[msg_size:]

# start receiving all data in another thread
threading.Thread(target=recv_all, daemon=True).start()

while True:
    if new_frame_data != frame_data:
        frame_data = new_frame_data

        # Skip if frame_data is None
        if frame_data is None:
            continue

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # Process frame with YOLOv5 model
        yolo_results = yolo_model(frame)

        # Get bounding boxes and labels
        boxes = yolo_results.xyxy[0].numpy()

        vehicle_images = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            # Check for non-zero width and height
            if x2 > x1 and y2 > y1:
                cls = int(box[5])  # class id

                if cls in labels:  # check if class id exists in labels
                    cls_name = labels[cls]  # get the class name from the YOLOv5 results
                    print(f"Detected a {cls_name}.")
                    vehicle_count += 1

                    region = frame[y1:y2, x1:x2]

                    # Process the region with TensorFlow model
                    region = cv2.resize(region, (244, 244))  # adjust this if your model expects a different size
                    region = region / 255.0  # normalize pixel values if your model expects this
                    region = np.expand_dims(region, axis=0)  # expand dimensions to match model input shape
                    prediction = tf_model.predict(region)

                    # Add color prediction to the bounding box label
                    color = np.argmax(prediction)
                    if color == 0:  # adjust this based on your model's color classes
                        color_name = 'brown'
                    elif color == 1:
                        color_name = 'gold'
                    elif color == 2:
                        color_name = 'green'
                    else:
                        color_name = 'other'

                    if color_name in ['brown', 'gold', 'green']:
                        print(f"Alert! Detected a {color_name} vehicle.")
                        alert_count += 1

                    # Add bounding box and label to the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{cls_name}, {color_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Display the region in a separate window
                    region_3d = np.squeeze(region)  # remove singleton dimensions

                    # Add a thinner border to the vehicle image
                    region_3d = cv2.copyMakeBorder(region_3d, top=5, bottom=5, left=5, right=5,
                                                   borderType=cv2.BORDER_CONSTANT, value=[0, 255, 0])

                    # Add a label to the vehicle image with black color
                    cv2.putText(region_3d, f'{cls_name}, {color_name}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    vehicle_images.append(region_3d)

        # Concatenate all the vehicle images together and display them in a separate window
        if vehicle_images:
            all_vehicles_image = np.concatenate(vehicle_images, axis=1)  # concatenate along the width axis
            cv2.imshow('All Vehicles', all_vehicles_image)

        frame_count += 1
        print(f"Received and processed frame {frame_count}")

        # Display counts on the frame
        cv2.putText(frame, f'Alerts: {alert_count}, Total vehicles: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('ImageWindow', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Closing connection and window...")
cv2.destroyAllWindows()
