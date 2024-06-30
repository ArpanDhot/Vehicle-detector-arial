# Aerial Drone Footage Transfer and Military Vehicle Detection

## Overview

This project simulates the transfer of aerial drone footage to a ground unit, which processes the video feed to identify and highlight green vehicles, potentially indicating military vehicles. The project leverages advanced computer vision techniques to perform real-time object detection and colour classification.

The sender component captures video from a camera or video file, encodes it, and transmits the frames to a server. The receiver component processes these frames using a YOLOv5 model for detecting various types of vehicles and a TensorFlow model for classifying their colours. If a green vehicle is detected, it is highlighted and an alert is generated. This setup mimics a practical application where drone footage is analysed on the ground to monitor and identify specific targets, enhancing situational awareness and decision-making in real-time scenarios.

The system also provides various performance metrics and visualisations to evaluate the effectiveness of the detection and classification models.

## Features

- **Real-time Video Streaming**: Captures video from a camera or video file and streams it to a server.
- **Object Detection**: Utilises YOLOv5 for detecting various types of vehicles in the video feed.
- **Color Classification**: Uses a TensorFlow model to classify the colour of detected vehicles, focusing on identifying green vehicles.
- **Alerts and Visualisations**: Highlights identified vehicles and displays alerts for green vehicles, along with various performance metrics.

## Demo

![Demo GIF](demo/test.gif)

## Getting Started

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- PyTorch
- TensorFlow

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/drone-footage-detection.git
    cd drone-footage-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Run the Sender**: Captures video and sends it to the server.
    ```sh
    python src/sender.py
    ```

2. **Run the Receiver**: Processes the received video feed, and performs object detection and color classification.
    ```sh
    python src/receiver.py
    ```

## Detailed Description of Key Components

### Sender Code

**Initialize TCP Socket**: Connects to the server using the specified IP address and port.
```python
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((server_ip, server_port))
