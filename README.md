# Aerial Drone Footage Transfer and Military Vehicle Detection

## Overview

This project simulates the transfer of aerial drone footage to a ground unit, which processes the video feed to identify and highlight green vehicles, potentially indicating military vehicles. The project demonstrates real-time video streaming, object detection using YOLOv5, and color classification using a TensorFlow model.

## Features

- **Real-time Video Streaming**: Captures video from a camera or video file and streams it to a server.
- **Object Detection**: Utilizes YOLOv5 for detecting various types of vehicles in the video feed.
- **Color Classification**: Uses a TensorFlow model to classify the color of detected vehicles, focusing on identifying green vehicles.
- **Alerts and Visualizations**: Highlights identified vehicles and displays alerts for green vehicles, along with various performance metrics.

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

2. **Run the Receiver**: Processes the received video feed, performs object detection and color classification.
    ```sh
    python src/receiver.py
    ```

## Detailed Description of Key Components

### Sender Code

**Initialize TCP Socket**: Connects to the server using the specified IP address and port.
```python
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((server_ip, server_port))
