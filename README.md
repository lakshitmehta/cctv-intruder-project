CCTV Intruder Detection System

A real-time CCTV-based intruder detection system built with Python, OpenCV, and InsightFace.
The system detects faces from a live camera feed, compares them against authorized faces, and automatically sends an email alert with an attached image if an intruder is detected.

Features

Real-time video stream with multi-threaded capture (low latency).

Face recognition using InsightFace (GPU accelerated).

Authorized user verification via stored embeddings.

Intruder detection with automatic email alerts and image attachments.

Intruder photo saving handled in background threads to avoid blocking.

Optimized performance using frame skipping and scaled-down detection.

Requirements

Python 3.8+

A working webcam or IP camera

CUDA-enabled GPU (recommended for performance)

Python dependencies (see requirements.txt):

opencv-python
insightface
numpy

Installation

Clone the repository

git clone https://github.com/lakshitmehta/cctv-intruder-detection.git
cd cctv-intruder-detection


Create and activate a virtual environment (recommended)

python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows


Install dependencies

pip install -r requirements.txt

Setup

Authorized faces

Create a folder named authorized_faces/ in the project root.

Add clear, front-facing images of authorized people (JPG/PNG).

Email configuration

The script uses Gmail SMTP for alerts.

Enable 2-Step Verification on your Gmail and create an App Password.

Replace credentials in main.py:

msg['From'] = "your-email@gmail.com"
msg['To'] = "receiver-email@gmail.com"
smtp.login("your-email@gmail.com", "YOUR_APP_PASSWORD")

Usage

Run the system:

python main.py


Controls:

Press Q to quit the CCTV window.

Output:

Authorized users → Green box labeled Authorized.

Unknown faces → Red box labeled Intruder.

Intruder triggers:

Image saved locally (intruder_X.jpg)

Email alert with attached photo

Configuration

The following parameters in main.py can be adjusted:

DETECTION_THRESHOLD → Face similarity threshold (default: 0.5)

EMAIL_DELAY → Seconds before sending an alert (default: 2)

CLEANUP_TIMEOUT → Forget intruders after inactivity (default: 10)

SCALE_FACTOR → Image downscaling for faster detection (default: 3)

SKIP_FRAMES → Process every Nth frame (default: 6)

Project Structure
cctv-intruder-detection/
│── main.py               # Main program
│── authorized_faces/     # Store authorized face images here
│── intruder_X.jpg        # Intruder snapshots (generated automatically)
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation

Future Work

Support for multiple/IP cameras

Web-based monitoring dashboard

Push notifications (Telegram/WhatsApp)

Cloud storage for intruder images

Author

Lakshit Mehta

Email: lakshitmehta1work@gmail.com

GitHub: https://github.com/lakshitmehta

License

This project is licensed under the MIT License. You are free to use, modify, and distribute it.
