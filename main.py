import cv2
import insightface
import numpy as np
from threading import Thread, Lock
import smtplib
from email.message import EmailMessage
import time
import os 
# -----------------------------
# Webcam capture thread class
# -----------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

# -----------------------------
# Load Face Model (GPU)
# -----------------------------
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(224, 224))  # small detector for speed

# -----------------------------
# Load known faces from a folder and create an average embedding
# -----------------------------
def load_known_faces(folder_path='authorized_faces'):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: '{folder_path}'. Please create it and add your photos.")

    known_embeddings = []
    for filename in os.listdir(folder_path):
        # Construct the full image path
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"⚠️ Could not read image: {filename}. Skipping.")
            continue
            
        # Get face embedding from the image
        faces = model.get(img)
        
        if len(faces) > 0:
            # Add the first detected face's embedding to our list
            known_embeddings.append(faces[0].embedding)
            print(f"✅ Processed {filename}")
        else:
            print(f"⚠️ No face detected in {filename}. Skipping.")

    if len(known_embeddings) == 0:
        raise ValueError(f"❌ No faces could be processed from the '{folder_path}' folder. Make sure it contains clear face photos.")

    # Calculate the average of all collected embeddings
    average_embedding = np.mean(known_embeddings, axis=0)
    print(f"✅ Average face embedding created successfully from {len(known_embeddings)} images.")
    return average_embedding

# Load the average embedding which will be used for comparison
known_face_embedding = load_known_faces()

# -----------------------------
# Email sending function
# -----------------------------
sending_lock = Lock()
sending_in_progress = set()

def send_intruder_email(image_path, intruder_id):
    def _send():
        try:
            msg = EmailMessage()
            msg['Subject'] = "🚨 Intruder Detected!"
            msg['From'] = "lakshitmehta1work@gmail.com"
            msg['To'] = "lakshitmehta1work@gmail.com"
            msg.set_content("An intruder has been detected! See attached photo.")

            with open(image_path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='intruder.jpg')

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login("lakshitmehta1work@gmail.com", "dwuovygrfvbjgobb")
                smtp.send_message(msg)
            print(f"✅ Intruder email sent for intruder {intruder_id}!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
        finally:
            with sending_lock:
                sending_in_progress.discard(intruder_id)

    with sending_lock:
        if intruder_id not in sending_in_progress:
            sending_in_progress.add(intruder_id)
            Thread(target=_send, daemon=True).start()

# -----------------------------
# Start webcam thread
# -----------------------------
vs = VideoStream(0).start()

intruder_memory = []
DETECTION_THRESHOLD = 0.5
EMAIL_DELAY = 2
CLEANUP_TIMEOUT = 10
SCALE_FACTOR = 2  # How much to shrink the image for detection (2 = half size)

# -----------------------------
# Main Processing Loop
# -----------------------------
frame_counter = 0
SKIP_FRAMES = 4
stored_results = []

while True:
    ret, frame = vs.read()
    if not ret:
        continue

    # Mirror webcam first
    frame = cv2.flip(frame, 1)
    frame_counter += 1

    # ---- 1. HEAVY PROCESSING BLOCK (RUNS PERIODICALLY) ----
    if frame_counter % SKIP_FRAMES == 0:
        # ---- NEW: Create a smaller frame for faster processing ----
        small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)
        
        # ---- MODIFIED: Run detection on the smaller frame ----
        faces = model.get(small_frame)
        current_intruders = []
        stored_results.clear()

        for face in faces:
            # ---- MODIFIED: Scale bounding box back to original frame size ----
            bbox = face.bbox * SCALE_FACTOR
            x1, y1, x2, y2 = bbox.astype(int)

            embedding = face.embedding
            similarity = np.dot(known_face_embedding, embedding) / (
                np.linalg.norm(known_face_embedding) * np.linalg.norm(embedding)
            )

            if similarity > DETECTION_THRESHOLD:
                label = "Authorized"
                color = (0, 255, 0)
            else:
                label = "Intruder"
                color = (0, 0, 255)
                # Use the scaled-up coordinates
                current_intruders.append({'embedding': embedding, 'bbox': np.array([x1, y1, x2, y2])})

            # Save the results to be drawn on every frame
            stored_results.append({'bbox': (x1, y1, x2, y2), 'label': label, 'color': color})

        # ---- Intruder tracking & email logic (no changes needed here) ----
        now = time.time()
        for idx, intruder in enumerate(current_intruders):
            matched = False
            for mem in intruder_memory:
                sim = np.dot(mem['embedding'], intruder['embedding']) / (
                    np.linalg.norm(mem['embedding']) * np.linalg.norm(intruder['embedding'])
                )
                if sim > 0.7:
                    matched = True
                    mem['last_seen'] = now
                    if not mem['email_sent']:
                        if 'start_time' not in mem:
                            mem['start_time'] = now
                        elif now - mem['start_time'] >= EMAIL_DELAY:
                            x1_intruder, y1_intruder, x2_intruder, y2_intruder = intruder['bbox'].astype(int)
                            # Crop from the original full-size frame
                            face_crop = frame[y1_intruder:y2_intruder, x1_intruder:x2_intruder]
                            intruder_image_path = f"intruder_{idx}.jpg"
                            cv2.imwrite(intruder_image_path, face_crop)
                            send_intruder_email(intruder_image_path, idx)
                            mem['email_sent'] = True
                    break
            if not matched:
                intruder_memory.append({'embedding': intruder['embedding'], 'start_time': now, 'email_sent': False, 'last_seen': now})

        # ---- Cleanup old intruders ----
        intruder_memory = [mem for mem in intruder_memory if now - mem.get('last_seen', now) <= CLEANUP_TIMEOUT]

    # ---- 2. DRAWING BLOCK (RUNS EVERY FRAME) ----
    for result in stored_results:
        x1, y1, x2, y2 = result['bbox']
        label = result['label']
        color = result['color']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ---- 3. DISPLAY FRAME ----
    cv2.imshow("Intruder Detection CCTV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()






















