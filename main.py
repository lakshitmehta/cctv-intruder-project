# --------------------------------------------------------------
# Real-Time CCTV Intruder Detection System Using Face Embeddings
# --------------------------------------------------------------

import cv2
import insightface
from dotenv import load_dotenv
import numpy as np
from threading import Thread, Lock
import smtplib
from email.message import EmailMessage
import time, os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
     accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
load_dotenv()
EMAIL = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

# --------------------------------------------------------------
# CONFIGURATION FOLDERS
# --------------------------------------------------------------
AUTHORIZED_FOLDER = "authorized_faces"   # Folder containing images of authorized user
INTRUDER_FOLDER = "intruder_faces"       # Folder of intruder sample images
ML_REPORT_FILE = "ml_evaluation_report.txt"


# --------------------------------------------------------------
# LOAD InsightFace Model
# --------------------------------------------------------------
print("🧠 Loading InsightFace model...")

# Create a face analysis object using CPU
model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])

# det_size = input size for face detector. 224x224 = good balance of speed & accuracy
model.prepare(ctx_id=0, det_size=(224, 224))

print("✅ InsightFace model ready.\n")


# --------------------------------------------------------------
# FUNCTION: Extract 512D Embeddings From a Folder
# --------------------------------------------------------------
def extract_embeddings(folder, label):
    """
    For each image in a folder:
    1. Load image
    2. Detect face using InsightFace
    3. Extract 512D embedding
    4. Append embedding + label (authorized =1, intruder =0)
    """
    X, y = [], []

    if not os.path.exists(folder):
        return X, y

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        faces = model.get(img)   # Detect faces + produce embeddings

        if len(faces) > 0:
            X.append(faces[0].embedding)  # Take first detected face embedding
            y.append(label)

    return X, y


# --------------------------------------------------------------
# PART 1 — MACHINE LEARNING EVALUATION USING EMBEDDINGS
# --------------------------------------------------------------
def run_evaluation():
    print("🔎 Running ML Evaluation...")

    # Load embeddings from both folders
    auth_X, auth_y = extract_embeddings(AUTHORIZED_FOLDER, 1)
    intr_X, intr_y = extract_embeddings(INTRUDER_FOLDER, 0)

    # Combine into one dataset
    X = np.array(auth_X + intr_X)
    y = np.array(auth_y + intr_y)

    # Not enough samples → skip ML evaluation
    if len(X) < 4 or len(np.unique(y)) < 2:
        msg = "⚠️ Not enough data for ML evaluation.\n"
        print(msg)
        with open(ML_REPORT_FILE, "w") as f:
            f.write(msg)
        return

    print(f"Dataset: {len(X)} samples → Authorized: {sum(y)}, Intruders: {len(y)-sum(y)}")

    # Split dataset (70% train, 30% test)
    # Stratify = keeps same proportion of authorized/intruder
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )


    # Dictionary of ML models used for evaluation
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=2),           # K=2 = best balance for small embedding dataset
        "Logistic Regression": LogisticRegression(max_iter=2000),  # More iterations for convergence
        "Linear SVM": SVC(kernel='linear'),                   # Linear because embeddings already separable
        "Random Forest": RandomForestClassifier(n_estimators=100),  # 100 trees = stable accuracy
        "Naive Bayes": GaussianNB()                           # Works well on clean embeddings
    }

    all_results = {}

    print("\n================== ML MODEL RESULTS ==================\n")

    # Train & evaluate each model
    for name, clf in models.items():
        clf.fit(X_train, y_train)      # Train model
        y_pred = clf.predict(X_test)   # Predict on test set

        # Calculate ML metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        all_results[name] = [accuracy, precision, recall, f1]

        print(f"▶ MODEL: {name}")
        print(f"   Accuracy : {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall   : {recall*100:.2f}%")
        print(f"   F1 Score : {f1*100:.2f}%\n")


    # ------------------------------------------------------
    # PCA VISUALIZATION (512D → 2D scatter plot)
    # ------------------------------------------------------
    print("📌 Generating PCA plot...")

    pca = PCA(n_components=2)    # Only 2 because we want 2D plot
    X_2D = pca.fit_transform(X)  # Reduce 512D to 2D keeping max variance

    plt.figure(figsize=(6,5))
    plt.scatter(X_2D[:,0], X_2D[:,1], c=y, cmap="viridis", s=40)

    plt.title("Authorized vs Intruder Face Clusters")
    plt.xlabel("Left = Intruders   |   Right = Authorized")
    plt.ylabel("Face Variation")

    plt.tight_layout()
    plt.savefig("pca_plot.png")
    plt.close()


    # ------------------------------------------------------
    # MODEL COMPARISON SUMMARY WRITTEN TO FILE
    # ------------------------------------------------------
    print("\n📌 MODEL COMPARISON SUMMARY\n")
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1"
    ))

    with open("model_comparison.txt", "w") as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("{:<20} {:<10} {:<10} {:<10} {:<10}\n".format(
            "Model", "Accuracy", "Precision", "Recall", "F1"
        ))

        for m, (a,p,r,f1_) in all_results.items():
            print("{:<20} {:.2f}%     {:.2f}%     {:.2f}%     {:.2f}%".format(
                m, a*100, p*100, r*100, f1_*100
            ))
            f.write("{:<20} {:.2f}      {:.2f}      {:.2f}      {:.2f}\n".format(
                m, a*100, p*100, r*100, f1_*100
            ))

    print("\n📁 Saved: pca_plot.png, model_comparison.txt")
    print("\n============== END OF ML EVALUATION ==============\n")

# Run ML Evaluation
run_evaluation()


# --------------------------------------------------------------
# PART 2 — REAL-TIME INTRUDER DETECTION USING WEBCAM
# --------------------------------------------------------------

class VideoStream:
    """
    Reads webcam frames in a separate background thread.
    This prevents lag & ensures latest frame is always available.
    """

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)  # Open webcam

        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 640px width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 480px height
        self.cap.set(cv2.CAP_PROP_FPS, 30)            # 30 frames per sec

        # Buffer size = 1 → Always keep only latest frame (reduces delay)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # FOURCC → format for faster frame decoding
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))

        # Initialize frame and thread control
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Lock()  # Prevents race conditions


    def start(self):
        # Start frame grabbing in background thread
        Thread(target=self.update, daemon=True).start()
        return self


    def update(self):
        # Continuously grab frames until stopped
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                # Use lock so only one thread accesses frame at a time
                with self.lock:
                    self.ret, self.frame = ret, frame


    def read(self):
        # Safely return latest frame
        with self.lock:
            return self.ret, self.frame.copy()


    def stop(self):
        # Stop thread + release camera
        self.stopped = True
        self.cap.release()



# --------------------------------------------------------------
# LOAD AVERAGE EMBEDDING OF AUTHORIZED USER
# --------------------------------------------------------------
def load_known_faces(folder_path="authorized_faces"):
    """
    Extract embeddings of all authorized images,
    compute average embedding for stable comparison.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    embeddings = []

    for file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, file))
        if img is None:
            continue

        faces = model.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)

    if len(embeddings) == 0:
        raise ValueError("No faces found.")

    # Average embedding → smoother and more robust
    return np.mean(embeddings, axis=0)


# --------------------------------------------------------------
# EMAIL ALERT FUNCTION
# --------------------------------------------------------------

sending_lock = Lock()
sending_in_progress = set()

def send_intruder_email(image_path, intruder_id):
    """
    Send an email with intruder image.
    Uses SMTP_SSL on port 465.
    """
    def _send():
        try:
            msg = EmailMessage()
            msg['Subject'] = "🚨 Intruder Detected!"
            msg['From'] = EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg.set_content("Intruder detected. See attached image.")

            # Attach intruder face image
            with open(image_path, 'rb') as f:
                msg.add_attachment(f.read(), maintype='image', subtype='jpeg')

            # SMTP_SSL(port=465) = secure encrypted email transfer
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL, EMAIL_PASSWORD)
                smtp.send_message(msg)

        except Exception as e:
            print("Email error:", e)

        # Remove from active sending list
        with sending_lock:
            sending_in_progress.discard(intruder_id)

    # Prevent sending multiple emails for same intruder
    with sending_lock:
        if intruder_id not in sending_in_progress:
            sending_in_progress.add(intruder_id)
            Thread(target=_send, daemon=True).start()



# --------------------------------------------------------------
# COSINE SIMILARITY FUNCTION (Embedding Comparison)
# --------------------------------------------------------------
def cosine_similarity(a, b):
    """
    Returns cosine similarity between two embeddings.
    1 = identical direction
    0 = no similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)



# --------------------------------------------------------------
# MAIN REAL-TIME LOOP
# --------------------------------------------------------------
def run_realtime():
    # Load authorized person template embedding
    known_face = load_known_faces(AUTHORIZED_FOLDER)

    # Start webcam thread
    vs = VideoStream(0).start()

    # Parameters
    SCALE_FACTOR = 2     # Shrink image before face detection (faster)
    SKIP_FRAMES = 4      # Process every 4th frame
    DETECTION_THRESHOLD = 0.5  # Similarity > 0.5 → authorized
    EMAIL_DELAY = 2      # Intruder must persist for 2 sec
    CLEANUP_TIMEOUT = 10 # Remove memory of old intruders

    intruder_memory = []  # Track intruders over time
    frame_counter = 0

    print("🎥 Starting real-time detection... Press 'q' to exit.")

    while True:
        ret, frame = vs.read()
        if not ret:
            continue

        # Mirror effect for natural look
        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # Skip frames to reduce CPU load
        if frame_counter % SKIP_FRAMES == 0:
            # Resize for faster detection
            small = cv2.resize(frame, (0,0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)

            faces = model.get(small)  # Detect faces in small frame

            stored_results = []
            current_intruders = []
            now = time.time()

            for face in faces:
                # Scale bounding box back to full resolution
                bbox = (face.bbox * SCALE_FACTOR).astype(int)
                x1, y1, x2, y2 = bbox
                embedding = face.embedding

                # Compare embedding with authorized user
                similarity = cosine_similarity(known_face, embedding)

                if similarity > DETECTION_THRESHOLD:
                    label = f"Authorized ({similarity:.2f})"
                    color = (0,255,0)
                else:
                    label = f"Intruder ({similarity:.2f})"
                    color = (0,0,255)
                    # Save for intruder tracking
                    current_intruders.append({"embedding": embedding, "bbox": bbox})

                stored_results.append({"bbox": bbox, "label": label, "color": color})


            # --------------------------------------------------------
            # INTRUDER TRACKING (to avoid duplicate email alerts)
            # --------------------------------------------------------
            for idx, intr in enumerate(current_intruders):
                matched = False

                for mem in intruder_memory:
                    # Check if same intruder across frames (similar > 0.7)
                    sim = cosine_similarity(mem["embedding"], intr["embedding"])

                    if sim > 0.7:
                        mem["last_seen"] = now

                        # If intruder present for >2 sec → send alert
                        if not mem.get("email_sent", False):
                            if "start_time" not in mem:
                                mem["start_time"] = now
                            elif now - mem["start_time"] > EMAIL_DELAY:
                                x1,y1,x2,y2 = intr["bbox"]
                                crop = frame[y1:y2, x1:x2]
                                path = f"intruder_{idx}.jpg"
                                cv2.imwrite(path, crop)
                                send_intruder_email(path, idx)
                                mem["email_sent"] = True

                        matched = True
                        break

                # If completely new intruder → add to tracking list
                if not matched:
                    intruder_memory.append({
                        "embedding": intr["embedding"],
                        "start_time": now,
                        "email_sent": False,
                        "last_seen": now
                    })

            # Remove intruders not seen recently
            intruder_memory = [
                mem for mem in intruder_memory
                if now - mem["last_seen"] < CLEANUP_TIMEOUT
            ]

        # --------------------------------------------------------
        # DRAW BOUNDING BOXES & LABELS
        # --------------------------------------------------------
        try:
            for res in stored_results:
                x1, y1, x2, y2 = res["bbox"]
                cv2.rectangle(frame, (x1,y1), (x2,y2), res["color"], 2)
                cv2.putText(frame, res["label"], (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, res["color"], 2)
        except:
            pass

        # Display video feed
        cv2.imshow("Real-Time Intruder Detection", frame)

        # Quit window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()             # Stop webcam
    cv2.destroyAllWindows()  # Close windows


# Run Real-Time Detection
run_realtime()
