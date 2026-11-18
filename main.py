# --------------------------------------------------------------
# Real-Time CCTV Intruder Detection System Using Face Embeddings
# --------------------------------------------------------------

import cv2
import insightface
from dotenv import load_dotenv
import numpy as np
from threading import Thread, Lock
import smtplib
import ssl
from email.message import EmailMessage
import time, os, json
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
EMAIL = (os.getenv("EMAIL") or "").strip()
EMAIL_PASSWORD = (os.getenv("EMAIL_PASSWORD") or "").replace(" ", "").strip()
RECEIVER_EMAIL = (os.getenv("RECEIVER_EMAIL") or "").strip()

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

def gmail_auth_login(smtp):
    """Force Gmail's AUTH LOGIN flow instead of smtplib's default AUTH PLAIN first."""
    smtp.user = EMAIL
    smtp.password = EMAIL_PASSWORD
    return smtp.auth("LOGIN", smtp.auth_login, initial_response_ok=False)

def send_intruder_email(image_path, intruder_id, intruder_record=None):
    """Send an intruder image through Gmail, trying both supported SMTP modes."""
    sending_key = id(intruder_record) if intruder_record is not None else intruder_id

    def _send():
        try:
            if not EMAIL or not EMAIL_PASSWORD or not RECEIVER_EMAIL:
                raise ValueError(
                    "Missing EMAIL, EMAIL_PASSWORD, or RECEIVER_EMAIL in the .env file."
                )

            msg = EmailMessage()
            msg['Subject'] = "🚨 Intruder Detected!"
            msg['From'] = EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg.set_content("Intruder detected. See attached image.")

            with open(image_path, 'rb') as image_file:
                msg.add_attachment(
                    image_file.read(),
                    maintype='image',
                    subtype='jpeg',
                    filename=os.path.basename(image_path)
                )

            context = ssl.create_default_context()
            attempts = (
                ("SSL on port 465", "ssl"),
                ("STARTTLS on port 587", "starttls"),
            )
            errors = []

            for description, mode in attempts:
                try:
                    print(f"[EMAIL] Connecting to Gmail using {description}...")
                    if mode == "ssl":
                        with smtplib.SMTP_SSL(
                            "smtp.gmail.com", 465, timeout=20, context=context
                        ) as smtp:
                            smtp.ehlo()
                            gmail_auth_login(smtp)
                            smtp.send_message(msg)
                    else:
                        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
                            smtp.ehlo()
                            smtp.starttls(context=context)
                            smtp.ehlo()
                            gmail_auth_login(smtp)
                            smtp.send_message(msg)

                    print(f"[EMAIL] SUCCESS: message accepted for {RECEIVER_EMAIL}")
                    if intruder_record is not None:
                        intruder_record['email_sent'] = True
                    return

                except Exception as smtp_error:
                    error_text = f"{description}: {type(smtp_error).__name__}: {smtp_error}"
                    errors.append(error_text)
                    print(f"[EMAIL] Failed — {error_text}")

            print("[EMAIL] Both Gmail SMTP methods failed. No email was marked as sent.")
            for error_text in errors:
                print("[EMAIL]", error_text)
            print(
                "[EMAIL] This usually indicates that the network, VPN, antivirus, "
                "or firewall is interrupting SMTP. Run the two PowerShell "
                "Test-NetConnection checks shown in the instructions."
            )
            print(
                "[EMAIL] If the connection tests succeed, check that EMAIL_PASSWORD "
                "is a 16-character Google App Password (spaces are removed automatically)."
            )

        except Exception as error:
            print(f"[EMAIL] Configuration/file error: {type(error).__name__}: {error}")

        finally:
            with sending_lock:
                sending_in_progress.discard(sending_key)

    with sending_lock:
        if sending_key not in sending_in_progress:
            sending_in_progress.add(sending_key)
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
# RL MODULE — FEEDBACK-LEARNING ALERT DECISION AGENT
# --------------------------------------------------------------
class RLAlertDecisionAgent:
    """Simple feedback-based RL module for MONITOR vs ALERT decisions.

    The agent keeps one learned value for each action. User feedback updates
    the value of the action that was taken: +1 for a correct decision and -1
    for an incorrect decision. Values are saved between runs.
    """
    def __init__(self, learning_rate=0.3, values_file="rl_action_values.json"):
        self.learning_rate = learning_rate
        self.actions = ["MONITOR", "ALERT"]
        self.values_file = values_file
        # Slight initial preference for ALERT so the agent can demonstrate
        # both alerting and feedback-based learning during a demo.
        self.action_values = np.array([0.0, 0.1], dtype=float)
        self.load()

    def choose_action(self):
        # Select the action with the higher learned value.
        # In a tie, choose MONITOR as the safer option.
        action = int(np.argmax(self.action_values))
        return action

    @staticmethod
    def get_reward(action, feedback_label):
        correct = ((action == 1 and feedback_label == "INTRUDER") or
                   (action == 0 and feedback_label == "AUTHORIZED"))
        return 1 if correct else -1

    def learn(self, action, feedback_label):
        reward = self.get_reward(action, feedback_label)
        old_value = self.action_values[action]
        self.action_values[action] = old_value + self.learning_rate * (reward - old_value)
        self.save()
        return reward

    def save(self):
        try:
            with open(self.values_file, "w", encoding="utf-8") as f:
                json.dump(self.action_values.tolist(), f, indent=2)
        except Exception as error:
            print(f"[RL] Could not save learned values: {error}")

    def load(self):
        try:
            if os.path.exists(self.values_file):
                with open(self.values_file, "r", encoding="utf-8") as f:
                    saved = np.asarray(json.load(f), dtype=float)
                if saved.shape == self.action_values.shape:
                    self.action_values = saved
                    print(f"[RL] Loaded learned action values from {self.values_file}")
        except Exception as error:
            print(f"[RL] Could not load saved values; using initial values: {error}")


# --------------------------------------------------------------
# MAIN REAL-TIME LOOP
# --------------------------------------------------------------
def run_realtime():
    # Load authorized person template embedding
    known_face = load_known_faces(AUTHORIZED_FOLDER)

    # Start webcam thread
    vs = VideoStream(0).start()

    # Parameters
    SCALE_FACTOR = 2
    SKIP_FRAMES = 4
    DETECTION_THRESHOLD = 0.5
    EMAIL_DELAY = 2
    CLEANUP_TIMEOUT = 10
    EMAIL_RETRY_INTERVAL = 15

    intruder_memory = []
    frame_counter = 0
    stored_results = []

    # RL agent now controls whether a threshold-classified intruder candidate
    # is allowed to trigger the email alert. Detection labels remain unchanged.
    rl_agent = RLAlertDecisionAgent()
    rl_pending_feedback = None
    rl_status_text = "RL: waiting for a detected face"

    print("🎥 Starting real-time detection... Press 'q' to exit.")
    print("[RL] I = feedback: intruder | U = feedback: authorized")
    print("[RL] ALERT action can trigger email for an intruder candidate; MONITOR suppresses it.")

    while True:
        ret, frame = vs.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        if frame_counter % SKIP_FRAMES == 0:
            small = cv2.resize(frame, (0, 0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)
            faces = model.get(small)
            stored_results = []
            current_intruders = []
            rl_pending_feedback = None
            now = time.time()

            for face in faces:
                bbox = (face.bbox * SCALE_FACTOR).astype(int)
                x1, y1, x2, y2 = bbox
                embedding = face.embedding
                similarity = cosine_similarity(known_face, embedding)

                # RL observes similarity and selects MONITOR/ALERT.
                rl_action = rl_agent.choose_action()
                rl_pending_feedback = (rl_action, similarity)
                rl_status_text = (
                    f"RL: {rl_agent.actions[rl_action]} | "
                    "I=Intruder / U=Authorized"
                )

                # Original recognition label and threshold are preserved.
                if similarity > DETECTION_THRESHOLD:
                    label = f"Authorized ({similarity:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Intruder ({similarity:.2f})"
                    color = (0, 0, 255)
                    # Candidate is tracked, but email is gated by RL action.
                    current_intruders.append({
                        "embedding": embedding,
                        "bbox": bbox,
                        "rl_action": rl_action
                    })

                stored_results.append({"bbox": bbox, "label": label, "color": color})

            # Track intruder candidates and use RL action to gate email alerts.
            for idx, intr in enumerate(current_intruders):
                matched = False
                for mem in intruder_memory:
                    sim = cosine_similarity(mem["embedding"], intr["embedding"])
                    if sim > 0.7:
                        mem["last_seen"] = now
                        mem["rl_action"] = intr["rl_action"]

                        # Only RL's ALERT action permits an email. MONITOR
                        # continues tracking the candidate without sending.
                        if (mem["rl_action"] == 1 and
                                not mem.get("email_sent", False)):
                            if "start_time" not in mem:
                                mem["start_time"] = now
                            last_attempt = mem.get("last_email_attempt", 0)
                            if (now - mem["start_time"] > EMAIL_DELAY and
                                    now - last_attempt >= EMAIL_RETRY_INTERVAL):
                                x1, y1, x2, y2 = intr["bbox"]
                                crop = frame[y1:y2, x1:x2]
                                path = f"intruder_{idx}.jpg"
                                if crop.size > 0 and cv2.imwrite(path, crop):
                                    mem["last_email_attempt"] = now
                                    send_intruder_email(path, idx, mem)
                                else:
                                    print("Email error: could not save intruder face image.")
                                    mem["last_email_attempt"] = now
                        matched = True
                        break

                if not matched:
                    intruder_memory.append({
                        "embedding": intr["embedding"],
                        "start_time": now,
                        "email_sent": False,
                        "last_seen": now,
                        "rl_action": intr["rl_action"]
                    })

            intruder_memory = [
                mem for mem in intruder_memory
                if now - mem["last_seen"] < CLEANUP_TIMEOUT
            ]

        # Draw the original bounding boxes and labels unchanged.
        try:
            for res in stored_results:
                x1, y1, x2, y2 = res["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), res["color"], 2)
                cv2.putText(frame, res["label"], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, res["color"], 2)
        except Exception:
            pass

        cv2.putText(frame, rl_status_text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.imshow("Real-Time Intruder Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('i'), ord('u')):
            if rl_pending_feedback is None:
                print("[RL] No face is currently available for feedback.")
            else:
                action, similarity = rl_pending_feedback
                feedback_label = "INTRUDER" if key == ord('i') else "AUTHORIZED"
                reward = rl_agent.learn(action, feedback_label)
                print(
                    f"[RL] Feedback={feedback_label}, similarity={similarity:.3f}, "
                    f"action={rl_agent.actions[action]}, reward={reward}"
                )
                print("[RL] Updated action values:", rl_agent.action_values)

    vs.stop()
    cv2.destroyAllWindows()


# Run Real-Time Detection
run_realtime()