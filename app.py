import cv2
import sqlite3
import datetime
import time
import threading
import os
import queue
import pygame
import torch
from flask import Flask, render_template, Response, redirect, url_for, request, session, jsonify
from ultralytics import YOLO

MODEL_PATH_FIRE = "/home/fypmachine/SAD3/models/fire.pt"
MODEL_PATH_WEAPON = "/home/fypmachine/SAD3/models/weapon.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_fire = YOLO(MODEL_PATH_FIRE).to(device)
model_weapon = YOLO(MODEL_PATH_WEAPON).to(device)

app = Flask(__name__, static_url_path='/static')
app.secret_key = "your_secret_key"

db_path = "/home/fypmachine/SAD3/vista_database/vista.db"

is_running = False
frame_queue = queue.Queue(maxsize=10) 
lock = threading.Lock()
droidcam_url = "http://192.168.18.6:4747/video"
alert_active = False
last_acknowledged = 0
ALERT_DELAY = 600  
ALERT_SOUND_PATH = "/home/fypmachine/SAD3/sound/alert.mp3"


pygame.mixer.init()

def get_db_connection():
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database Connection Error: {e}")
        return None

def capture_frames():
    """ Continuously captures frames from DroidCam and places them in a queue """
    global is_running, frame_queue
    cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)  
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
    cap.set(cv2.CAP_PROP_FPS, 60) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while is_running:
        success, frame = cap.read()
        if success:
            if frame_queue.full():
                frame_queue.get()  
            frame_queue.put(frame)  
    cap.release()

def process_frames(username):
    """ Runs YOLO inference in a separate thread to avoid blocking Flask """
    global alert_active, last_acknowledged, detected_object
    last_detection = None  

    while is_running:
        if frame_queue.empty():
            time.sleep(0.01) 

        frame = frame_queue.get()  
        results_fire = model_fire(frame, device=device)[0]
        results_weapon = model_weapon(frame, device=device)[0]

        detection, detection_type = "No", "None"
        detected_object = None 

        for results, model_name in zip([results_fire, results_weapon], ["Fire", "Weapon"]):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = model_fire.names[int(box.cls[0])] if model_name == "Fire" else model_weapon.names[int(box.cls[0])]

                if conf > 0.40:
                    detection = "Yes"
                    detection_type = cls.capitalize()
                    detected_object = model_name  

                    label = f"{cls}: {conf:.2f}"

                    
                    box_color = (0, 0, 255) if model_name == "Fire" else (0, 255, 0)

                   
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                   
                    if not alert_active and (time.time() - last_acknowledged) > ALERT_DELAY:
                        alert_active = True
                        threading.Thread(target=play_alert_sound, daemon=True).start()

       
        if detected_object:
            log_detection(username, detection, detected_object)

        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

def log_detection(username, detection, detection_type):
    """ Inserts detection logs into the database only when new detection occurs """
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO logs (username, timestamp, detection, detection_type) VALUES (?, ?, ?, ?)",
                      (username, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), detection, detection_type))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database Logging Error: {e}")

def play_alert_sound():
    """ Plays alert sound only when required """
    if not os.path.exists(ALERT_SOUND_PATH):
        print(f"?? Sound file not found: {ALERT_SOUND_PATH}")
        return
    pygame.mixer.music.load(ALERT_SOUND_PATH)
    pygame.mixer.music.play() 

@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return Response(process_frames(session['username']), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_detection():
    global is_running
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if not is_running:
        is_running = True
        threading.Thread(target=capture_frames, daemon=True).start()
        print("Camera started successfully.")
    return redirect(url_for('index'))

@app.route('/stop')
def stop_detection():
    global is_running
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if is_running:
        is_running = False
        print("Camera stopped.")
    return redirect(url_for('index'))


@app.route('/alert_status')
def alert_status():
    """Returns current alert status and detected anomaly"""
    global detected_object 
    if alert_active and detected_object:
        return jsonify({"is_alert_active": True, "anomaly": detected_object})
    else:
        return jsonify({"is_alert_active": False, "anomaly": "None"})


@app.route('/acknowledge_alert', methods=['POST'])
def acknowledge_alert():
    global alert_active, last_acknowledged
    pygame.mixer.music.stop()
    alert_active = False
    last_acknowledged = time.time()
    print("? Alert acknowledged. Sound stopped.")
    return jsonify({"message": "Alert acknowledged and sound stopped."})

@app.route('/')
def login():
    if 'logged_in' in session:
        return redirect(url_for('index') if session.get('role') == "user" else url_for('admin_dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    email = request.form['email']
    password = request.form['password']

    print(f"Attempting login with Email: {email}, Password: {password}")

    if email == "admin@admin.com" and password == "admin":
        session['logged_in'] = True
        session['username'] = "admin"
        session['role'] = "admin"
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    if conn is None:
        return render_template('login.html', error="Database connection failed.")

    c = conn.cursor()

    try:
        c.execute("SELECT username, email, password FROM authentication WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if user:
            db_username = user['username'].strip()  
            db_email = user['email']
            db_password = user['password']

            print(f"Found User: Username={db_username}, Email={db_email}, Stored Password={db_password}")

            if db_email.lower() == email.lower() and db_password == password:
                session['logged_in'] = True
                session['username'] = db_username  
                session['role'] = "user"
                print(f"Session Data: {session}")
                return redirect(url_for('index'))
            else:
                print("Incorrect password entered.")

        else:
            print("No user found with that email.")

    except Exception as e:
        print(f"Database Query Error: {e}")
        return render_template('login.html', error="Database query failed.")

    return render_template('login.html', error="Invalid Credentials")


@app.route('/index')
def index():
    if not session.get('logged_in') or session.get('role') != "user":
        return redirect(url_for('login'))
    
    username = session['username']
    print(f"Fetching recent logs for: {username}") 

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT timestamp, detection, detection_type FROM logs WHERE username = ? AND detection = 'Yes' ORDER BY timestamp DESC LIMIT 4", (username,))
    logs = c.fetchall()
    conn.close()

    print(f"Recent Logs Retrieved: {logs}") 

    return render_template('index.html', username=username, logs=logs)

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('logged_in') or session.get('role') != "admin":
        return redirect(url_for('login'))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT username, email FROM authentication")
    users = c.fetchall()
    conn.close()

    return render_template('admin.html', users=users)


@app.route('/add_user', methods=['POST'])
def add_user():
    if not session.get('logged_in') or session.get('role') != "admin":
        return redirect(url_for('login'))
    
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if not username or not email or not password:
        return "All fields are required!", 400

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO authentication (username, email, password) VALUES (?, ?, ?)", 
                  (username, email, password))  
        conn.commit()
        conn.close()
        print(f"? User added: Username={username}, Email={email}")
        return "User added successfully!"

    except sqlite3.IntegrityError:
        print("? Error: Username or Email already exists.")
        return "Username or Email already exists!", 400

    except Exception as e:
        print(f"? Database Insertion Error: {e}")
        return f"Error: {str(e)}", 500


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/logs')
def view_logs():
    if not session.get('logged_in') or session.get('role') != "user":
        return redirect(url_for('login'))

    username = session['username']
    print(f"Fetching logs for user: {username}") 

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM logs WHERE username = ? AND detection = 'Yes' ORDER BY timestamp DESC", (username,))
    logs = c.fetchall()
    conn.close()

    print(f"Logs Fetched: {logs}")  

    return render_template('logs.html', logs=logs)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)