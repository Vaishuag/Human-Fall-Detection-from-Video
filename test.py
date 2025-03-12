from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import threading
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Replace with your secret key for CSRF protection
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize YOLO model
model = YOLO("fall_det_1.pt")

# Initialize pygame mixer
pygame.mixer.init()

# Load alert voice (replace with your own alert voice file)
alert_voice = pygame.mixer.Sound('./alert.wav')

# Email configuration (replace with your own email credentials)
EMAIL = 'your_email@example.com'  # Replace with sender's email
PASSWORD = 'your_password'  # Replace with sender's email password
TO_EMAIL = 'recipient_email@example.com'  # Replace with recipient's email
SMTP_SERVER = 'smtp.example.com'  # Replace with SMTP server
SMTP_PORT = 587  # Replace with SMTP port

# Global variable for email notification status
email_sent = False

# Function to play alert voice
def play_alert_voice():
    alert_voice.play()

# Function to send email notification
def send_email_notification():
    global email_sent
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = TO_EMAIL
        msg['Subject'] = 'Fall Detected!'
        body = 'A fall has been detected in the video stream.'
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, TO_EMAIL, msg.as_string())
        server.quit()
        print('Email notification sent successfully!')
        email_sent = True  # Set flag to True after sending email
    except Exception as e:
        print(f'Error sending email notification: {str(e)}')

# Function to analyze video in a separate thread
def analyze_video_thread():
    global video_path
    global email_sent

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        flash('Error: Could not open video file')
        return

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, conf=0.5)
            for result in results:
                if result.names[0] == 'Fall-Detected' and not email_sent:  # Only send email if not already sent
                    play_alert_voice()
                    send_email_notification()

            annotated_frame = results[0].plot() if results else frame

            # Encode the frame as jpeg
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()

# Route for home page
@app.route('/about')
def index():
    return render_template('about.html')

@app.route('/')
def ndex():
    return render_template('index.html')

@app.route('/contact')
def inex():
    return render_template('contact.html')

# Route to handle video upload and start analysis
@app.route('/analyze', methods=['POST'])
def analyze_video():
    global video_path
    global email_sent

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Reset email_sent flag
        email_sent = False

        # Start analysis in a separate thread
        thread = threading.Thread(target=analyze_video_thread)
        thread.start()

        return redirect(url_for('video_feed'))

    return redirect(url_for('index'))

# Route to stream analyzed video to the client
@app.route('/video_feed')
def video_feed():
    return Response(analyze_video_thread(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
