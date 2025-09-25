import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
RIGHT_EYEBROW = [296, 334, 293, 300, 276, 283, 295]

# Face outline and mouth landmarks
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MOUTH = [78, 95, 88, 178, 87, 14, 317, 405, 320, 307, 375, 321, 308, 324, 318]

# Data storage
score_history = deque(maxlen=15)  # Increased for smoother averaging
blink_history = deque(maxlen=30)
head_movement_history = deque(maxlen=20)
gaze_history = deque(maxlen=20)
mouth_history = deque(maxlen=20)

distraction = 0
session_stats = {
    'total_blinks': 0,
    'focus_time': 0,
    'distraction_time': 0,
    'head_movement_episodes': 0,
    'yawn_count': 0
}

# Logging setup
log_filename = f"concentration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
detailed_log_filename = f"detailed_concentration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_interval = 15  # Log every 15 frames
frame_count = 0

def init_log_files():
    """Initialize CSV files with comprehensive headers"""
    # Main log file
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'timestamp', 'concentration_score', 'gaze_score', 'head_pose_score', 
            'blink_detected', 'distraction_count', 'avg_ear', 'status',
            'blink_rate_per_min', 'head_movement_intensity', 'gaze_stability',
            'mouth_openness', 'eyebrow_raise', 'face_tilt', 'session_focus_time',
            'session_distraction_time', 'total_session_blinks'
        ])
    
    # Detailed log file
    with open(detailed_log_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'timestamp', 'frame_number', 'left_ear', 'right_ear', 'avg_ear',
            'left_eyebrow_height', 'right_eyebrow_height', 'mouth_aspect_ratio',
            'nose_x', 'nose_y', 'head_tilt_angle', 'left_iris_x', 'right_iris_x',
            'gaze_direction', 'face_area', 'lighting_quality', 'face_symmetry',
            'concentration_raw', 'concentration_smoothed'
        ])

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def eye_aspect_ratio(landmarks, eye_points, image_w, image_h):
    """Calculate Eye Aspect Ratio"""
    points = []
    for idx in eye_points:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        points.append((x, y))

    # Vertical distances
    A = calculate_distance(points[1], points[5])
    B = calculate_distance(points[2], points[4])
    # Horizontal distance
    C = calculate_distance(points[0], points[3])
    
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, image_w, image_h):
    """Calculate Mouth Aspect Ratio to detect yawning"""
    try:
        # Use more reliable mouth landmarks
        top_lip = landmarks[13]      # Upper lip center
        bottom_lip = landmarks[14]   # Lower lip center  
        left_corner = landmarks[61]  # Left mouth corner
        right_corner = landmarks[291] # Right mouth corner
        
        # Convert to pixel coordinates
        top_y = top_lip.y * image_h
        bottom_y = bottom_lip.y * image_h
        left_x = left_corner.x * image_w
        right_x = right_corner.x * image_w
        
        # Calculate mouth opening
        vertical_distance = abs(bottom_y - top_y)
        horizontal_distance = abs(right_x - left_x)
        
        if horizontal_distance == 0:
            return 0
            
        mar = vertical_distance / horizontal_distance
        return mar
    except:
        return 0

def eyebrow_height(landmarks, eyebrow_points, image_w, image_h):
    """Calculate average eyebrow height"""
    points = []
    for idx in eyebrow_points:
        lm = landmarks[idx]
        y = lm.y * image_h
        points.append(y)
    
    return np.mean(points) if points else 0

def get_head_tilt(landmarks, image_w, image_h):
    """Calculate head tilt angle"""
    left_eye_corner = landmarks[33]
    right_eye_corner = landmarks[263]
    
    left_x, left_y = left_eye_corner.x * image_w, left_eye_corner.y * image_h
    right_x, right_y = right_eye_corner.x * image_w, right_eye_corner.y * image_h
    
    angle = np.degrees(np.arctan2(right_y - left_y, right_x - left_x))
    return abs(angle)

def get_gaze_direction(landmarks, image_w, image_h):
    """Enhanced gaze direction analysis"""
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    
    # Get eye corners for reference
    left_corner = landmarks[33]
    right_corner = landmarks[263]
    
    # Calculate relative iris position
    left_iris_x = left_iris.x
    right_iris_x = right_iris.x
    
    avg_iris_x = (left_iris_x + right_iris_x) / 2.0
    
    # Determine gaze direction
    if avg_iris_x < 0.45:
        return "LEFT"
    elif avg_iris_x > 0.55:
        return "RIGHT" 
    else:
        return "CENTER"

def calculate_face_area(landmarks, image_w, image_h):
    """Calculate face area to detect distance from camera"""
    face_points = [landmarks[i] for i in FACE_OVAL[:8]]  # Use subset for efficiency
    
    coords = [(int(lm.x * image_w), int(lm.y * image_h)) for lm in face_points]
    
    if len(coords) >= 4:
        # Simple bounding box area
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return width * height
    return 0

def estimate_lighting_quality(frame):
    """Estimate lighting quality of the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    # Lighting score (0-1, higher is better)
    lighting_score = min(1.0, (mean_brightness / 255.0) * (contrast / 100.0))
    return lighting_score

def calculate_face_symmetry(landmarks, image_w, image_h):
    """Calculate face symmetry score"""
    # Use key facial points
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    nose_x = nose_tip.x * image_w
    left_eye_x = left_eye.x * image_w
    right_eye_x = right_eye.x * image_w
    
    # Calculate distances from nose to each eye
    left_distance = abs(nose_x - left_eye_x)
    right_distance = abs(right_eye_x - nose_x)
    
    if left_distance + right_distance == 0:
        return 1.0
    
    # Symmetry score (closer to 1 is more symmetric)
    symmetry = 1.0 - abs(left_distance - right_distance) / (left_distance + right_distance)
    return max(0, symmetry)

def is_blinking(ear, threshold=0.2):
    return ear < threshold

def is_yawning(mar, threshold=0.6):
    return mar > threshold

def get_head_pose_score(landmarks, image_w, image_h):
    """Enhanced head pose scoring"""
    nose = landmarks[1]
    x = nose.x * image_w
    y = nose.y * image_h
    
    center_x, center_y = image_w / 2, image_h / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # More lenient distance threshold
    max_distance = min(image_w, image_h) * 0.25  # Reduced from 0.4
    normalized_distance = min(1.0, distance / max_distance)
    
    # Score decreases with distance from center
    score = max(0, 1.0 - normalized_distance)
    return score

def get_enhanced_gaze_score(landmarks, image_w, image_h):
    """Enhanced gaze scoring with stability"""
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    
    avg_x = (left_iris.x + right_iris.x) / 2.0
    avg_y = (left_iris.y + right_iris.y) / 2.0
    
    # More lenient scoring - wider acceptable range
    x_score = max(0, 1.0 - abs(avg_x - 0.5) * 1.5)  # Reduced penalty
    y_score = max(0, 1.0 - abs(avg_y - 0.45) * 1.2)  # Slightly lower center, less penalty
    
    gaze_score = max(0, (x_score + y_score) / 2)
    
    # Add to history for stability calculation
    gaze_history.append((avg_x, avg_y))
    
    return gaze_score

def calculate_gaze_stability():
    """Calculate gaze stability over recent frames"""
    if len(gaze_history) < 5:
        return 1.0
    
    positions = list(gaze_history)[-10:]  # Last 10 positions
    
    # Calculate variance in gaze positions
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]
    
    x_variance = np.var(x_positions)
    y_variance = np.var(y_positions)
    
    # Stability score (lower variance = higher stability)
    stability = 1.0 / (1.0 + (x_variance + y_variance) * 10)
    return min(1.0, stability)

def compute_enhanced_concentration_score(gaze, head_pose, blink, mouth_open, eyebrow_activity, stability):
    """Enhanced concentration scoring with more factors"""
    # Base score with adjusted weights for more generous scoring
    base_score = 0.35 * gaze + 0.3 * head_pose + 0.1 * (0 if blink else 1)
    base_score += 0.05 * (0 if mouth_open else 1)  # Reduced yawn penalty
    base_score += 0.05 * max(0, 1 - eyebrow_activity * 5)  # Reduced eyebrow penalty
    base_score += 0.15 * stability                 # Increased stability reward
    
    # Add a baseline score to make it easier to achieve higher scores
    base_score = min(1.0, base_score + 0.2)  # Add 20% baseline
    
    return round(base_score * 100, 2)

def calculate_blink_rate():
    """Calculate blinks per minute"""
    if len(blink_history) < 2:
        return 0
    
    recent_blinks = sum(blink_history)
    time_window = len(blink_history) / 30.0  # Assuming 30 FPS
    blinks_per_minute = (recent_blinks / time_window) * 60 if time_window > 0 else 0
    
    return round(blinks_per_minute, 1)

def log_main_data(timestamp, concentration, gaze, head_pose, blink, distraction_count, avg_ear, 
                 blink_rate, head_movement, gaze_stability, mouth_openness, eyebrow_raise, 
                 face_tilt, focus_time, distraction_time, total_blinks):
    """Log main concentration data"""
    status = "DISTRACTED" if distraction_count > 0 else "FOCUSED"
    
    with open(log_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            timestamp, concentration, gaze, head_pose, blink, distraction_count, 
            round(avg_ear, 4), status, blink_rate, round(head_movement, 3), 
            round(gaze_stability, 3), round(mouth_openness, 3), round(eyebrow_raise, 3),
            round(face_tilt, 2), focus_time, distraction_time, total_blinks
        ])

def log_detailed_data(timestamp, frame_num, left_ear, right_ear, avg_ear, left_eyebrow, 
                     right_eyebrow, mar, nose_x, nose_y, tilt_angle, left_iris_x, 
                     right_iris_x, gaze_dir, face_area, lighting, symmetry, 
                     concentration_raw, concentration_smoothed):
    """Log detailed frame-by-frame data"""
    with open(detailed_log_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            timestamp, frame_num, round(left_ear, 4), round(right_ear, 4), 
            round(avg_ear, 4), round(left_eyebrow, 2), round(right_eyebrow, 2),
            round(mar, 4), round(nose_x, 2), round(nose_y, 2), round(tilt_angle, 2),
            round(left_iris_x, 4), round(right_iris_x, 4), gaze_dir, int(face_area),
            round(lighting, 3), round(symmetry, 3), round(concentration_raw, 2),
            round(concentration_smoothed, 2)
        ])

def enhanced_bar(score, frame, additional_info):
    """Enhanced visual bar with additional metrics"""
    bar_width = 250
    bar_height = 25
    bar_x = 30
    bar_y = 100
    
    # Main concentration bar
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (50, 50, 50), -1)
    
    fill_width = int(score * bar_width / 100)
    if score > 70:
        color = (0, 255, 0)  # Green for high concentration
    elif score > 40:
        color = (0, 255, 255)  # Yellow for medium concentration
    else:
        color = (0, 100, 255)  # Red for low concentration
        
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + fill_width, bar_y + bar_height), 
                 color, -1)
 
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (200, 200, 200), 2)
    
    cv2.putText(frame, f"{score}%", 
               (bar_x + bar_width + 10, bar_y + bar_height//2 + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Additional metrics
    y_offset = 140
    metrics = [
        f"Blink Rate: {additional_info.get('blink_rate', 0)}/min",
        f"Gaze: {additional_info.get('gaze_direction', 'CENTER')}",
        f"Stability: {additional_info.get('gaze_stability', 0):.2f}",
        f"Head Tilt: {additional_info.get('head_tilt', 0):.1f}°"
    ]
    
    for i, metric in enumerate(metrics):
        cv2.putText(frame, metric, (bar_x, y_offset + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

# Initialize logging
init_log_files()
print(f"Logging main data to: {log_filename}")
print(f"Logging detailed data to: {detailed_log_filename}")

cap = cv2.VideoCapture(0)
start_time = time.time()
last_blink_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # UI background
    ui_bg = frame.copy()
    cv2.rectangle(ui_bg, (0, 0), (frame.shape[1], 280), (30, 30, 30), -1)
    cv2.addWeighted(ui_bg, 0.7, frame, 0.3, 0, frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape
    results = face_mesh.process(frame_rgb)

    # Default values
    concentration_raw = 0
    additional_metrics = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate all metrics
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h)
            avg_ear = (left_ear + right_ear) / 2
            
            left_eyebrow_h = eyebrow_height(landmarks, LEFT_EYEBROW, image_w, image_h)
            right_eyebrow_h = eyebrow_height(landmarks, RIGHT_EYEBROW, image_w, image_h)
            eyebrow_activity = abs(left_eyebrow_h - right_eyebrow_h) / image_h
            
            mar = mouth_aspect_ratio(landmarks, image_w, image_h)
            head_tilt = get_head_tilt(landmarks, image_w, image_h)
            gaze_direction = get_gaze_direction(landmarks, image_w, image_h)
            face_area = calculate_face_area(landmarks, image_w, image_h)
            lighting_quality = estimate_lighting_quality(frame)
            face_symmetry = calculate_face_symmetry(landmarks, image_w, image_h)
            
            # Enhanced scores
            blink = is_blinking(avg_ear)
            yawning = is_yawning(mar)
            gaze_score = get_enhanced_gaze_score(landmarks, image_w, image_h)
            head_score = get_head_pose_score(landmarks, image_w, image_h)
            gaze_stability = calculate_gaze_stability()
            
            # Update histories
            blink_history.append(1 if blink else 0)
            mouth_history.append(mar)
            
            # Calculate blink rate
            blink_rate = calculate_blink_rate()
            
            # Track session statistics with better blink detection
            if blink and (current_time - last_blink_time) > 0.3:  # Reduced from 0.5
                # Additional check: ensure EAR is significantly low
                if avg_ear < 0.15:  # Very low threshold for actual blinks
                    session_stats['total_blinks'] += 1
                    last_blink_time = current_time
            
            if yawning and mar > 1.0:  # Additional check for yawning
                session_stats['yawn_count'] += 1
            
            # Compute concentration
            concentration_raw = compute_enhanced_concentration_score(
                gaze_score, head_score, blink, yawning, eyebrow_activity, gaze_stability
            )
            
            score_history.append(concentration_raw)
            smooth_score = int(np.mean(score_history))
            
            # Update session stats
            if smooth_score >= 40:
                session_stats['focus_time'] += 1
                if distraction > 0:
                    distraction = 0
            else:
                session_stats['distraction_time'] += 1
                distraction += 1
                if distraction > 1000:
                    distraction = 0
                    print('System shutdown due to extended distraction')
            
            # Prepare additional metrics for display
            additional_metrics = {
                'blink_rate': blink_rate,
                'gaze_direction': gaze_direction,
                'gaze_stability': gaze_stability,
                'head_tilt': head_tilt,
                'face_area': face_area,
                'lighting': lighting_quality,
                'symmetry': face_symmetry
            }
            
            # Enhanced visualization
            enhanced_bar(smooth_score, frame, additional_metrics)
            
            # Display main metrics
            cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if blink:
                cv2.putText(frame, "BLINKING", (30, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
            
            if yawning and mar > 1.0:  # Only show if really yawning
                cv2.putText(frame, "YAWNING", (150, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if smooth_score < 40:
                cv2.putText(frame, f"Distraction: {distraction}", (30, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            
            # Log data at intervals
            if frame_count % log_interval == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Main log
                log_main_data(
                    timestamp, smooth_score, gaze_score, head_score, blink, 
                    distraction, avg_ear, blink_rate, head_tilt, gaze_stability,
                    mar, eyebrow_activity, head_tilt, session_stats['focus_time'],
                    session_stats['distraction_time'], session_stats['total_blinks']
                )
                
                # Detailed log
                log_detailed_data(
                    timestamp, frame_count, left_ear, right_ear, avg_ear,
                    left_eyebrow_h, right_eyebrow_h, mar, 
                    landmarks[1].x * image_w, landmarks[1].y * image_h, head_tilt,
                    landmarks[468].x, landmarks[473].x, gaze_direction, face_area,
                    lighting_quality, face_symmetry, concentration_raw, smooth_score
                )
                
                print(f"Frame {frame_count}: Concentration {smooth_score}%, Blinks: {session_stats['total_blinks']}")
    
    # Status indicators
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (image_w - 120, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    
    status_color = (0, 255, 0) if distraction == 0 else (0, 100, 255)
    cv2.circle(frame, (image_w - 30, 60), 12, status_color, -1)
    
    # Session statistics
    session_duration = current_time - start_time
    cv2.putText(frame, f"Session: {session_duration/60:.1f}min", (image_w - 150, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, f"Blinks: {session_stats['total_blinks']}", (image_w - 150, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Advanced Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final session summary
session_duration = time.time() - start_time
focus_percentage = (session_stats['focus_time'] / max(1, session_stats['focus_time'] + session_stats['distraction_time'])) * 100

print(f"\n=== SESSION SUMMARY ===")
print(f"Duration: {session_duration/60:.1f} minutes")
print(f"Frames processed: {frame_count}")
print(f"Focus time: {session_stats['focus_time']} frames ({focus_percentage:.1f}%)")
print(f"Total blinks: {session_stats['total_blinks']}")
print(f"Average blink rate: {session_stats['total_blinks']/(session_duration/60):.1f} per minute")
print(f"Yawn count: {session_stats['yawn_count']}")
print(f"Main log: {log_filename}")
print(f"Detailed log: {detailed_log_filename}")

cap.release()
cv2.destroyAllWindows()