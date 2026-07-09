import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

# ── Download model ──────────────────────────────────────────
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Done!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# ── State ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
canvas_strokes = []       # list of 2D polylines
current_stroke = []       # points being drawn right now
mode = "IDLE"

# 3D state
is_3d = False
angle_x = 0.0
angle_y = 0.0
extrude_depth = 80
prev_2f = None            # previous 2-finger midpoint for rotation delta

# ── Helpers ──────────────────────────────────────────────────
def count_fingers(lm):
    tips = [8, 12, 16, 20]
    return sum(1 for t in tips if lm[t].y < lm[t - 2].y)

def get_midpoint(lm, w, h):
    x = int((lm[8].x + lm[12].x) / 2 * w)
    y = int((lm[8].y + lm[12].y) / 2 * h)
    return x, y

def rotate_x(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return pts @ R.T

def rotate_y(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return pts @ R.T

def project(pts3d, cx, cy, fov=400):
    out = []
    for p in pts3d:
        z = p[2] + fov
        if z == 0: z = 0.001
        sx = int(p[0] * fov / z + cx)
        sy = int(p[1] * fov / z + cy)
        out.append((sx, sy))
    return out

def draw_3d(frame, strokes, angle_x, angle_y, depth):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Find bounding center of all strokes
    all_pts = [p for s in strokes for p in s]
    if not all_pts:
        return
    mx = sum(p[0] for p in all_pts) / len(all_pts)
    my = sum(p[1] for p in all_pts) / len(all_pts)

    for stroke in strokes:
        if len(stroke) < 2:
            continue

        # Build front face (centered)
        front = np.array([[p[0]-mx, p[1]-my, 0.0] for p in stroke])
        back  = np.array([[p[0]-mx, p[1]-my, float(depth)] for p in stroke])

        # Rotate both faces
        front = rotate_x(rotate_y(front, angle_y), angle_x)
        back  = rotate_x(rotate_y(back,  angle_y), angle_x)

        # Shade: back face darker
        front_col = (60, 180, 255)
        back_col  = (20,  80, 160)
        side_col  = (40, 130, 200)

        fp = project(front, cx, cy)
        bp = project(back,  cx, cy)

        # Draw side faces (connecting front to back) with fill
        for i in range(len(fp) - 1):
            quad = np.array([fp[i], fp[i+1], bp[i+1], bp[i]], dtype=np.int32)
            cv2.fillPoly(frame, [quad], side_col)
            cv2.polylines(frame, [quad], True, (255,255,255), 1)

        # Draw back face
        cv2.polylines(frame, [np.array(bp)], False, back_col, 2)

        # Draw front face (on top)
        cv2.polylines(frame, [np.array(fp)], False, front_col, 3)

        # Dots on front
        for p in fp:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)

def draw_2d(frame, strokes, current):
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(frame, stroke[i-1], stroke[i], (0, 165, 255), 3)
    for i in range(1, len(current)):
        cv2.line(frame, current[i-1], current[i], (0, 255, 0), 3)

# ── Main loop ────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    fingers = 0
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # Draw hand skeleton
        for l in lm:
            cv2.circle(frame, (int(l.x*w), int(l.y*h)), 4, (255, 0, 255), -1)

        fingers = count_fingers(lm)
        ix = int(lm[8].x * w)
        iy = int(lm[8].y * h)

        # ── 1 FINGER: Draw ──
        if fingers == 1:
            is_3d = False
            prev_2f = None
            mode = "DRAWING"
            current_stroke.append((ix, iy))

        # ── 2 FINGERS: Extrude + Rotate ──
        elif fingers == 2:
            # Save current stroke if switching from draw
            if current_stroke:
                canvas_strokes.append(current_stroke.copy())
                current_stroke.clear()

            mid = get_midpoint(lm, w, h)
            is_3d = True
            mode = "3D ROTATE"

            if prev_2f is not None:
                dx = mid[0] - prev_2f[0]
                dy = mid[1] - prev_2f[1]
                angle_y += dx * 0.01
                angle_x += dy * 0.01
            prev_2f = mid

            # Show 2-finger midpoint
            cv2.circle(frame, mid, 8, (0, 255, 255), -1)

        # ── 3 FINGERS: Clear ──
        elif fingers == 3:
            canvas_strokes.clear()
            current_stroke.clear()
            is_3d = False
            angle_x = 0.0
            angle_y = 0.0
            prev_2f = None
            mode = "CLEARED!"

        else:
            # Save stroke on lift
            if current_stroke:
                canvas_strokes.append(current_stroke.copy())
                current_stroke.clear()
            prev_2f = None
            mode = "IDLE"

    else:
        if current_stroke:
            canvas_strokes.append(current_stroke.copy())
            current_stroke.clear()
        prev_2f = None

    # ── Render ──────────────────────────────────────────────
    if is_3d and canvas_strokes:
        draw_3d(frame, canvas_strokes, angle_x, angle_y, extrude_depth)
    else:
        draw_2d(frame, canvas_strokes, current_stroke)

    # ── UI overlay ──────────────────────────────────────────
    colors = {"DRAWING":"(0,255,0)", "3D ROTATE":"(0,255,255)",
              "CLEARED!":"(0,0,255)", "IDLE":"(200,200,200)"}
    col_map = {"DRAWING":(0,255,0), "3D ROTATE":(0,255,255),
               "CLEARED!":(0,0,255), "IDLE":(200,200,200)}
    cv2.putText(frame, f"MODE: {mode}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_map.get(mode,(255,255,255)), 2)
    cv2.putText(frame, "1=Draw  2=3D Rotate  3=Clear  ESC=Exit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("GestureCAD 3D", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()