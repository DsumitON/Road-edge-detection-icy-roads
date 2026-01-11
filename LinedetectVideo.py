
import cv2
import numpy as np
import math

# ================= VIDEO INPUT =================
video_path = "/home/sksuser/Downloads/pori_snow_road.MOV"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

h, w = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120:
    fps =30.0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ================= VIDEO OUTPUT =================
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("lane_detection_output.avi", fourcc, fps, (w, h))
if not out.isOpened():
    raise RuntimeError("VideoWriter failed")

# ================= DISPLAY =================
cv2.namedWindow("Snow Lane Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Snow Lane Detection", 960, 540)

# ================= PARAMETERS (RELAXED & WORKING) =================
ALPHA = 0.25
DISPLAY_SCALE = 0.5
MIN_ANGLE = 15
MAX_ANGLE = 80

prev_left = None
prev_right = None
prev_center = None

# ================= HELPERS =================
def smooth(prev, curr, alpha):
    if curr is None:
        return prev
    if prev is None:
        return curr
    return tuple(alpha * c + (1 - alpha) * p for p, c in zip(prev, curr))

def fit(points):
    if len(points) < 6:
        return None
    pts = np.array(points)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx), float(vy), float(x0), float(y0)

def draw_lane(img, line, color, y_bottom, y_top):
    if line is None:
        return None
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-3:
        return None
    x1 = int(x0 + (y_bottom - y0) * vx / vy)
    x2 = int(x0 + (y_top - y0) * vx / vy)
    cv2.line(img, (x1, y_bottom), (x2, y_top), color, 4)
    return (x1, y_bottom, x2, y_top)

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    ROAD_TOP = int(h * 0.75)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 80)

    mask = np.zeros_like(edges)
    roi = np.array([[
        (int(w * 0.15), h),
        (int(w * 0.45), ROAD_TOP),
        (int(w * 0.55), ROAD_TOP),
        (int(w * 0.85), h)
    ]], np.int32)

    cv2.fillPoly(mask, roi, 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=50,
        maxLineGap=60
    )

    left_pts, right_pts = [], []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            angle = abs(math.degrees(math.atan(slope)))
            if angle < MIN_ANGLE or angle > MAX_ANGLE:
                continue

            if slope < 0 and x1 < w * 0.55:
                left_pts += [(x1, y1), (x2, y2)]
            elif slope > 0 and x1 > w * 0.45:
                right_pts += [(x1, y1), (x2, y2)]

    raw_left = fit(left_pts)
    raw_right = fit(right_pts)

    left = smooth(prev_left, raw_left, ALPHA)
    right = smooth(prev_right, raw_right, ALPHA)
    prev_left, prev_right = left, right

    left_draw = draw_lane(frame, left, (0, 255, 0), h, ROAD_TOP)
    right_draw = draw_lane(frame, right, (0, 255, 0), h, ROAD_TOP)

    if left_draw and right_draw:
        lx1, ly1, lx2, ly2 = left_draw
        rx1, ry1, rx2, ry2 = right_draw

        center = (
            (lx1 + rx1) / 2,
            (ly1 + ry1) / 2,
            (lx2 + rx2) / 2,
            (ly2 + ry2) / 2
        )

        center = smooth(prev_center, center, ALPHA)
        prev_center = center

        cx1, cy1, cx2, cy2 = map(int, center)
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 5)

    display = cv2.resize(frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
    cv2.imshow("Snow Lane Detection", display)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
























































'''import cv2
import numpy as np
import math

# ================= VIDEO INPUT =================
video_path = "/home/sksuser/Downloads/pori_snow_road.MP4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

h, w = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120:
    fps = 25.0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ================= VIDEO OUTPUT =================
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("lane_detection_hybrid.avi", fourcc, fps, (w, h))
if not out.isOpened():
    raise RuntimeError("VideoWriter failed")

# ================= DISPLAY =================
cv2.namedWindow("Hybrid Lane Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hybrid Lane Detection", 960, 540)

# ================= PARAMETERS =================
DISPLAY_SCALE = 0.5
ALPHA = 0.25
STRAIGHT_SLOPE_VAR_THRESH = 0.15   # lower = stricter straightness

prev_left = None
prev_right = None
prev_left_poly = None
prev_right_poly = None

# ================= HELPERS =================
def smooth(prev, curr, alpha):
    if curr is None:
        return prev
    if prev is None:
        return curr
    return tuple(alpha * c + (1 - alpha) * p for p, c in zip(prev, curr))

def smooth_poly(prev, curr, alpha):
    if curr is None:
        return prev
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev

def fit_line(points):
    if len(points) < 6:
        return None
    pts = np.array(points)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx), float(vy), float(x0), float(y0)

def draw_line(img, line, color, y_bottom, y_top):
    if line is None:
        return None
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-3:
        return None
    x1 = int(x0 + (y_bottom - y0) * vx / vy)
    x2 = int(x0 + (y_top - y0) * vx / vy)
    cv2.line(img, (x1, y_bottom), (x2, y_top), color, 4)
    return (x1, y_bottom, x2, y_top)

def draw_poly(img, poly, color, y_bottom, y_top):
    if poly is None:
        return
    ys = np.linspace(y_top, y_bottom, 60)
    xs = poly[0] * ys**2 + poly[1] * ys + poly[2]
    pts = np.vstack((xs, ys)).T.astype(np.int32)
    cv2.polylines(img, [pts], False, color, 4)

def is_straight(y, x):
    if len(y) < 20:
        return False
    slopes = np.diff(x) / (np.diff(y) + 1e-6)
    return np.var(slopes) < STRAIGHT_SLOPE_VAR_THRESH

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ROAD_TOP = int(h * 0.75)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    mask = np.zeros_like(edges)
    roi = np.array([[
        (int(w * 0.15), h),
        (int(w * 0.45), ROAD_TOP),
        (int(w * 0.55), ROAD_TOP),
        (int(w * 0.85), h)
    ]], np.int32)

    cv2.fillPoly(mask, roi, 255)
    edges = cv2.bitwise_and(edges, mask)

    ys, xs = np.where(edges > 0)

    left_x, left_y = [], []
    right_x, right_y = [], []

    for x, y in zip(xs, ys):
        if x < w * 0.5:
            left_x.append(x)
            left_y.append(y)
        else:
            right_x.append(x)
            right_y.append(y)

    # ---------- LEFT LANE ----------
    left_line = None
    left_poly = None
    if len(left_x) > 300:
        if is_straight(left_y, left_x):
            pts = list(zip(left_x, left_y))
            left_line = smooth(prev_left, fit_line(pts), ALPHA)
        else:
            left_poly = smooth_poly(prev_left_poly, np.polyfit(left_y, left_x, 2), ALPHA)

    # ---------- RIGHT LANE ----------
    right_line = None
    right_poly = None
    if len(right_x) > 300:
        if is_straight(right_y, right_x):
            pts = list(zip(right_x, right_y))
            right_line = smooth(prev_right, fit_line(pts), ALPHA)
        else:
            right_poly = smooth_poly(prev_right_poly, np.polyfit(right_y, right_x, 2), ALPHA)

    prev_left = left_line
    prev_right = right_line
    prev_left_poly = left_poly
    prev_right_poly = right_poly

    # ---------- DRAW ----------
    if left_line is not None:
        draw_line(frame, left_line, (0, 255, 0), h, ROAD_TOP)
    else:
        draw_poly(frame, left_poly, (0, 255, 0), h, ROAD_TOP)

    if right_line is not None:
        draw_line(frame, right_line, (0, 255, 0), h, ROAD_TOP)
    else:
        draw_poly(frame, right_poly, (0, 255, 0), h, ROAD_TOP)

    display = cv2.resize(frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
    cv2.imshow("Hybrid Lane Detection", display)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()






								========================================================================================================

'''



