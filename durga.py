import cv2
import numpy as np
import turtle

# 1. Load exact image
image_path = "durga.jpg"
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    print(f"Error: '{image_path}' file nahi mili! Folder check karein.")
    exit()

# 2. Window scaling
target_width = 720
aspect_ratio = img_bgr.shape[0] / img_bgr.shape[1]
target_height = int(target_width * aspect_ratio)
resized = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)

split_y = int(target_height * 0.26)

# A) Top Shubh Navratri Text Detection (Orange)
hsv = cv2.cvtColor(resized[:split_y, :], cv2.COLOR_BGR2HSV)
lower_orange = np.array([3, 70, 70])
upper_orange = np.array([25, 255, 255])
text_mask = cv2.inRange(hsv, lower_orange, upper_orange)

# B) Main Silhouette Figure (Dark body)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
smoothed_gray = cv2.medianBlur(gray, 3)
_, fig_mask = cv2.threshold(smoothed_gray, 135, 255, cv2.THRESH_BINARY_INV)
fig_mask[:split_y, :] = 0  # Separate text area

# RETR_TREE extracts both outer solid body AND inner white cutout pockets
fig_cnts, fig_hier = cv2.findContours(fig_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
text_cnts, text_hier = cv2.findContours(text_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

if fig_hier is None or len(fig_cnts) == 0:
    print("Error: Shapes detect nahi hui.")
    exit()
fig_hier = fig_hier[0]

# 3. Setup Turtle Window
screen = turtle.Screen()
screen.setup(width=target_width + 80, height=target_height + 80)
screen.title("Live Sketch - Maa Durga Mahishasura Mardini 🔱")
screen.bgcolor("white")
screen.tracer(1)

pen = turtle.Turtle()
pen.speed(0)
pen.shape("classic")
pen.showturtle()

half_w = target_width / 2
half_h = target_height / 2

def to_turtle(x, y):
    return (x - half_w, half_h - y)

def get_depth(hier, idx):
    depth = 0
    parent = hier[idx][3]
    while parent != -1:
        depth += 1
        parent = hier[parent][3]
    return depth

# Draw and fill shapes with cursor
def render_shape(cnt, fill_color, border_color, pen_thickness=1.5):
    first_pt = cnt[0][0]
    tx, ty = to_turtle(first_pt[0], first_pt[1])
    
    pen.pensize(pen_thickness)
    pen.penup()
    pen.goto(tx, ty)
    pen.pendown()
    
    pen.color(border_color, fill_color)
    pen.begin_fill()
    
    step = 2 if len(cnt) > 70 else 1
    for i in range(1, len(cnt), step):
        pt = cnt[i][0]
        pen.goto(to_turtle(pt[0], pt[1]))
        
    pen.goto(to_turtle(first_pt[0], first_pt[1]))
    pen.end_fill()

# Grouping: Solid Black Base vs Inner White Cutout Pockets
black_shapes = []
white_cutouts = []

for i, cnt in enumerate(fig_cnts):
    if cv2.contourArea(cnt) < 12:
        continue
    d = get_depth(fig_hier, i)
    if d % 2 == 0:
        black_shapes.append((i, cnt))
    else:
        white_cutouts.append((i, cnt))

# Drawing Priority for Black Shapes: Head -> Lion -> Arms & Trishul -> Mahishasur
def figure_priority(item):
    idx, cnt = item
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2
    cy = y + h / 2
    
    if cy < (target_height * 0.48) and (target_width * 0.38 < cx < target_width * 0.62):
        return (1, cy)
    elif cx < (target_width * 0.42):
        return (2, -w * h)
    elif cy < (target_height * 0.75) and cx < (target_width * 0.70):
        return (3, cy)
    else:
        return (4, cx)

black_shapes.sort(key=figure_priority)
# White cutouts sorted top to bottom
white_cutouts.sort(key=lambda item: cv2.boundingRect(item[1])[1])

fill_dark = "#1c1410"      # Solid Dark Black/Brown
festive_orange = "#E8590C" # Vibrant Orange for Text

print("Live drawing started...")

# Phase 1: Draw the full outer shape with black pen & fill solid black
for idx, cnt in black_shapes:
    render_shape(cnt, fill_dark, fill_dark, pen_thickness=1.5)

# Phase 2: White pen draws inside shapes & fills solid WHITE (carving arms, lion & body)
print("White pen drawing internal shapes and filling white...")
for idx, cnt in white_cutouts:
    render_shape(cnt, "white", "white", pen_thickness=1.4)

# Phase 3: Draw "Shubh Navratri" in Festive Orange at the top
print("Drawing Shubh Navratri Text in Orange...")
if text_hier is not None and len(text_cnts) > 0:
    t_hier = text_hier[0]
    sorted_text = sorted(enumerate(text_cnts), key=lambda x: cv2.boundingRect(x[1])[1])
    for idx, cnt in sorted_text:
        if cv2.contourArea(cnt) < 5:
            continue
        d = get_depth(t_hier, idx)
        f_color = festive_orange if (d % 2 == 0) else "white"
        render_shape(cnt, f_color, festive_orange, pen_thickness=1.2)

# Finish
pen.hideturtle()
print("Drawing completed with perfection! Ready for screen record.")
screen.mainloop()