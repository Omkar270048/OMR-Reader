import cv2
import numpy as np
import pandas as pd
from collections import Counter
answer = {}
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    edges = cv2.Canny(blurred, 30, 45)
    return edges

def find_rectangles(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            rectangles.append((approx, area))
    rectangles.sort(key=lambda x: x[1], reverse=True)
    return rectangles

def draw_rectangles(image, rectangles):
    if rectangles:
        cv2.drawContours(image, [rectangles[0][0]], -1, (0, 255, 0), 2)
        if len(rectangles) > 1:
            cv2.drawContours(image, [rectangles[1][0]], -1, (0, 0, 255), 2)
        if len(rectangles) > 2:
            blue_contour = rectangles[2][0]
            cv2.drawContours(image, [blue_contour], -1, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(blue_contour)
            text = f"X: {x}, Y: {y}, W: {w}, H: {h}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return blue_contour, (x, y, w, h)
    return None, None

def divide_contour_into_columns(bounding_box, num_columns=5):
    x, y, w, h = bounding_box
    column_width = w // num_columns
    columns = []

    for i in range(num_columns):
        roi_left = x + i * column_width
        roi_right = x + (i + 1) * column_width if i < num_columns - 1 else x + w
        columns.append((roi_left, roi_right, y, y + h))

    return columns

def detect_circles_in_region(image, roi_left, roi_right, roi_top, roi_bottom):
    roi_image = image[roi_top:roi_bottom, roi_left:roi_right]
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=20
    )

    circle_data = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0] + roi_left, circle[1] + roi_top)
            radius = circle[2]

            mask = np.zeros_like(gray)
            cv2.circle(mask, (circle[0], circle[1]), radius, 255, -1)
            masked_image = cv2.bitwise_and(roi_image, roi_image, mask=mask)

            masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            circle_area = np.pi * (radius ** 2)
            filled_area = cv2.countNonZero(masked_gray)
            area_threshold = circle_area * 0.5

            if filled_area > area_threshold:
                circle_type = 'Ring-like'
                cv2.circle(image, center, radius, (0, 255, 0), 2)
                cv2.circle(image, center, 2, (0, 255, 0), 3)
            else:
                circle_type = 'Filled'
                cv2.circle(image, center, radius, (0, 0, 255), 2)
                cv2.circle(image, center, 2, (0, 0, 255), 3)

            circle_data.append({
                'center_x': center[0],
                'center_y': center[1],
                'radius': radius,
                'type': circle_type
            })

    circle_data.sort(key=lambda x: (x['center_y'], x['center_x']))

    adjusted_circle_data = []
    for i in range(0, len(circle_data), 4):
        group = circle_data[i:i+4]
        if len(group) < 4:
            adjusted_circle_data.extend(group)
            continue

        y_values = [circle['center_y'] for circle in group]
        most_common_y = Counter(y_values).most_common(1)[0][0]

        for circle in group:
            circle['center_y'] = most_common_y
            adjusted_circle_data.append(circle)

    adjusted_circle_data.sort(key=lambda x: (x['center_y'], x['center_x']))

    return adjusted_circle_data

def detect_black_filled_regions(image, roi_left, roi_right, roi_top, roi_bottom):
    roi_image = image[roi_top:roi_bottom, roi_left:roi_right]
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_region_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            filled_region_count += 1
            cv2.drawContours(image, [contour + [roi_left, roi_top]], -1, (0, 255, 255), 2)

    return filled_region_count

def display_circle_table(circles):
    df = pd.DataFrame(circles)
    df.index += 1
    df.index.name = 'Circle No.'
    df.columns = ['Center X', 'Center Y', 'Radius', 'Type']
    print(df.to_string(index=True))

def process_image(image_path):
    global answer
    userAnswer = []
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    edges = preprocess_image(image)
    rectangles = find_rectangles(edges)
    blue_contour, bounding_box = draw_rectangles(image, rectangles)

    if blue_contour is not None and bounding_box is not None:
        columns = divide_contour_into_columns(bounding_box)
        for i, region in enumerate(columns):
            roi_left, roi_right, roi_top, roi_bottom = region
            cv2.rectangle(image, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 255), 2)

            circle_data = detect_circles_in_region(image, roi_left, roi_right, roi_top, roi_bottom)
            # print(f"Region {i + 1}:")
            # print(f"  Number of circles detected: {len(circle_data)}")

            adjusted_circle_data = detect_circles_in_region(image, roi_left, roi_right, roi_top, roi_bottom)
            # display table
            # display_circle_table(adjusted_circle_data)

            for index, circle in enumerate(adjusted_circle_data):
                x, y = circle['center_x'], circle['center_y']
                userAnswer.append(circle["type"])
                text = str(index + 1)
                cv2.putText(image, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            filled_region_count = detect_black_filled_regions(image, roi_left, roi_right, roi_top, roi_bottom)
            # print(f"  Number of black-filled regions detected: {filled_region_count}")

    groups = [userAnswer[i:i + 4] for i in range(0, len(userAnswer), 4)]

    for x in range(len(groups)):
        for y in range(len(groups[x])):
            if groups[x][y] == "Filled":
                answer[x + 1] = y + 1
                break
        else:
            answer[x + 1] = None

    # for x in answer:
        # print(x, answer[x])

    cv2.imshow('Detected Shapes', image)
    cv2.imwrite("solution/output.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
