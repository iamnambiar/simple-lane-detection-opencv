import cv2
import numpy as np
import sys

def canny(image):
    # Function to apply canny edge detection algorithm
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image

def region_of_interest(image):
    # Function to crop the image to required area
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    # Function to create coordinates for line generation from slope and intercept
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int((3/5)*y1)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slop_intercept(image, lines):
    # Function to determine average slope and intercept for lines in both left and right side of road.
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def display_lines(image, lines):
    # Function to create line on the image from the coordinates.
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 20)
    return line_image


def process_video_from_file():
    # Function to load and process video file.
    cap = cv2.VideoCapture("Resource/test2.mp4")
    while(cap.isOpened()):
        _, frame = cap.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slop_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(line_image, 0.8, frame, 1, 1)
        cv2.imshow("simple-lane-detection", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cap.release()
    cap.destroyAllWindows()
    return 0

def process_image_from_file():
    # Function to load and process image file.
    image = cv2.imread("Resource/test_image.jpg")
    if image is None:
        sys.exit("Could not read the image")
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slop_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(line_image, 0.8, lane_image, 1, 1)
    cv2.imshow("simple-lane-detection", combo_image)
    cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    process_video_from_file()
