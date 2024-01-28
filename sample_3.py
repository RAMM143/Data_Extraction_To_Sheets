import cv2
import numpy as np
import pytesseract
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# Set the path to the Tesseract executable (change this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

drawing = False
mode = True
ix, iy = -1, -1
rectangles = []
img = cv2.imread("C:/Users/amadh/Downloads/WhatsApp Image 2024-01-25 at 18.54.20 (1).jpeg")

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img = get_grayscale(img)

# Get the height and width of the image
height, width = img.shape[:2]

# Set the display size
display_width = 800
display_height = int((height / width) * display_width)

# Create a canvas for drawing rectangles
canvas = np.zeros_like(img)

# Create a new window
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Resize the window to fit the display size
cv2.resizeWindow('Image', display_width, display_height)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mode, rectangles, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y + cv2.getTrackbarPos('Y', 'Image')

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            canvas = np.zeros_like(img)  # Clear the canvas
            if mode == True:
                cv2.rectangle(canvas, (ix, iy), (x, y + cv2.getTrackbarPos('Y', 'Image')), (255, 0, 0), 33)
            else:
                cv2.circle(canvas, (x, y + cv2.getTrackbarPos('Y', 'Image')), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y + cv2.getTrackbarPos('Y', 'Image')), (0, 255, 0), 0)
            name = input("Enter name for the rectangle: ")
            rectangles.append({"x": ix, "y": iy, "w": x - ix, "h": y + cv2.getTrackbarPos('Y', 'Image') - iy, "name": name})
            canvas = np.zeros_like(img)  # Clear the canvas

# Create a trackbar for scrolling
cv2.createTrackbar('Y', 'Image', 0, height - display_height, lambda x: None)

cv2.setMouseCallback('Image', draw_rectangle)

def get_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(tuple(image_extensions))]
    
    return image_files

# Function to perform OCR on the selected sections and save to CSV
def extract_text_and_save_single(img_path):
    global rectangles
    
    result = {}
    
    img = cv2.imread(img_path)
    img = get_grayscale(img)
    
    for rect in rectangles:
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        roi = img[y:y + h, x:x + w]

        # Perform OCR on the region of interest (ROI)
        text = pytesseract.image_to_string(roi, config='--psm 6')  # Adjust the psm parameter based on your requirements

        result[rect["name"]] = text.strip()

    return result

def extract_text_and_save():
    global rectangles

    values = {}
    
    for i in rectangles:
        name = i["name"]
        values[name] = []
    
    folder_path = "C:/Users/amadh/OneDrive/Desktop/Project_Extraction/Test_Images"
    image_files = get_image_files(folder_path)
    
    # Display the floating label
    floating_label = "Data is extracting..."
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_text_and_save_single, os.path.join(folder_path, img_name)) for img_name in image_files]
        
        for future in futures:
            result = future.result()
            for key, value in result.items():
                values[key].append(value)

            # Overlay the label on the image
            cv2.putText(img, floating_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the image with the label
            cv2.imshow('Image', img)
            cv2.waitKey(1)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(values)

    # Save the DataFrame to a CSV file
    df.to_csv('extracted_data.csv', index=False)
    print("Data extracted and saved to extracted_data.csv")

# Main loop for displaying and interacting with the image
while True:
    y_offset = cv2.getTrackbarPos('Y', 'Image')
    img_to_display = img.copy()[y_offset:y_offset + display_height, :]

    for rect in rectangles:
        cv2.rectangle(img_to_display, (rect["x"], rect["y"] - y_offset), (rect["x"] + rect["w"], rect["y"] + rect["h"] - y_offset), (0, 255, 0), 2)
        
    overlay = cv2.addWeighted(img_to_display, 1, canvas[y_offset:y_offset + display_height, :], 0.5, 0)
    cv2.imshow('Image', overlay)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == ord('d'):
        if rectangles:
            rectangles.pop()
            canvas = np.zeros_like(img)
    elif k == ord('r'):
        if rectangles:
            new_name = input("Enter corrected name for the last rectangle: ")
            rectangles[-1]["name"] = new_name
    elif k == ord('e'):
        extract_text_and_save()  # Press 'e' to extract text and save to CSV

    elif k == 27:
        break

print(rectangles)
cv2.destroyAllWindows()
