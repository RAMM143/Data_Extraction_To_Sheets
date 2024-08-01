Image to Sheet Data Extractor
Overview
This Python software extracts data from certificates of marks or any other images and converts it into sheets. Designed to handle bulk image inputs, it automates the process of data extraction using Optical Character Recognition (OCR) technology.

Features
Bulk Image Processing: Handles large volumes of images in one go.
Automated Data Extraction: Uses OCR to extract text and data from specified regions in the images.
Rectangles and Labels: Allows for marking specific areas (rectangles) on the images and labeling them for accurate data extraction.
Output to Sheets: Consolidates extracted data into well-organized sheets for easy analysis and use.
How It Works
Input Images: Provide a bulk of images that need to be processed.
Rectangle Marking: The software allows you to draw rectangles around the areas of interest in the images and label these regions.
Data Extraction: Using OCR, the software extracts the text from the labeled rectangles.
Sheet Generation: All extracted data is compiled into sheets, providing a structured format for further use.
Usage
Prepare your images: Ensure all images are ready for processing and placed in the input directory.
Run the software: Execute the Python script to start the extraction process.
Review Output: Check the generated sheets for extracted data.
Dependencies
Python 3.x
OCR Library (e.g., Tesseract)
Image Processing Libraries (e.g., OpenCV, PIL)
