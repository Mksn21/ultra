import streamlit as st
from fpdf import FPDF
import base64
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import pandas as pd

top_cut = 88
bottom_cut = 30
left_cut = 38
right_cut = 155

#create link to download PDF
def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

#get label from user
def get_label_from_user():
    label_str = st.text_input("Enter the labels of objects you want to measure (separate with comma and '&' between two labels):")
    labels = []
    if label_str:
        pairs = label_str.split(',')
        for pair in pairs:
            label_pair = tuple(map(int, pair.strip().split('&')))
            if len(label_pair) == 2:
                labels.append(label_pair)
    return labels

#classify orientation
def classify_orientation(centroid1, centroid2):
    vertical_diff = abs(centroid1[0] - centroid2[0])
    horizontal_diff = abs(centroid1[1] - centroid2[1])
    if vertical_diff > horizontal_diff:
        return 'vertical'
    else:
        return 'horizontal'

#pixel measure
def pixel_measure(crop_im):
    top_cut = 240
    bottom_cut = 340
    left_cut = 200
    right_cut = 590
    return 

#streamlit program
def main():
    st.title("Ultrasound Calibration")
    st.subheader("Distance and Pixel Measurement")

    #user input for data
    name = st.text_input("Name")
    address = st.text_input("Address")
    equipment_name = st.text_input("Equipment Name")
    manufacturer = st.text_input("Manufacturer")
    equipment_type = st.text_input("Type")
    serial_number = st.text_input("Serial Number")
    report_text = f"""
    Owner's Identification
    Name: {name}
    Address: {address}

    Instrument Identification
    Equipment Name: {equipment_name}
    Manufacturer: {manufacturer}
    Type: {equipment_type}
    Serial Number: {serial_number}
    """
    #upload image
    uploaded_file = st.file_uploader("Insert image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        im = np.array(image)
        gray_im = im[top_cut:-bottom_cut, left_cut:-right_cut]
        hsv_im = cv2.cvtColor(gray_im, cv2.COLOR_RGB2HSV)
        gray_image = np.array(im)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([130, 255, 255])
        white_mask = cv2.inRange(hsv_im, lower_white, upper_white)
        white_object = cv2.bitwise_and(gray_im, gray_im, mask=white_mask)
        white_object_gray = cv2.cvtColor(white_object, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(white_object_gray, 1, 255, cv2.THRESH_BINARY)
        min_size = 20
        cleaned_binary_image = remove_small_objects(binary_image, min_size=min_size)
        label_image = label(cleaned_binary_image)

        fig, ax = plt.subplots()
        ax.imshow(label_image, cmap='nipy_spectral')

        for props in regionprops(label_image):
            y, x = props.centroid
            ax.text(x, y, f"{props.label}", color='red', fontsize=8, ha='center', va='center')

        ax.set_title('Labeled Objects')
        ax.axis('off')
        st.pyplot(fig)

        label_pairs = get_label_from_user()
        pixel_to_mm = 0.2645833333

        #graylevels measurement
        crop_im = gray_image[top_cut:-bottom_cut, left_cut:-right_cut]
        average_pixel_value = np.mean(crop_im) 
        if average_pixel_value <= 100  :
            result = "(Unqualified)"
        else :
            result = "(Qualified)"

        #show label and result
        result_data = []
        for i, label_pair in enumerate(label_pairs):
            obj1 = next((obj for obj in regionprops(label_image) if obj.label == label_pair[0]), None)
            obj2 = next((obj for obj in regionprops(label_image) if obj.label == label_pair[1]), None)

            if obj1 and obj2:
                centroid1 = obj1.centroid
                centroid2 = obj2.centroid
                orientation = classify_orientation(centroid1, centroid2)
                distance_pixels = euclidean(centroid1, centroid2)
                distance_mm = distance_pixels * pixel_to_mm
                location = 'Vertical' if orientation == 'vertical' else 'Horizontal'
                result_data.append({
                    'No.': i + 1,
                    'Location': location,
                    'Label 1': str(int(label_pair[0])),
                    'Label 2': str(int(label_pair[1])),
                    'Distance (mm)': distance_mm
                })

        #convertresult data to DataFrame
        result_df = pd.DataFrame(result_data)

        #mean distance and standard deviation calculation
        mean_distance = np.mean([row['Distance (mm)'] for row in result_data])
        std_deviation_distance = np.std([row['Distance (mm)'] for row in result_data])

        #report PDF
        export_as_pdf = st.button("Export Calibration Report")
        if export_as_pdf:
            pdf = FPDF()
            pdf.header()
            pdf.add_page()
            pdf.set_font('Times', 'B', 30)
            pdf.cell(200, 13, txt="CALIBRATION RESULTS", ln=True, align="C")
            pdf.set_font('Times', 'BU', 20)
            pdf.cell(200, 13, txt="Standards", align="L", ln = True)
            pdf.set_font('Times', '', 15)
            pdf.cell(200, 5, txt="Ultrasound Phantom : Gammex Inc", align='L', ln = True)
            pdf.set_font('Times', '', 15)
            pdf.multi_cell(200, 8, report_text, align='L')
            pdf.set_font('Times', 'BU', 20)
            pdf.cell(200, 15, "Distance Measurement", ln=True, align='L')
            #headers
            pdf.set_font('Times', 'B', 15)
            pdf.cell(30, 10, 'No.', 1, align='C')
            pdf.cell(30, 10, 'Location', 1, align='C')
            pdf.cell(30, 10, 'Label 1', 1, align='C')
            pdf.cell(30, 10, 'Label 2', 1, align='C')
            pdf.cell(40, 10, 'Distance (mm)', 1, align='C')
            pdf.ln()
            #data 
            for _, row in result_df.iterrows():
                pdf.set_font('Times', '', 15)
                pdf.cell(30, 10, str(row['No.']), 1, align='C')
                pdf.cell(30,10, str(row['Location']), 1, align='C')
                pdf.cell(30, 10, row['Label 1'], 1, align='C')
                pdf.cell(30, 10, row['Label 2'], 1, align='C')
                pdf.cell(40, 10, f"{row['Distance (mm)']:.2f}", 1, align='C')
                pdf.ln()
            #mean distance and standard deviation
            pdf.set_font('Times', '', 15)
            pdf.cell(200, 10, f"Mean Distance: {mean_distance:.2f}", ln=True)
            pdf.cell(200, 8, f"Standard Deviation: {std_deviation_distance:.2f}", ln=True)
            #average graylevels
            pdf.set_font('Times', 'BU', 20)
            pdf.cell(200, 15, "Grayscale Levels Measurement", ln=True, align='L')
            pdf.set_font('Times', '', 15)
            pdf.cell(200, 2, f"Average Grayscale Levels: {average_pixel_value:.2f} {result}", ln = True)

            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "calibration_report")
            st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
