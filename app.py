import os
import json
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Marker", layout="centered")
st.title("📝 Multi-Page Question Paper Marker")

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API Key to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

# Model configuration
MODEL_NAME = "gemini-3.8-flash"

# Allow multiple question pages
q_files = st.file_uploader(
    "1. Upload Question Paper Pages (Select multiple)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)
ans_file = st.file_uploader("2. Upload Answer Key Page", type=["png", "jpg", "jpeg"])

if q_files and ans_file:
    st.write(f"📁 **{len(q_files)} page(s) uploaded.**")
    ans_img = Image.open(ans_file).convert("RGB")

    if st.button("Mark All Pages"):
        # Step 1: Read Answer Key once
        with st.spinner("Extracting answer key..."):
            key_prompt = (
                "Extract all question numbers and their corresponding correct options from this answer key. "
                "Output strictly a JSON mapping like: {\"1\": \"A\", \"2\": \"C\", ...}"
            )
            key_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            answer_key = json.loads(key_response.text)
            st.success(f"Detected Answer Key ({len(answer_key)} answers found)")

        annotated_pages = []

        # Step 2: Loop through each question page
        progress_bar = st.progress(0)
        for idx, file in enumerate(q_files):
            st.write(f"🔍 Processing Page {idx + 1} of {len(q_files)}...")
            q_img = Image.open(file).convert("RGB")

            detect_prompt = f"""
            You are an image annotation assistant. Here is the answer key mapping: {json.dumps(answer_key)}.
            Find all the questions on this page that are present in the answer key.
            For each matching question, find the bounding box of the text/label of the CORRECT OPTION.
            Bounding boxes must be normalized coordinates scaled to [0, 1000] in the format: [ymin, xmin, ymax, xmax].
            Return a JSON list of objects:
            [
              {{"question_number": "1", "correct_option": "A", "box_2d": [ymin, xmin, ymax, xmax]}}
            ]
            """
            
            box_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[q_img, detect_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            try:
                detections = json.loads(box_response.text)
            except Exception:
                detections = []

            # Draw green bounding boxes
            annotated_img = q_img.copy()
            draw = ImageDraw.Draw(annotated_img)
            w, h = annotated_img.size

            for item in detections:
                box = item.get("box_2d")
                if box and len(box) == 4:
                    ymin, xmin, ymax, xmax = box
                    top = (ymin / 1000) * h
                    left = (xmin / 1000) * w
                    bottom = (ymax / 1000) * h
                    right = (xmax / 1000) * w
                    draw.rectangle([left, top, right, bottom], outline="#00FF00", width=6)

            annotated_pages.append(annotated_img)
            st.image(annotated_img, caption=f"Page {idx + 1} Marked", use_container_width=True)
            progress_bar.progress((idx + 1) / len(q_files))

        # Step 3: Bundle all annotated pages into a single PDF
        if annotated_pages:
            pdf_buf = BytesIO()
            first_page = annotated_pages[0]
            rest_pages = annotated_pages[1:] if len(annotated_pages) > 1 else []
            
            first_page.save(
                pdf_buf, 
                format="PDF", 
                save_all=True, 
                append_images=rest_pages
            )

            st.download_button(
                label="📥 Download All Marked Pages as PDF",
                data=pdf_buf.getvalue(),
                file_name="marked_question_paper.pdf",
                mime="application/pdf"
            )
