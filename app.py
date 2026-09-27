import os
import json
import streamlit as st
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Marker", layout="centered")
st.title("📝 Question Paper Auto-Marker")

# Read API Key from Render environment variable or manual input
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API Key above to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

# Image Uploaders
q_file = st.file_uploader("1. Upload Question Paper Page", type=["png", "jpg", "jpeg"])
ans_file = st.file_uploader("2. Upload Answer Key Page", type=["png", "jpg", "jpeg"])

if q_file and ans_file:
    q_img = Image.open(q_file).convert("RGB")
    ans_img = Image.open(ans_file).convert("RGB")
    
    st.image(q_img, caption="Question Paper", use_container_width=True)

    if st.button("Mark Answers"):
        with st.spinner("Extracting answer key..."):
            key_prompt = (
                "Extract all question numbers and their corresponding correct options from this answer key. "
                "Output strictly a JSON mapping like: {\"1\": \"A\", \"2\": \"C\", ...}"
            )
            key_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            answer_key = json.loads(key_response.text)
            st.write("Detected Key:", answer_key)

        with st.spinner("Finding questions and bounding boxes..."):
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
                model="gemini-2.5-flash",
                contents=[q_img, detect_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            detections = json.loads(box_response.text)

        # Draw green boxes on the questions
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
                draw.rectangle([left, top, right, bottom], outline="#00FF00", width=5)

        st.subheader("✅ Marked Question Paper")
        st.image(annotated_img, use_container_width=True)

        # Download buffer
        from io import BytesIO
        buf = BytesIO()
        annotated_img.save(buf, format="JPEG")
        st.download_button("Download Marked Image", data=buf.getvalue(), file_name="marked.jpg", mime="image/jpeg")
