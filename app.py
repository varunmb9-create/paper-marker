import os
import json
import time
import gc
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Marker", layout="centered")
st.title("📝 Question Paper Auto-Marker")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API Key to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]

def generate_with_fallback(contents, config=None):
    for model_name in MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(2)
                    continue
                break
    raise RuntimeError("API service busy. Please retry in a moment.")

def optimize_image(uploaded_file, max_dim=1600):
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

q_files = st.file_uploader(
    "1. Upload Question Paper Pages", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)
ans_file = st.file_uploader("2. Upload Answer Key Page", type=["png", "jpg", "jpeg"])

if q_files and ans_file:
    st.write(f"📁 **{len(q_files)} page(s) ready.**")

    if st.button("Mark Answers"):
        # Step 1: Read Answer Key
        with st.spinner("Step 1: Reading Answer Key..."):
            ans_img = optimize_image(ans_file, max_dim=1200)
            key_prompt = (
                "Extract all question numbers and their corresponding correct options from this answer key image. "
                "Output strictly a JSON key-value map, like: {\"1\": \"A\", \"2\": \"B\", \"3\": \"C\"}."
            )
            key_response = generate_with_fallback(
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            answer_key = json.loads(key_response.text)
            st.success(f"Extracted {len(answer_key)} answers from key:")
            st.json(answer_key)

        annotated_pages = []

        # Step 2: Annotate Question Pages
        for idx, file in enumerate(q_files):
            st.markdown(f"--- \n### 📄 Page {idx + 1}")
            q_img = optimize_image(file, max_dim=1600)

            with st.spinner(f"Detecting question locations on Page {idx + 1}..."):
                detect_prompt = f"""
                You are an exam paper grader analyzing this question paper page.
                Here is the ground truth answer key mapping:
                {json.dumps(answer_key)}

                Your task:
                1. Identify which questions from the answer key are printed on this page.
                2. For each question found, locate the bounding box of the CORRECT OPTION letter or text (e.g. A, B, C, D or 1, 2, 3, 4).
                3. Bounding box coordinates must be normalized between 0 and 1000 in [ymin, xmin, ymax, xmax] format.

                Output strictly a JSON list:
                [
                  {{"q_no": "1", "ans": "A", "box_2d": [ymin, xmin, ymax, xmax]}}
                ]
                """

                box_response = generate_with_fallback(
                    contents=[q_img, detect_prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )

                try:
                    detections = json.loads(box_response.text)
                except Exception:
                    detections = []

            st.write(f"Found **{len(detections)}** marked questions on this page.")

            # Draw green highlighter / box on the image
            annotated_img = q_img.copy()
            draw = ImageDraw.Draw(annotated_img)
            w, h = annotated_img.size

            for item in detections:
                box = item.get("box_2d")
                if box and len(box) == 4:
                    ymin, xmin, ymax, xmax = box
                    top = max(0, int((ymin / 1000) * h))
                    left = max(0, int((xmin / 1000) * w))
                    bottom = min(h, int((ymax / 1000) * h))
                    right = min(w, int((xmax / 1000) * w))

                    # Expand slightly for visibility
                    draw.rectangle(
                        [left - 4, top - 2, right + 4, bottom + 2], 
                        outline="#00FF00", 
                        width=6
                    )

            annotated_pages.append(annotated_img)
            st.image(annotated_img, caption=f"Marked Page {idx + 1}", use_container_width=True)
            gc.collect()

        if annotated_pages:
            pdf_buf = BytesIO()
            annotated_pages[0].save(
                pdf_buf, 
                format="PDF", 
                save_all=True, 
                append_images=annotated_pages[1:] if len(annotated_pages) > 1 else []
            )

            st.download_button(
                label="📥 Download Marked PDF",
                data=pdf_buf.getvalue(),
                file_name="marked_paper.pdf",
                mime="application/pdf"
            )
