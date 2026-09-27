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

# Initialize Session State so downloads don't reset the page
if "annotated_pages" not in st.session_state:
    st.session_state.annotated_pages = []
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None
if "answer_key" not in st.session_state:
    st.session_state.answer_key = None

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API Key to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]

def generate_with_fallback(contents, config=None):
    """Fallback handler to bypass traffic surges and 503 limits."""
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
    raise RuntimeError("API servers are currently busy. Please retry in a few moments.")

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
        # Reset previous run data
        st.session_state.annotated_pages = []
        st.session_state.pdf_data = None
        st.session_state.answer_key = None

        # Step 1: Read Answer Key
        with st.spinner("Reading answer key..."):
            ans_img = optimize_image(ans_file, max_dim=1200)
            key_prompt = (
                "Extract all question numbers and their corresponding correct options from this answer key image. "
                "Every question has strictly one correct answer from A, B, C, or D. "
                "Output strictly a JSON key-value map, like: {\"1\": \"A\", \"2\": \"B\", \"3\": \"C\"}."
            )
            key_response = generate_with_fallback(
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            st.session_state.answer_key = json.loads(key_response.text)

        annotated_list = []

        # Step 2: Annotate Question Pages
        for idx, file in enumerate(q_files):
            with st.spinner(f"Marking choices on Page {idx + 1}..."):
                q_img = optimize_image(file, max_dim=1600)

                detect_prompt = f"""
                You are an exam paper annotator. 
                Answer key: {json.dumps(st.session_state.answer_key)}

                CRITICAL RULES:
                1. Every question (1, 2, 3...) has exactly four options (A, B, C, D) and ONLY ONE correct answer.
                2. Only check questions from the answer key that actually appear on this page.
                3. For each question, locate ONLY the correct option letter label itself (e.g. 'A', 'B', 'C', 'D' or '(A)', '(B)', '(C)', '(D)').
                4. NEVER mark multiple options for the same question number. Strictly ONE label box per question.
                5. Coordinates must be normalized [0, 1000] in format [ymin, xmin, ymax, xmax].

                Output strictly a JSON list:
                [
                  {{"q_no": "1", "option_letter": "A", "box_2d": [ymin, xmin, ymax, xmax]}}
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

                # Deduplicate: enforce strictly 1 answer per question
                unique_detections = {}
                for item in detections:
                    q_no = str(item.get("q_no", "")).strip()
                    if q_no and q_no not in unique_detections:
                        unique_detections[q_no] = item

                annotated_img = q_img.copy()
                draw = ImageDraw.Draw(annotated_img)
                w, h = annotated_img.size

                for q_no, item in unique_detections.items():
                    box = item.get("box_2d")
                    if box and len(box) == 4:
                        ymin, xmin, ymax, xmax = box
                        top = max(0, int((ymin / 1000) * h))
                        left = max(0, int((xmin / 1000) * w))
                        bottom = min(h, int((ymax / 1000) * h))
                        right = min(w, int((xmax / 1000) * w))

                        pad = 4
                        draw.ellipse(
                            [left - pad, top - pad, right + pad, bottom + pad],
                            outline="#00DD00",
                            width=5
                        )

                annotated_list.append(annotated_img)
                gc.collect()

        st.session_state.annotated_pages = annotated_list

        # Pre-compile the combined PDF into session state
        if annotated_list:
            pdf_buf = BytesIO()
            annotated_list[0].save(
                pdf_buf,
                format="PDF",
                save_all=True,
                append_images=annotated_list[1:] if len(annotated_list) > 1 else []
            )
            st.session_state.pdf_data = pdf_buf.getvalue()

# Display persisted results from session state
if st.session_state.annotated_pages:
    st.success(f"Detected {len(st.session_state.answer_key)} answers in key.")
    
    if st.session_state.pdf_data:
        st.download_button(
            label="📥 Download All Marked Pages as Single PDF",
            data=st.session_state.pdf_data,
            file_name="all_marked_pages.pdf",
            mime="application/pdf",
            key="btn_download_pdf"
        )
        st.markdown("---")

    for idx, page_img in enumerate(st.session_state.annotated_pages):
        st.subheader(f"📄 Page {idx + 1}")
        st.image(page_img, caption=f"Marked Page {idx + 1}", use_container_width=True)

        img_buf = BytesIO()
        page_img.save(img_buf, format="JPEG", quality=90)
        st.download_button(
            label=f"⬇️ Download Page {idx + 1} (JPG)",
            data=img_buf.getvalue(),
            file_name=f"marked_page_{idx + 1}.jpg",
            mime="image/jpeg",
            key=f"btn_dl_page_{idx + 1}"
        )
        st.markdown("---")
