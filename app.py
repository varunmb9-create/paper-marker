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

if "annotated_pages" not in st.session_state:
    st.session_state.annotated_pages = []
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None
if "answer_key" not in st.session_state:
    st.session_state.answer_key = None

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
    raise RuntimeError("API servers busy. Please retry in a moment.")

def optimize_image(uploaded_file, max_dim=1800):
    """Keeps higher resolution so small Malayalam option labels are razor sharp."""
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
        st.session_state.annotated_pages = []
        st.session_state.pdf_data = None
        st.session_state.answer_key = None

        # Step 1: Read Answer Key
        with st.spinner("Extracting Answer Key..."):
            ans_img = optimize_image(ans_file, max_dim=1400)
            key_prompt = (
                "Extract all question numbers (1, 2, 3...) and their single correct option (A, B, C, or D) from this answer key. "
                "Output strictly a JSON object: {\"1\": \"A\", \"2\": \"D\", ...}"
            )
            key_response = generate_with_fallback(
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            st.session_state.answer_key = json.loads(key_response.text)

        annotated_list = []

        # Step 2: Annotate Question Pages using Anchored Option Detection
        for idx, file in enumerate(q_files):
            with st.spinner(f"Accurately locating options for Page {idx + 1}..."):
                q_img = optimize_image(file, max_dim=1800)

                detect_prompt = f"""
                You are an expert OCR document analyzer.
                Answer Key: {json.dumps(st.session_state.answer_key)}

                INSTRUCTIONS:
                1. Identify the question numbers (1, 2, 3, etc.) located on the LEFT side of each question.
                2. For each question number found on this page that exists in the answer key:
                   - Look ONLY inside the immediate body/options block belonging to that specific question number.
                   - Do NOT jump across columns or look into other questions.
                   - Locate the bounding box of ONLY the correct option label symbol: e.g. 'A', 'B', 'C', 'D' or '(A)', '(B)', '(C)', '(D)'.
                3. STRICT RULE: Output exactly ONE bounding box per question number.
                4. Coordinates must be normalized integers [0, 1000] in format [ymin, xmin, ymax, xmax].

                Output format strictly as JSON:
                [
                  {{"q_no": "1", "matched_option": "B", "box_2d": [ymin, xmin, ymax, xmax]}}
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

                # Enforce strictly 1 mark per question
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

                        # Draw a clear green circle around the option letter
                        pad_x = max(3, int((right - left) * 0.2))
                        pad_y = max(3, int((bottom - top) * 0.2))
                        
                        draw.ellipse(
                            [left - pad_x, top - pad_y, right + pad_x, bottom + pad_y],
                            outline="#00E600",
                            width=4
                        )

                annotated_list.append(annotated_img)
                gc.collect()

        st.session_state.annotated_pages = annotated_list

        if annotated_list:
            pdf_buf = BytesIO()
            annotated_list[0].save(
                pdf_buf,
                format="PDF",
                save_all=True,
                append_images=annotated_list[1:] if len(annotated_list) > 1 else []
            )
            st.session_state.pdf_data = pdf_buf.getvalue()

# Persistent UI Display
if st.session_state.annotated_pages:
    st.success(f"Detected {len(st.session_state.answer_key)} questions in Answer Key.")
    
    if st.session_state.pdf_data:
        st.download_button(
            label="📥 Download All Marked Pages as PDF",
            data=st.session_state.pdf_data,
            file_name="marked_paper.pdf",
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
