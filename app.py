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
st.title("📝 Question Paper Auto-Marker (Multi-Column)")

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
    st.info("Please enter your Gemini API Key to proceed.")
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
    raise RuntimeError("API busy. Please retry.")

def optimize_image(uploaded_file, max_dim=2200):
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

q_files = st.file_uploader(
    "1. Upload Question Pages (Newspaper format)", 
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
        with st.spinner("Extracting complete Answer Key..."):
            ans_img = optimize_image(ans_file, max_dim=1600)
            key_prompt = (
                "Extract all question numbers (1, 2, 3... up to 100) and their exact single correct option (A, B, C, or D) "
                "from this answer key image (e.g. from 'Answers with Explanation' box). "
                "Format strictly as JSON map: {\"1\": \"B\", \"2\": \"A\", \"3\": \"C\", ...}"
            )
            key_response = generate_with_fallback(
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw_key = json.loads(key_response.text)
            # Normalize to uppercase
            st.session_state.answer_key = {str(k).strip(): str(v).strip().upper() for k, v in raw_key.items()}
            st.success(f"Extracted {len(st.session_state.answer_key)} answers from key.")

        annotated_list = []

        # Step 2: Annotate Question Pages using 4-Column Strip Division
        for page_idx, file in enumerate(q_files):
            st.markdown(f"--- \n### 📄 Processing Page {page_idx + 1}")
            full_img = optimize_image(file, max_dim=2200)
            img_w, img_h = full_img.size
            annotated_img = full_img.copy()
            draw = ImageDraw.Draw(annotated_img)

            # Define 4 column strips with slight overlap
            num_cols = 4
            col_width = img_w / num_cols

            col_progress = st.progress(0)
            for c in range(num_cols):
                col_left = max(0, int(c * col_width - 15))
                col_right = min(img_w, int((c + 1) * col_width + 15))
                
                # Crop vertical strip
                col_crop = full_img.crop((col_left, 0, col_right, img_h))
                c_w, c_h = col_crop.size

                strip_prompt = f"""
                You are analyzing a SINGLE COLUMN vertical strip of a question paper.
                Answer Key: {json.dumps(st.session_state.answer_key)}

                RULES:
                1. Identify which question numbers appear in this single vertical column.
                2. For each question in this column, locate the exact bounding box around the CORRECT OPTION letter label:
                   (a), (b), (c), or (d) matching the answer key.
                3. Do NOT mark question numbers or question text. Circle ONLY the letter label itself.
                4. Coordinates must be normalized integers [0, 1000] relative to THIS COLUMN STRIP: [ymin, xmin, ymax, xmax].

                Output JSON:
                [
                  {{"q_no": "1", "matched_option": "B", "box_2d": [ymin, xmin, ymax, xmax]}}
                ]
                """

                try:
                    box_response = generate_with_fallback(
                        contents=[col_crop, strip_prompt],
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    detections = json.loads(box_response.text)
                except Exception:
                    detections = []

                # Draw detections back onto the full page
                for item in detections:
                    box = item.get("box_2d")
                    if box and len(box) == 4:
                        ymin, xmin, ymax, xmax = box
                        # Map strip coordinates back to full image space
                        top = max(0, int((ymin / 1000) * c_h))
                        bottom = min(img_h, int((ymax / 1000) * c_h))
                        left = max(0, int(col_left + (xmin / 1000) * c_w))
                        right = min(img_w, int(col_left + (xmax / 1000) * c_w))

                        # Draw green highlight circle around the option label
                        pad = 4
                        draw.ellipse(
                            [left - pad, top - pad, right + pad, bottom + pad],
                            outline="#00E600",
                            width=4
                        )

                col_progress.progress((c + 1) / num_cols)
                gc.collect()

            annotated_list.append(annotated_img)

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

# Display Results and Downloads
if st.session_state.annotated_pages:
    st.success("✅ All columns annotated successfully!")
    
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
        page_img.save(img_buf, format="JPEG", quality=92)
        st.download_button(
            label=f"⬇️ Download Page {idx + 1} (JPG)",
            data=img_buf.getvalue(),
            file_name=f"marked_page_{idx + 1}.jpg",
            mime="image/jpeg",
            key=f"btn_dl_page_{idx + 1}"
        )
        st.markdown("---")
