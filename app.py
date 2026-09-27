import os
import json
import time
import gc
import re
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Answer Key Appender", layout="centered")
st.title("📄 Question Paper + Accurate Answer Key Table")

if "processed_pages" not in st.session_state:
    st.session_state.processed_pages = []
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

MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]

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

def optimize_image(uploaded_file, max_dim=2000):
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

def render_answer_table(page_img, page_keys):
    """Draws a clean, multi-column answer grid directly below the question sheet."""
    img_w, img_h = page_img.size
    
    sorted_items = sorted(
        page_keys.items(), 
        key=lambda x: int(x[0]) if x[0].isdigit() else 999
    )
    
    if not sorted_items:
        return page_img

    cols_per_row = 10
    num_items = len(sorted_items)
    num_rows = (num_items + cols_per_row - 1) // cols_per_row
    
    cell_height = 45
    header_height = 55
    table_height = header_height + (num_rows * cell_height) + 35
    
    new_img = Image.new("RGB", (img_w, img_h + table_height), color=(245, 247, 250))
    new_img.paste(page_img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    # Header Banner
    header_top = img_h + 10
    draw.rectangle([0, header_top, img_w, header_top + 40], fill=(15, 23, 42))
    draw.text((25, header_top + 10), "ANSWER KEY FOR THIS PAGE", fill=(255, 255, 255))

    margin_x = 20
    available_w = img_w - (2 * margin_x)
    col_w = available_w / cols_per_row

    for idx, (q_num, ans) in enumerate(sorted_items):
        r = idx // cols_per_row
        c = idx % cols_per_row
        
        x0 = margin_x + (c * col_w)
        y0 = header_top + 50 + (r * cell_height)
        x1 = x0 + col_w - 6
        y1 = y0 + cell_height - 6
        
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        cell_text = f"{q_num}: ({ans})"
        draw.text((x0 + 10, y0 + 12), cell_text, fill=(30, 41, 59))

    return new_img

q_files = st.file_uploader(
    "1. Upload Question Pages (in page order: Page 1, Page 2, Page 3...)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)
ans_file = st.file_uploader("2. Upload Answer Key Page (containing 'Answers with Explanation')", type=["png", "jpg", "jpeg"])

if q_files and ans_file:
    st.write(f"📁 **{len(q_files)} question page(s) ready.**")

    if st.button("Generate Accurate Answer Tables"):
        st.session_state.processed_pages = []
        st.session_state.pdf_data = None
        st.session_state.answer_key = None

        # Step 1: Read Answer Key with targeted prompt
        with st.spinner("Extracting complete Answer Key accurately..."):
            ans_img = optimize_image(ans_file, max_dim=1800)
            key_prompt = """
            Focus specifically on the section titled 'Answers with Explanation' or the answer key block.
            Notice the formatted answers listed like:
            1(b), 2(a), 3(c), 4(c), 5(a), 6(b), 7(c)...
            and any standalone answer listings like:
            71. (d), 72. (d), 74. (c), 80. (a), 81(a), 82(b), 89(a), 90(b)... up to 100.

            Extract ALL questions from 1 to 100 with their corresponding single letter option (A, B, C, or D).
            IGNORE calculations, dates (like 2020), and mathematical formulas in the explanations.
            Output strictly a clean JSON object:
            {"1": "B", "2": "A", "3": "C", ...}
            """
            key_response = generate_with_fallback(
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw_key = json.loads(key_response.text)
            st.session_state.answer_key = {str(k).strip(): str(v).strip().upper() for k, v in raw_key.items()}
            
            st.success(f"Extracted {len(st.session_state.answer_key)} answers from key.")
            with st.expander("Review Extracted Answer Key"):
                st.json(st.session_state.answer_key)

        processed_list = []

        # Step 2: Detect first and last question on each page
        for idx, file in enumerate(q_files):
            st.markdown(f"--- \n### 📄 Analyzing Page {idx + 1}")
            page_img = optimize_image(file, max_dim=2000)

            with st.spinner(f"Detecting question range for Page {idx + 1}..."):
                range_prompt = """
                Examine this question paper page.
                Look at the very first question number printed on this page (top-left) and the very last question number printed on this page (bottom-right).
                Return strictly a JSON object with 'start_q' and 'end_q' integers:
                {"start_q": 1, "end_q": 30}
                """
                
                resp = generate_with_fallback(
                    contents=[page_img, range_prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                try:
                    q_range = json.loads(resp.text)
                    start_q = int(q_range.get("start_q", 1))
                    end_q = int(q_range.get("end_q", 30))
                except Exception:
                    # Sensible defaults for newspaper pages if detection fails
                    start_q = (idx * 30) + 1
                    end_q = (idx + 1) * 30

                st.write(f"Page {idx + 1} Question Range: **Q.{start_q} to Q.{end_q}**")

                # Slice strictly the continuous range from the answer key
                page_answers = {}
                for num in range(start_q, end_q + 1):
                    s_num = str(num)
                    if s_num in st.session_state.answer_key:
                        page_answers[s_num] = st.session_state.answer_key[s_num]

                final_page_img = render_answer_table(page_img, page_answers)
                processed_list.append(final_page_img)
                gc.collect()

        st.session_state.processed_pages = processed_list

        if processed_list:
            pdf_buf = BytesIO()
            processed_list[0].save(
                pdf_buf,
                format="PDF",
                save_all=True,
                append_images=processed_list[1:] if len(processed_list) > 1 else []
            )
            st.session_state.pdf_data = pdf_buf.getvalue()

# Display Persistent Results & Downloads
if st.session_state.processed_pages:
    st.success("✅ Clean answer key tables generated successfully!")
    
    if st.session_state.pdf_data:
        st.download_button(
            label="📥 Download All Pages with Answer Tables as PDF",
            data=st.session_state.pdf_data,
            file_name="paper_with_verified_keys.pdf",
            mime="application/pdf",
            key="btn_download_pdf"
        )
        st.markdown("---")

    for idx, page_img in enumerate(st.session_state.processed_pages):
        st.subheader(f"📄 Page {idx + 1}")
        st.image(page_img, caption=f"Page {idx + 1} with Verified Table", use_container_width=True)

        img_buf = BytesIO()
        page_img.save(img_buf, format="JPEG", quality=92)
        st.download_button(
            label=f"⬇️ Download Page {idx + 1} (JPG)",
            data=img_buf.getvalue(),
            file_name=f"page_{idx + 1}_with_answers.jpg",
            mime="image/jpeg",
            key=f"btn_dl_page_{idx + 1}"
        )
        st.markdown("---")
