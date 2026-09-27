import os
import json
import time
import gc
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Answer Key Appender", layout="centered")
st.title("📄 Question Paper + Answer Key Table")

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
    
    # Sort items by numeric question number
    sorted_items = sorted(
        page_keys.items(), 
        key=lambda x: int(x[0]) if x[0].isdigit() else 999
    )
    
    if not sorted_items:
        return page_img

    # Grid layout calculation: 10 items per row
    cols_per_row = 10
    num_items = len(sorted_items)
    num_rows = (num_items + cols_per_row - 1) // cols_per_row
    
    cell_height = 42
    header_height = 50
    table_height = header_height + (num_rows * cell_height) + 30
    
    # Create combined canvas
    new_img = Image.new("RGB", (img_w, img_h + table_height), color=(245, 247, 250))
    new_img.paste(page_img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    # Section Header Banner
    header_top = img_h + 10
    draw.rectangle([0, header_top, img_w, header_top + 36], fill=(30, 41, 59))
    draw.text((25, header_top + 8), "ANSWER KEY FOR THIS PAGE", fill=(255, 255, 255))

    # Calculate column widths
    margin_x = 20
    available_w = img_w - (2 * margin_x)
    col_w = available_w / cols_per_row

    # Draw Table Cells
    for idx, (q_num, ans) in enumerate(sorted_items):
        r = idx // cols_per_row
        c = idx % cols_per_row
        
        x0 = margin_x + (c * col_w)
        y0 = header_top + 45 + (r * cell_height)
        x1 = x0 + col_w - 4
        y1 = y0 + cell_height - 4
        
        # Cell background card
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        
        # Content formatting: "Q.1 : [B]"
        cell_text = f"Q.{q_num}: {ans}"
        draw.text((x0 + 8, y0 + 10), cell_text, fill=(15, 23, 42))

    return new_img

q_files = st.file_uploader(
    "1. Upload Question Pages (Newspaper / Paper format)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)
ans_file = st.file_uploader("2. Upload Answer Key Page", type=["png", "jpg", "jpeg"])

if q_files and ans_file:
    st.write(f"📁 **{len(q_files)} question page(s) ready.**")

    if st.button("Generate Answer Keys Below Pages"):
        st.session_state.processed_pages = []
        st.session_state.pdf_data = None
        st.session_state.answer_key = None

        # Step 1: Read Complete Answer Key
        with st.spinner("Extracting complete answer key..."):
            ans_img = optimize_image(ans_file, max_dim=1600)
            key_prompt = (
                "Extract all question numbers (1, 2, 3... up to 100) and their exact single correct option (A, B, C, or D) "
                "from this answer key image (e.g., from 'Answers with Explanation' box). "
                "Format strictly as JSON map: {\"1\": \"B\", \"2\": \"A\", \"3\": \"C\", ...}"
            )
            key_response = generate_with_fallback(
                contents=[ans_img, key_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw_key = json.loads(key_response.text)
            st.session_state.answer_key = {str(k).strip(): str(v).strip().upper() for k, v in raw_key.items()}
            st.success(f"Extracted {len(st.session_state.answer_key)} total answers from key.")

        processed_list = []

        # Step 2: Identify Questions on each page & append footer table
        for idx, file in enumerate(q_files):
            st.markdown(f"--- \n### 📄 Analyzing Page {idx + 1}")
            page_img = optimize_image(file, max_dim=2000)

            with st.spinner(f"Detecting which questions appear on Page {idx + 1}..."):
                range_prompt = """
                Examine this question paper page.
                Identify all question numbers that appear on this page (e.g. 1 to 30, or 31 to 70).
                Return strictly a JSON list of integers representing every question number present on this page:
                [1, 2, 3, 4, 5, ...]
                """
                
                resp = generate_with_fallback(
                    contents=[page_img, range_prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                try:
                    present_q_nums = json.loads(resp.text)
                except Exception:
                    present_q_nums = []

                # Filter global answer key for only the questions present on this page
                page_answers = {}
                for q_num in present_q_nums:
                    q_str = str(q_num).strip()
                    if q_str in st.session_state.answer_key:
                        page_answers[q_str] = st.session_state.answer_key[q_str]

                st.write(f"Found **{len(page_answers)}** questions for this page.")

                # Render combined image with table appended below
                final_page_img = render_answer_table(page_img, page_answers)
                processed_list.append(final_page_img)
                gc.collect()

        st.session_state.processed_pages = processed_list

        # Step 3: Bundle into single PDF
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
    st.success("✅ Answer key tables successfully appended to all pages!")
    
    if st.session_state.pdf_data:
        st.download_button(
            label="📥 Download All Pages with Answer Tables as PDF",
            data=st.session_state.pdf_data,
            file_name="paper_with_answer_keys.pdf",
            mime="application/pdf",
            key="btn_download_pdf"
        )
        st.markdown("---")

    for idx, page_img in enumerate(st.session_state.processed_pages):
        st.subheader(f"📄 Page {idx + 1}")
        st.image(page_img, caption=f"Page {idx + 1} with Answer Key Table", use_container_width=True)

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
