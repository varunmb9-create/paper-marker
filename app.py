import os
import json
import time
from io import BytesIO
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Answer Key Extractor", layout="centered")
st.title("📝 Question Paper & Key Tool")

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

def optimize_image(uploaded_file, max_dim=1800):
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

ans_file = st.file_uploader("Upload Answer Key Page", type=["png", "jpg", "jpeg"])

if ans_file:
    if st.button("Extract Answer Key"):
        with st.spinner("Extracting Answer Key accurately..."):
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
            st.success(f"Detected {len(st.session_state.answer_key)} answers!")

if st.session_state.answer_key:
    st.markdown("---")
    st.subheader("📋 Formatted Key for Notes App")

    sorted_items = sorted(
        st.session_state.answer_key.items(),
        key=lambda x: int(x[0]) if x[0].isdigit() else 999
    )

    # Formatted with high-contrast bold text, clear arrows, and large spacing
    bold_copy_text = "\n\n".join([
        f"Q.{str(q).zfill(2)}   ➔   【  {ans}  】"
        for q, ans in sorted_items
    ])

    # Big bold styled preview on the webpage
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            font-size: 26px !important;
            font-weight: 800 !important;
            font-family: monospace, sans-serif !important;
            color: #22c55e !important;
            background-color: #0f172a !important;
            line-height: 2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("Tap inside the box below, select all, and copy:")
    st.text_area(
        label="Copy Text",
        value=bold_copy_text,
        height=350,
        label_visibility="collapsed"
    )

    st.download_button(
        label="📥 Download Bold Key (.txt)",
        data=bold_copy_text,
        file_name="bold_answer_key.txt",
        mime="text/plain"
    )
