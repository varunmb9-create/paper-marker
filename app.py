import os
import json
import time
import gc
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

st.set_page_config(page_title="Paper Marker & Floating Key", layout="centered")

# Persistent state
if "answer_key" not in st.session_state:
    st.session_state.answer_key = None
if "processed_pages" not in st.session_state:
    st.session_state.processed_pages = []
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

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

def inject_floating_widget(answer_key_dict):
    """Renders a floating expandable bubble widget directly on the screen."""
    sorted_items = sorted(
        answer_key_dict.items(),
        key=lambda x: int(x[0]) if x[0].isdigit() else 999
    )
    
    rows_html = "".join([
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #e2e8f0; font-family:sans-serif;">
            <span style="font-weight:600; color:#334155; font-size:16px;">Q.{q}</span>
            <span style="background:#0f172a; color:#ffffff; font-weight:700; padding:4px 12px; border-radius:6px; font-size:16px;">{ans}</span>
        </div>
        """
        for q, ans in sorted_items
    ])

    widget_code = f"""
    <div id="float-container" style="position:fixed; bottom:25px; right:20px; z-index:999999;">
        <!-- Collapsed Floating Button -->
        <button id="float-btn" onclick="toggleWidget()" style="
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: #2563eb;
            color: white;
            border: none;
            box-shadow: 0 4px 14px rgba(0,0,0,0.35);
            font-size: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;">
            📝
        </button>

        <!-- Expanded Vertical Sidebar -->
        <div id="float-panel" style="
            display: none;
            width: 200px;
            height: 380px;
            background: #ffffff;
            border-radius: 14px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            border: 2px solid #2563eb;
            flex-direction: column;
            overflow: hidden;">
            
            <div style="background:#2563eb; color:white; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; font-family:sans-serif; font-weight:bold;">
                <span>Key List</span>
                <span onclick="toggleWidget()" style="cursor:pointer; font-size:18px;">✕</span>
            </div>
            
            <div style="overflow-y:auto; flex:1; background:#f8fafc;">
                {rows_html}
            </div>
        </div>
    </div>

    <script>
        function toggleWidget() {{
            var btn = document.getElementById("float-btn");
            var panel = document.getElementById("float-panel");
            if (panel.style.display === "none" || panel.style.display === "") {{
                panel.style.display = "flex";
                btn.style.display = "none";
            }} else {{
                panel.style.display = "none";
                btn.style.display = "flex";
            }}
        }}
    </script>
    """
    components.html(widget_code, height=0)

st.title("📄 Question Paper & Floating Answer Key")

# File Uploaders
q_files = st.file_uploader(
    "1. Upload Question Pages", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)
ans_file = st.file_uploader("2. Upload Answer Key Page", type=["png", "jpg", "jpeg"])

if ans_file:
    if st.button("Extract Answer Key & Launch Floating Tool"):
        with st.spinner("Extracting Answer Key..."):
            ans_img = optimize_image(ans_file, max_dim=1800)
            key_prompt = """
            Extract all question numbers (1 to 100) and their single correct option (A, B, C, or D) from this answer key image.
            Ignore mathematical solutions, explanations, and formulas.
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

# When Answer Key is active
if st.session_state.answer_key:
    # Inject the floating widget onto the webpage
    inject_floating_widget(st.session_state.answer_key)
    
    st.subheader("💡 Floating Widget Active")
    st.info("Tap the blue round 📝 icon at the bottom-right of your screen to toggle the scrollable answer bar open and closed!")

    # Format text for Android external floating apps
    sorted_key = sorted(st.session_state.answer_key.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
    raw_text_key = "\n".join([f"Q.{q} : {ans}" for q, ans in sorted_key])
    
    st.download_button(
        label="📋 Download Key as Plain Text File (.txt)",
        data=raw_text_key,
        file_name="answer_key.txt",
        mime="text/plain"
    )

    with st.expander("Show Raw Answer Key List"):
        st.text_area("Answer Key Copy-Paste", raw_text_key, height=200)
