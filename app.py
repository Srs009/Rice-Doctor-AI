import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Rice Doctor AI", page_icon="🌾")

# --- 2. THE CONSULTANT LOGIC (Research-Based) ---
rice_consultant = {
    "Brown Spot": {
        "type": "Fungal Infection (Cochliobolus miyabeanus)",
        "cause": "Often linked to Potassium (K) & Silicon (Si) deficiency.",
        "remedy": "Spray Propiconazole (1ml/liter).",
        "prevention": "Boost Potassium (K) to thicken cell walls. Apply Calcium Silicate."
    },
    "Leaf Blast": {
        "type": "Fungal Infection (Magnaporthe oryzae)",
        "cause": "Triggered by Excess Nitrogen (toxic).",
        "remedy": "Spray Tricyclazole 75 WP.",
        "prevention": "Avoid excess Urea. Split N application into 3-4 doses."
    },
    "Bacterial Leaf Blight": {
        "type": "Bacterial Infection (Xanthomonas oryzae)",
        "cause": "High Nitrogen + High Humidity.",
        "remedy": "Drain field. Spray Copper Oxychloride + Streptocycline.",
        "prevention": "Maintain proper drainage. Avoid late-stage Nitrogen."
    },
    "Tungro": {
        "type": "Viral Disease (Vector: Green Leafhopper)",
        "cause": "Spread by Leafhoppers.",
        "remedy": "Control vector with Imidacloprid.",
        "prevention": "Use resistant varieties (like IR36). Synchronous planting."
    },
    "Khaira": {
        "type": "Abiotic Stress (Micronutrient Deficiency)",
        "cause": "Zinc (Zn) Deficiency. Common in high pH soils.",
        "remedy": "Apply Zinc Sulphate (25kg/ha) + Urea spray.",
        "prevention": "Basal application of Zinc Sulphate. Green manuring (Dhaincha)."
    }
}

# --- 3. UI LAYOUT ---
st.title("🌾 Rice Doctor AI")
st.markdown("**Real-time Diagnostic System**")
st.info("💡 Point camera at leaf for instant analysis.")

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Looks for the model in the same folder
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model not found! Please run the 'Move Model' code first.")
    st.stop()

# --- 5. DIAGNOSTIC FUNCTION ---
def diagnose(image_source):
    st.image(image_source, caption="Analyzing Leaf...", use_column_width=True)
    
    with st.spinner("🤖 AI is analyzing symptoms..."):
        # Save temp file for YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_source.save(tmp.name)
            tmp_path = tmp.name

        # Run Inference
        results = model(tmp_path)
        os.remove(tmp_path) # Clean up
        
        # Extract Results
        probs = results[0].probs
        top_idx = probs.top1
        pred_name = results[0].names[top_idx]
        conf = probs.top1conf.item() * 100

    # Display Diagnosis
    st.success(f"**Diagnosis: {pred_name}** ({conf:.1f}%)")
    
    # Get Advice
    advice = rice_consultant.get(pred_name)
    if advice:
        with st.expander("📋 View Treatment Plan", expanded=True):
            st.markdown(f"**Type:** {advice['type']}")
            st.error(f"**❌ Root Cause:** {advice['cause']}")
            st.success(f"**💊 Remedy:** {advice['remedy']}")
            st.info(f"**🛡️ Prevention:** {advice['prevention']}")
    else:
        st.warning("No specific advice available.")

# --- 6. INPUT TABS (Camera vs Upload) ---
tab1, tab2 = st.tabs(["📸 Live Camera", "📂 Gallery Upload"])

with tab1:
    cam_img = st.camera_input("Tap to Snap")
    if cam_img:
        img = Image.open(cam_img)
        diagnose(img)

with tab2:
    up_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if up_file:
        img = Image.open(up_file)
        if st.button("Analyze Image"):
            diagnose(img)
