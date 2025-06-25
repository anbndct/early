import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import io
import tempfile

# Configuration
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/anbndct/NII_MRI_CMB/resolve/main"

# CSS Styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #D36BA3, #974578);
    color: white;
}
.circle-img {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center;
}
.section-title {
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-top: 2rem;
}
.section-headline {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
}
.section-subheadline {
    font-size: 20px;
    font-weight: 400;
    color: white;
    margin-bottom: 1.5rem;
}
.text-white {
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Cerebral Microbleeds Detection",
        ["Home", "Get to Know Microbleeds", "Ground Truth", "CMB Detection", "Chatbot", "Project FAQ"],
        default_index=0
    )

# Home Page
if selected == "Home":
    st.markdown('<div class="section-headline">🧠 Deteksi Otomatis Perdarahan Mikro Serebral</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheadline">Sistem Cerdas Berbasis Deep Learning untuk Citra MRI</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📍 Apa Itu Proyek Ini?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    Proyek ini bertujuan untuk mengembangkan metode <b>otomatis</b> dalam mendeteksi <i>Cerebral Microbleeds (CMB)</i> – lesi kecil yang mencerminkan mikropendarahan di otak.<br><br>
    Lesi CMB penting dalam diagnosis karena berhubungan dengan:
    <ul>
    <li>🧠 Penurunan fungsi kognitif</li>
    <li>🧓 Risiko stroke & demensia</li>
    <li>❤️ Penyakit neurodegeneratif dan vaskular</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔬 Bagaimana Sistem Ini Bekerja?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    Sistem ini menggunakan pendekatan <b>Two-Stage Cascaded Framework</b>:<br><br>
    1. <b>3D FCN</b> untuk menyaring kandidat CMB awal<br>
    2. <b>3D CNN</b> untuk memastikan deteksi hanya pada lesi CMB asli<br>
    3. <b>Post-processing & evaluasi</b> untuk menyempurnakan hasil
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎓 Tentang Kami</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<img src="https://i.imgur.com/ZPrugfv.jpg" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown("**👩‍🔬 Benedicta Sabdaningtyas Pratita Pratanjana**  \nMahasiswa Teknik Biomedik ITS")

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<img src="https://i.imgur.com/5WFquuD.jpg" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown("**👨‍🏫 Dr. Norma Hermawan, S.T., M.T., M.Sc.**  \nDosen Pembimbing I")

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<img src="https://i.imgur.com/sGTdZfZ.jpg" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown("**👨‍🏫 Prof. Dr. Tri Arief Sardjono, S.T., M.T.**  \nDosen Pembimbing II")

# Get to Know Microbleeds Page
elif selected == "Get to Know Microbleeds":
    st.title("🧠 Get to Know Microbleeds")

    st.markdown("---")

    st.subheader("What are Cerebral Microbleeds?")
    st.write("""
    Cerebral microbleeds (CMBs) are small, round areas of chronic blood leakage in the brain. 
    They are often invisible in conventional MRI but can be detected using susceptibility-based sequences.
    While they are small in size, they may reflect more significant underlying vascular damage.
    """)

    st.markdown("---")

    st.subheader("🔍 Why Do They Matter?")
    st.write("""
    CMBs are clinically significant as they are associated with:
    - **Traumatic brain injury (TBI)**
    - **Stroke and cerebrovascular disease**
    - **Cognitive decline and dementia**
    
    Studies show that their presence may increase the risk of future hemorrhages and neurodegeneration.
    """)

    st.markdown("---")

    st.subheader("🧬 What Causes Them?")
    st.write("""
    CMBs often result from weakened or damaged small blood vessels. Common contributing factors include:
    - **Hypertension**
    - **Cerebral amyloid angiopathy**
    - **Aging**
    - **Head trauma**
    """)

    st.markdown("---")

    st.subheader("📸 Visual Example")
    st.image("https://www.ajnr.org/content/ajnr/32/6/1043/F1.large.jpg", caption="Source: AJNR, 2011")

    st.markdown("---")

    st.subheader("📚 Learn More")
    st.write("""
    - [Cerebral Microbleeds: A Guide to Detection and Clinical Relevance (AJNR)](https://www.ajnr.org/content/32/6/1043)
    - [Cerebral Microbleeds: Imaging and Clinical Significance (RSNA)](https://pubs.rsna.org/doi/abs/10.1148/radiol.2018170803)
    - [Neuroimaging of CMBs (Radiopaedia)](https://radiopaedia.org/articles/cerebral-microbleeds)
    """)

# Ground Truth Page
elif selected == "Ground Truth":
    st.title("🎯 Ground Truth Visualization")
    st.markdown("**Interactive CMB Ground Truth Explorer**")
    
    st.info("🚧 This section requires nibabel and scipy libraries for MRI processing.")
    st.info("📊 Dataset: NII_MRI_CMB with 20 volumes (~315MB each)")
    
    st.markdown("""
    ### 📋 Ground Truth Features (Coming Soon):
    - Load MRI volumes from Hugging Face dataset
    - Interactive slice navigation
    - CMB highlighting with ground truth annotations
    - Multi-volume comparison
    
    **Technical Requirements:**
    - nibabel for NIfTI file processing
    - scipy for .mat file handling
    - Large file downloading capabilities
    """)
    
    # Demo visualization
    st.subheader("📊 Demo Visualization")
    
    # Create sample data
    np.random.seed(42)
    sample_slice = np.random.rand(256, 256) * 0.5
    
    # Add some "CMB-like" spots
    for i in range(3):
        x, y = np.random.randint(50, 206, 2)
        sample_slice[x-5:x+5, y-5:y+5] = 0.1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(sample_slice, cmap='gray')
    
    # Add sample CMB annotations
    cmb_coords = [(100, 120), (180, 80), (150, 200)]
    for i, (x, y) in enumerate(cmb_coords):
        rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x+8, y-8, f'CMB {i+1}', color='lime', fontsize=12, weight='bold')
    
    ax.set_title('Sample MRI Slice with CMB Annotations', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    st.pyplot(fig)

# CMB Detection Page
elif selected == "CMB Detection":
    st.title("🧠 CMB Detection Pipeline")
    st.markdown("**Two-Stage Cascaded Network: FCN + CNN**")
    
    st.info("🚧 Full detection pipeline requires TensorFlow and model loading capabilities.")
    
    st.markdown("""
    ### 🔄 Detection Workflow:
    1. **FCN Stage:** 3D FCN processes full volume to generate candidate locations
    2. **CNN Stage:** 3D CNN with SPP validates each candidate  
    3. **Visualization:** Interactive results with ground truth comparison
    
    ### 📊 Available Options:
    - **Pre-computed FCN Results:** Use existing FCN outputs from Hugging Face
    - **Upload Custom FCN Results:** Upload your own .npy files
    - **Real-time Inference:** Full pipeline processing (~45 minutes)
    
    ### 🤖 Model Information:
    - **FCN Model:** fcn_precision_focused_best.h5
    - **CNN Model:** stage2_cnn_final.h5 (with custom SPP layer)
    - **Source:** Available on Hugging Face dataset
    """)
    
    # Demo interface
    st.subheader("🎯 Detection Interface Demo")
    
    detection_mode = st.selectbox(
        "Select detection mode:",
        ["Pre-computed FCN Results", "Upload Custom FCN Results", "Real-time Inference"]
    )
    
    if detection_mode == "Pre-computed FCN Results":
        st.markdown("#### 📂 Pre-computed FCN Results")
        
        selection_mode = st.selectbox(
            "Volume selection mode:",
            ["Single Volume", "Multiple Volumes (Comma-separated)", "Volume Range", "All Volumes"]
        )
        
        if selection_mode == "Single Volume":
            volume_id = st.selectbox("Select volume:", list(range(1, 21)))
            st.info(f"Selected: Volume {volume_id}")
            
        elif selection_mode == "Multiple Volumes (Comma-separated)":
            vol_input = st.text_input("Enter volume IDs (e.g., 1,3,5,7):", "1,3,5")
            if vol_input:
                try:
                    volumes = [int(x.strip()) for x in vol_input.split(",")]
                    st.success(f"Selected volumes: {volumes}")
                except:
                    st.error("Invalid format. Use: 1,3,5,7")
        
        st.button("🚀 Start CNN Processing", type="primary", help="Requires TensorFlow and models")
    
    elif detection_mode == "Upload Custom FCN Results":
        st.markdown("#### 📤 Upload Custom FCN Results")
        
        uploaded_file = st.file_uploader("Upload FCN results (.npy)", type=['npy'])
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            st.info("File analysis and CNN processing would happen here.")
    
    else:  # Real-time Inference
        st.markdown("#### ⚡ Real-time Inference")
        st.warning("⏱️ Estimated processing time: ~45 minutes")
        
        volume_id = st.selectbox("Select volume for inference:", list(range(1, 21)))
        
        st.markdown("**Model Configuration:**")
        col1, col2 = st.columns(2)
        with col1:
            st.info("FCN: 16x16x10 patches, stride 4,4,2, batch 512")
        with col2:
            st.info("CNN: 20x20x16 patches with SPP layer")
        
        st.button("🚀 Start Full Pipeline", type="primary", help="Requires models and heavy processing")

# Chatbot Page
elif selected == "Chatbot":
    st.title("🧠 Cerebral Microbleeds Expert Chatbot")
    
    st.info("🚧 Chatbot requires OpenAI API configuration.")
    
    st.markdown("""
    ### 💬 AI Medical Assistant Features:
    - Expert knowledge on cerebral microbleeds
    - Bilingual support (English/Indonesian)
    - Medical consultation style responses
    - Real-time streaming answers
    
    **Technical Requirements:**
    - OpenAI API key configuration
    - Streamlit secrets management
    """)
    
    # Demo chat interface
    st.subheader("💬 Demo Chat Interface")
    
    # Simple demo without actual AI
    user_input = st.text_input("Ask about cerebral microbleeds:")
    
    if user_input:
        if "what" in user_input.lower() and "cmb" in user_input.lower():
            st.markdown("""
            **Dr. Neuro:** Cerebral microbleeds (CMBs) are small areas of chronic blood leakage in the brain, 
            typically 2-10mm in diameter. They appear as small, dark, round lesions on susceptibility-weighted 
            imaging (SWI) or gradient-recalled echo (GRE) MRI sequences.
            """)
        elif "cause" in user_input.lower():
            st.markdown("""
            **Dr. Neuro:** CMBs are commonly caused by:
            - Hypertensive vasculopathy
            - Cerebral amyloid angiopathy (CAA)
            - Traumatic brain injury
            - Age-related vascular changes
            """)
        else:
            st.markdown("""
            **Dr. Neuro:** Thank you for your question about cerebral microbleeds. 
            For a fully functional AI assistant, please configure the OpenAI API integration.
            """)

# Project FAQ Page
elif selected == "Project FAQ":
    st.title("❓ Frequently Asked Questions")

    faqs = {
        "Apa itu Cerebral Microbleeds (CMB)?":
            "CMB adalah lesi kecil pada otak akibat kebocoran darah mikro, biasanya terlihat pada MRI jenis SWI atau GRE.",

        "Mengapa penting mendeteksi CMB?":
            "Karena CMB dapat menjadi indikator awal penyakit seperti stroke, demensia, atau gangguan vaskular lainnya.",

        "Jenis citra MRI apa yang digunakan dalam proyek ini?":
            "Sistem ini dirancang untuk mendeteksi CMB dari citra MRI jenis SWI (Susceptibility-Weighted Imaging) atau T2*-weighted GRE.",

        "Apa itu 3D FCN dan 3D CNN dalam konteks proyek ini?":
            "3D FCN digunakan untuk menyaring kandidat CMB awal dari seluruh volume otak, sedangkan 3D CNN bertugas untuk memverifikasi kandidat tersebut sebagai CMB sejati.",

        "Bagaimana cara kerja Two-Stage Cascaded Framework?":
            "Stage 1 (FCN) memproses seluruh volume MRI untuk menemukan kandidat CMB. Stage 2 (CNN dengan SPP layer) memvalidasi setiap kandidat untuk mengurangi false positive.",

        "Berapa lama waktu processing untuk satu volume?":
            "Untuk real-time inference: ~45 menit untuk full pipeline. Menggunakan pre-computed FCN results: ~2-5 menit untuk CNN stage saja.",

        "Apakah sistem ini menggantikan diagnosis dokter?":
            "Tidak. Sistem ini hanya alat bantu deteksi awal. Diagnosis akhir tetap harus dikonfirmasi oleh radiolog atau dokter spesialis saraf.",

        "Apakah proyek ini open-source dan bisa dikembangkan?":
            "Ya, kode sumber proyek ini akan tersedia untuk umum dan bisa dikembangkan lebih lanjut untuk keperluan akademis dan riset."
    }

    for question, answer in faqs.items():
        with st.expander(f"❓ {question}"):
            st.markdown(f"<div class='text-white'>{answer}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
<p>🧠 Cerebral Microbleeds Detection System</p>
<p>Built with Streamlit | Powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)
