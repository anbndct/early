import streamlit as st
from streamlit_option_menu import option_menu
import nibabel as nib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import zipfile
import openpyxl

def download_and_extract_hf_zip(url, output_path="data"):
    zip_path = "temp_data.zip"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    os.remove(zip_path)

# Jalankan hanya jika folder belum ada
if not os.path.exists("data/rCMB_DefiniteSubject"):
    hf_url = "https://huggingface.co/datasets/anbndct/rcmb/resolve/main/rCMB_DefiniteSubject.zip"
    download_and_extract_hf_zip(hf_url)


# Page Styles
page_styles = {
    "Home": """
        <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(#D36BA3);
                color: white;
            }
        </style>
    """
}

# CSS Styling Global
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
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Cerebral Microbleeds Detection",
        ["Home", "Get to Know Microbleeds", "Ground Truth", "Chatbot", "Project FAQ"],
        default_index=0
    )

# Apply style
if selected in page_styles:
    st.markdown(page_styles[selected], unsafe_allow_html=True)

# Home Page
if selected == "Home":
    # Custom CSS for background and team card style
    st.markdown(
        """
        <style>
        .main {
            background-color: #A05278;
            padding: 2rem;
        }
        .team-card {
            background-color: white;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
        }
        .team-img {
            width: 90px;
            height: 90px;
            object-fit: cover;
            border-radius: 12px;
            margin-right: 1rem;
        }
        .team-info {
            font-size: 18px;
            color: #333;
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
        ul {
            padding-left: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown('<div class="section-title">⚙️ Langkah Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    1. Unggah citra MRI (SWI atau T2*-weighted)<br>
    2. Sistem akan melakukan pre-processing: <i>normalisasi, peningkatan kontras, brain masking</i>, dan <i>noise removal</i><br>
    3. Deteksi otomatis dilakukan<br>
    4. Hasil divisualisasikan secara interaktif
    </div>
    """, unsafe_allow_html=True)

    # CSS bulet + fix ukuran
    st.markdown("""
        <style>
            .circle-img {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                object-fit: cover;
                object-position: center;
            }
        </style>
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

        
# Ground Truth Page
elif selected == "Ground Truth":
    st.title("Ground Truth Visualization")

    if 'slice_idx' not in st.session_state:
        st.session_state.slice_idx = 0

    nii_folder = "data/rCMB_DefiniteSubject"
    excel_path = "rCMBInformationInfo.xlsx"

    try:
        df = pd.read_excel(excel_path)
        nii_files = df.iloc[:, 0].tolist()
        selected_file = st.selectbox("Select NIfTI file:", nii_files)

        nii_path = os.path.join(nii_folder, selected_file)
        img = nib.load(nii_path)
        data = img.get_fdata()
        row = df[df.iloc[:, 0] == selected_file].iloc[0].values[1:]

        cmb_slices = set()
        cmb_info = []
        cmb_count = 0
        for i in range(0, len(row), 3):
            coords = row[i:i+3]
            if pd.isna(coords).any() or any(str(x).strip() == '' for x in coords):
                continue
            try:
                z = int(float(coords[2]))
                cmb_count += 1
                cmb_slices.add(z)
                cmb_info.append((z, cmb_count))
            except (ValueError, TypeError):
                continue

        if cmb_slices:
            if 'prev_file' not in st.session_state or st.session_state.prev_file != selected_file:
                st.session_state.slice_idx = sorted(cmb_slices)[0]
                st.session_state.prev_file = selected_file
        else:
            st.session_state.slice_idx = data.shape[2] // 2

        if cmb_slices:
            st.write("**Quick navigation to microbleeds:**")
            slices_sorted = sorted(cmb_slices)
            cols_per_row = 4
            for row_idx in range(0, len(slices_sorted), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, z in enumerate(slices_sorted[row_idx:row_idx+cols_per_row]):
                    cmbs_on_slice = [c[1] for c in cmb_info if c[0] == z]
                    label = f"Slice {z} ({len(cmbs_on_slice)} CMB)"
                    with cols[col_idx]:
                        if st.button(label, key=f"cmb_nav_{z}"):
                            st.session_state.slice_idx = z
                            st.rerun()

        slice_idx = st.slider("Manual slice selection:", 0, data.shape[2] - 1, st.session_state.slice_idx, key="slice_slider")
        if slice_idx != st.session_state.slice_idx:
            st.session_state.slice_idx = slice_idx

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(data[:, :, st.session_state.slice_idx].T, cmap="gray", origin="lower")

        current_cmb_count = 0
        for i in range(0, len(row), 3):
            coords = row[i:i+3]
            if pd.isna(coords).any() or any(str(x).strip() == '' for x in coords):
                continue
            try:
                x, y, z = int(float(coords[0])), int(float(coords[1])), int(float(coords[2]))
                if z == st.session_state.slice_idx:
                    current_cmb_count += 1
                    rect = patches.Rectangle((x - 2, y - 2), 4, 4, linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x + 3, y - 3, f"CMB {current_cmb_count}", color='lime', fontsize=10, weight='bold')
            except (ValueError, TypeError):
                continue

        st.pyplot(fig)

        if cmb_slices:
            st.success(f"**File summary:** {cmb_count} microbleeds across {len(cmb_slices)} slice(s)")
            if st.session_state.slice_idx in cmb_slices:
                cmbs_here = [c[1] for c in cmb_info if c[0] == st.session_state.slice_idx]
                st.info(f"**Current slice {st.session_state.slice_idx}:** {len(cmbs_here)} microbleed(s) (CMB {', '.join(map(str, cmbs_here))})")
            else:
                st.warning("No microbleeds on current slice")
        else:
            st.warning("No microbleeds found in this file")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

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

    st.subheader("🧑‍⚕️ How Are They Detected?")
    st.write("""
    These microbleeds are typically found using:
    - **Susceptibility-Weighted Imaging (SWI)**
    - **Gradient-Recalled Echo (GRE) MRI**
    
    They appear as small, dark, round lesions—often located in deep or cortical brain regions.
    """)

    st.markdown("---")

    st.subheader("📸 Visual Example")
    st.image("https://www.ajnr.org/content/ajnr/32/6/1043/F1.large.jpg", caption="Source: AJNR, 2011")

    st.markdown("---")

    st.subheader("✨ Did You Know?")
    st.write("""
    Many individuals with CMBs are asymptomatic, and the microbleeds are only discovered incidentally 
    during imaging for unrelated conditions.
    """)

    st.markdown("---")

    st.subheader("📚 Learn More")
    st.write("""
    - [Cerebral Microbleeds: A Guide to Detection and Clinical Relevance (AJNR)](https://www.ajnr.org/content/32/6/1043)
    - [Cerebral Microbleeds: Imaging and Clinical Significance (RSNA)](https://pubs.rsna.org/doi/abs/10.1148/radiol.2018170803)
    - [Neuroimaging of CMBs (Radiopaedia)](https://radiopaedia.org/articles/cerebral-microbleeds)
    """)

    st.markdown("---")

    with st.expander("🎥 Optional: Watch a Lecture on CMBs"):
        st.write("""
        📺 Recommended segment: [2:36 to 17:00](https://www.youtube.com/watch?v=oSb_xKGhytY&t=156s)  
        This part covers how cerebral microbleeds appear on scans, their causes, and their clinical impact.
        """)
        st.video("https://www.youtube.com/watch?v=oSb_xKGhytY")


elif selected == "Chatbot":
    st.title("🧠 Cerebral Microbleeds Expert Chatbot")
    
    # Custom CSS untuk tampilan lebih baik
    st.markdown("""
    <style>
        /* Gaya untuk chat container */
        .stChatFloatingInputContainer {
            background-color: #f0f2f6;
        }
        /* Warna teks input */
        .stTextInput input {
            color: #333333 !important;
        }
        /* Gaya bubble chat */
        .stChatMessage {
            border-radius: 15px !important;
            padding: 12px !important;
            margin: 8px 0 !important;
        }
        /* Gaya khusus untuk asisten */
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Inisialisasi chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I'm Dr. Neuro, your AI specialist in cerebral microbleeds. How can I help you today?"
            }
        ]

    # Tampilkan chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input pengguna
    if prompt := st.chat_input("Ask about cerebral microbleeds..."):
        # Tambahkan ke history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate respons
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                from openai import OpenAI
                
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )

                # Format prompt untuk dokter spesialis
                system_prompt = """Anda adalah Dr. Neuro, ahli neurologi dan radiologi dengan spesialisasi cerebral microbleeds yang bisa menjawab dengan bahasa apa saja. 
                Berikan jawaban yang:
                - Singkat dan langsung (maksimal 3 paragraf)
                - Gunakan bahasa natural seperti sedang konsultasi
                - Fokus pada pertanyaan spesifik pasien
                - Sertakan poin-poin penting saja"""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ]
                
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-distill-llama-70b:free",
                    messages=messages,
                    temperature=0.7,
                    stream=True  # Untuk efek streaming
                )
                
                # Tampilkan respons secara bertahap
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Maaf, terjadi error: {str(e)}")
                full_response = "Maaf, saya sedang tidak bisa menjawab. Silakan coba lagi nanti."
                message_placeholder.markdown(full_response)
        
        # Tambahkan ke history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif selected == "Project FAQ":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">❓ Frequently Asked Questions (FAQ)</div>', unsafe_allow_html=True)

    faqs = {
        "Apa itu Cerebral Microbleeds (CMB)?":
            "CMB adalah lesi kecil pada otak akibat kebocoran darah mikro, biasanya terlihat pada MRI jenis SWI atau GRE.",

        "Mengapa penting mendeteksi CMB?":
            "Karena CMB dapat menjadi indikator awal penyakit seperti stroke, demensia, atau gangguan vaskular lainnya.",

        "Jenis citra MRI apa yang digunakan dalam proyek ini?":
            "Sistem ini dirancang untuk mendeteksi CMB dari citra MRI jenis SWI (Susceptibility-Weighted Imaging) atau T2*-weighted GRE.",

        "Apa itu 3D FCN dan 3D CNN dalam konteks proyek ini?":
            "3D FCN digunakan untuk menyaring kandidat CMB awal dari seluruh volume otak, sedangkan 3D CNN bertugas untuk memverifikasi kandidat tersebut sebagai CMB sejati.",

        "Bisakah sistem ini digunakan untuk citra pasien asli?":
            "Iya, selama citra yang digunakan kompatibel (misalnya format .nii/.nii.gz dengan SWI/GRE), sistem ini dapat digunakan untuk membantu deteksi awal CMB.",

        "Apakah sistem ini menggantikan diagnosis dokter?":
            "Tidak. Sistem ini hanya alat bantu deteksi awal. Diagnosis akhir tetap harus dikonfirmasi oleh radiolog atau dokter spesialis saraf.",

        "Apakah proyek ini open-source dan bisa dikembangkan?":
            "Ya, kode sumber proyek ini akan tersedia untuk umum dan bisa dikembangkan lebih lanjut untuk keperluan akademis dan riset."
    }

    for question, answer in faqs.items():
        with st.expander(f"❓ {question}"):
            st.markdown(f"<div class='section-text'>{answer}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
