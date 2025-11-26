import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI

# --- Page Configuration and Global Styles ---
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
.section-title {
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.section-headline {
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.section-subheadline {
    font-size: 22px;
    font-weight: 400;
    color: white;
    margin-bottom: 2rem;
}
.text-white {
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Menu ---
with st.sidebar:
    selected = option_menu(
        "Cerebral Microbleeds",
        ["Home", "Get to Know Microbleeds", "CMB detection", "Chatbot", "Project FAQ"],
        icons=['house', 'search', 'box-arrow-up-right', 'chat-dots', 'question-circle'],
        menu_icon="activity",
        default_index=0
    )

# --- Page Routing ---

# Home Page
if selected == "Home":
    st.markdown('<div class="section-headline">🧠 Deteksi Otomatis Perdarahan Mikro Serebral</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheadline">Sistem Cerdas Berbasis Deep Learning untuk Citra MRI</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📍 Tentang Proyek Ini</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    Proyek ini bertujuan untuk mengembangkan metode <b>otomatis</b> dalam mendeteksi <i>Cerebral Microbleeds (CMB)</i>, yaitu lesi kecil yang mencerminkan mikropendarahan di otak. Deteksi dini CMB sangat penting karena berhubungan dengan:
    <ul>
        <li>🧠 Penurunan fungsi kognitif</li>
        <li>🧓 Peningkatan risiko stroke & demensia</li>
        <li>❤️ Indikasi penyakit neurodegeneratif dan vaskular</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔬 Cara Kerja Sistem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    Sistem ini menggunakan pendekatan <b>Two-Stage Cascaded Framework</b>:
    <ol>
        <li><b>3D FCN:</b> Menyaring kandidat awal CMB dari seluruh volume otak.</li>
        <li><b>3D CNN:</b> Memverifikasi kandidat untuk memastikan deteksi hanya pada lesi CMB asli.</li>
        <li><b>Post-processing:</b> Menyempurnakan hasil untuk akurasi maksimal.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎓 Tim Kami</div>', unsafe_allow_html=True)
    team_members = {
        "👩‍🔬 Benedicta Sabdaningtyas P. P.": ("Mahasiswa Teknik Biomedik ITS", "https://i.imgur.com/ZPrugfv.jpg"),
        "👨‍🏫 Dr. Norma Hermawan, S.T., M.T., M.Sc.": ("Dosen Pembimbing I", "https://i.imgur.com/5WFquuD.jpg"),
        "👨‍🏫 Prof. Dr. Tri Arief Sardjono, S.T., M.T.": ("Dosen Pembimbing II", "https://i.imgur.com/sGTdZfZ.jpg")
    }
    for member, (role, img_url) in team_members.items():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f'<img src="{img_url}" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{member}**\n<br>{role}", unsafe_allow_html=True)
        st.write("") # Spacer

# Get to Know Microbleeds Page
elif selected == "Get to Know Microbleeds":
    st.title("🧠 Mengenal Cerebral Microbleeds (CMB)")
    st.markdown("---")
    st.subheader("Apa itu Cerebral Microbleeds?")
    st.write("Cerebral Microbleeds (CMB) adalah deposit hemosiderin kecil di otak yang merupakan sisa dari perdarahan mikro. Lesi ini sering tidak terdeteksi pada MRI konvensional tetapi dapat diidentifikasi dengan jelas menggunakan sekuens yang sensitif terhadap suseptibilitas magnetik seperti SWI atau T2*-GRE.")
    st.markdown("---")
    st.subheader("🔍 Mengapa CMB Penting?")
    st.write("Meskipun ukurannya kecil, CMB adalah penanda klinis yang signifikan karena berasosiasi dengan: \n- **Cedera otak traumatis (TBI)** \n- **Penyakit serebrovaskular dan risiko stroke** \n- **Penurunan kognitif dan demensia**")
    st.markdown("---")
    st.subheader("🧬 Apa Penyebabnya?")
    st.write("CMB sering kali disebabkan oleh melemahnya pembuluh darah kecil di otak. Faktor risiko utamanya meliputi: \n- **Hipertensi (tekanan darah tinggi)** \n- **Angiopati amiloid serebral (CAA)** \n- **Proses penuaan alami**")
    st.markdown("---")
    st.subheader("🧑‍⚕️ Bagaimana Cara Mendeteksinya?")
    st.image("https://www.ajnr.org/content/ajnr/32/6/1043/F1.large.jpg", caption="Contoh visual CMB pada citra SWI. Sumber: AJNR, 2011")
    st.write("CMB paling baik dideteksi menggunakan sekuens MRI khusus seperti: \n- **Susceptibility-Weighted Imaging (SWI)** \n- **T2*-weighted Gradient-Recalled Echo (GRE)** \n\nPada citra ini, CMB tampak sebagai lesi hipointens (gelap) yang kecil dan berbentuk bulat.")
    st.markdown("---")
    st.subheader("📚 Pelajari Lebih Lanjut")
    st.write("""
    - [Cerebral Microbleeds: A Guide to Detection and Clinical Relevance (AJNR)](https://www.ajnr.org/content/32/6/1043)
    - [Cerebral Microbleeds: Imaging and Clinical Significance (RSNA)](https://pubs.rsna.org/doi/abs/10.1148/radiol.2018170803)
    """)
    with st.expander("🎥 Tonton Video Penjelasan (Inggris)"):
        st.video("https://youtu.be/oSb_xKGhytY?si=P-YowzwZHTcSSAoN", start_time=156) # Starts at 2:36
        st.caption("Segmen yang direkomendasikan: 2:36 hingga 17:00, membahas penampilan CMB pada scan, penyebab, dan dampak klinisnya.")

# CMB detection Page
elif selected == "CMB detection":
    st.title("🚀 Aplikasi Deteksi CMB")
    st.markdown(
        """
        <style>
        .link-button {
            display: inline-block;
            padding: 0.8rem 1.6rem;
            font-size: 1.1rem;
            font-weight: bold;
            color: white;
            background: #A05278;
            border-radius: 12px;
            text-align: center;
            text-decoration: none;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
            transition: all 0.3s ease;
            border: 2px solid white;
        }
        .link-button:hover {
            background-color: white;
            color: #A05278;
            text-decoration: none;
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size: 18px; margin-top: 1rem;">Klik tombol di bawah ini untuk membuka aplikasi deteksi CMB. Anda akan diarahkan ke aplikasi eksternal.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align: center; margin-top: 2rem;"><a href="https://streamicro.streamlit.app/" target="_blank" class="link-button">Buka Aplikasi Deteksi CMB</a></div>',
        unsafe_allow_html=True
    )
    st.info("Catatan: Aplikasi akan terbuka di tab browser baru untuk pengalaman terbaik.", icon="ℹ️")

# Chatbot Page
elif selected == "Chatbot":
    st.title("💬 Chatbot Ahli CMB")
    st.markdown("""
    <style>
        .stChatMessage { border-radius: 15px !important; padding: 14px !important; margin: 8px 0 !important; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya Dr. Neuro, AI spesialis Anda untuk Cerebral Microbleeds. Ada yang bisa saya bantu?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan tentang microbleeds..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                client = OpenAI("base_url=https://openrouter.ai/api/v1", api_key=st.secrets["OPENAI_API_KEY"])
                system_prompt = "Anda adalah Dr. Neuro, ahli neurologi dan radiologi dengan spesialisasi cerebral microbleeds. Jawablah dengan singkat, jelas (maksimal 3 paragraf), dan dalam bahasa natural seolah sedang berkonsultasi. Fokus pada pertanyaan spesifik pengguna."
                messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                response = client.chat.completions.create(
                    model="deepseek-r1t2-chimera-instruct",
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Maaf, terjadi kendala: {str(e)}. Silakan coba lagi nanti."
                message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Project FAQ Page
elif selected == "Project FAQ":
    st.title("❓ Frequently Asked Questions (FAQ)")
    faqs = {
        "Apa itu Cerebral Microbleeds (CMB)?": "CMB adalah lesi kecil di otak akibat kebocoran darah mikro dari pembuluh darah kecil, yang terlihat jelas pada citra MRI jenis SWI atau T2*-GRE.",
        "Mengapa deteksi CMB itu penting?": "Karena CMB adalah indikator kuat untuk berbagai penyakit serebrovaskular, termasuk peningkatan risiko stroke, demensia vaskular, dan penurunan kognitif.",
        "Jenis citra MRI apa yang digunakan?": "Sistem ini dirancang khusus untuk menganalisis citra MRI 3D jenis SWI (Susceptibility-Weighted Imaging) atau T2*-weighted GRE.",
        "Apa itu 3D FCN dan 3D CNN?": "3D FCN (Fully Convolutional Network) berfungsi sebagai penyaring kandidat untuk menemukan area yang berpotensi CMB. Kemudian, 3D CNN (Convolutional Neural Network) memverifikasi setiap kandidat untuk mengurangi hasil positif palsu.",
        "Apakah sistem ini menggantikan diagnosis dokter?": "Tidak. Sistem ini adalah alat bantu (decision support tool) untuk membantu radiolog dalam mendeteksi CMB secara lebih efisien. Diagnosis akhir tetap menjadi tanggung jawab tenaga medis profesional.",
        "Apakah proyek ini bersifat open-source?": "Ya, kode sumber proyek ini direncanakan untuk tersedia secara publik guna mendukung pengembangan lebih lanjut dalam riset akademis."
    }
    for question, answer in faqs.items():
        with st.expander(f"**{question}**"):
            st.write(answer)


