import streamlit as st
import numpy as np
import pandas as pd
import zipfile
import tempfile
import cv2
import os
from io import BytesIO

from PIL import Image

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.applications.xception import preprocess_input


# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Klasifikasi Penyakit Kulit",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>

/* Hilangkan Header Streamlit */
header[data-testid="stHeader"]{
    display:none;
}

/* Hilangkan Toolbar kanan atas */
[data-testid="stToolbar"]{
    display:none;
}

/* Hilangkan tombol Deploy */
[data-testid="stDecoration"]{
    display:none;
}

/* Hilangkan menu hamburger */
#MainMenu{
    visibility:hidden;
}

/* Hilangkan footer */
footer{
    visibility:hidden;
}

/* Hilangkan jarak atas */
.block-container{
    padding-top:1rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#f5f7fb;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e3a8a
    );
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Header Card */
.dashboard-header{
    background: linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );
    padding:25px;
    border-radius:20px;
    color:white;
    text-align:center;
    margin-bottom:20px;
    box-shadow:0 5px 20px rgba(0,0,0,0.15);
}

/* Statistic Card */
.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 3px 15px rgba(0,0,0,0.08);
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-5px);
}

/* Prediction Card */
.result-card{
    background:white;
    border-left:6px solid #2563eb;
    padding:20px;
    border-radius:15px;
    box-shadow:0 3px 15px rgba(0,0,0,0.08);
}

/* Footer */
.footer{
    text-align:center;
    padding:20px;
    color:gray;
    margin-top:40px;
}
            .card{
    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.1);
}
            .metric-card{
    background:white;
    padding:25px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 5px 20px rgba(0,0,0,0.08);
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-5px);
}

.result-card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 5px 20px rgba(0,0,0,0.08);
    height:100%;
}

</style>
""", unsafe_allow_html=True)
# =====================================================
# LOAD MODEL
# =====================================================

import os
import gdown

MODEL_PATH = "model_weights.weights.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=1JpOplrt29XE0U-HQfKWfR9csjJAFtjEc",
        MODEL_PATH,
        quiet=False
    )

@st.cache_resource(show_spinner=False)
def load_model():

    base_model = Xception(
        include_top=False,
        weights=None,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        Conv2D(
            64,
            (3,3),
            activation='relu',
            padding='same'
        ),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dropout(0.5),
        Dense(
            128,
            activation='relu'
        ),
        Dropout(0.3),
        Dense(
            5,
            activation='softmax'
        )
    ])

    model.load_weights(
        MODEL_PATH
    )

    return model

model = load_model()

# =====================================================
# SESSION STATE RIWAYAT
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

    # =====================================================
# SESSION LOGIN
# =====================================================

if "users" not in st.session_state:
    st.session_state.users = {
        "admin": "admin123"
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# =====================================================
# LABEL KELAS
# =====================================================

classes = [
    "Nail_psoriasis",
    "SJS-TEN",
    "Vitiligo",
    "acne",
    "hyperpigmentation"
]

# =====================================================
# LOGIN & REGISTER
# =====================================================


if not st.session_state.logged_in:

    st.markdown("""
    <style>

    [data-testid="stSidebar"]{
        display:none;
    }

    .hero{
        background:linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #1e3a8a,
            #2563eb
        );
        padding:50px;
        border-radius:30px;
        text-align:center;
        color:white;
        margin-bottom:30px;
        box-shadow:0 15px 40px rgba(0,0,0,0.25);
    }

    .hero h1{
        font-size:48px;
        font-weight:800;
        margin-bottom:10px;
    }

    .hero p{
        color:#dbeafe;
        font-size:18px;
    }

    .login-card{
        background:white;
        padding:35px;
        border-radius:25px;
        box-shadow:0 10px 30px rgba(0,0,0,0.15);
        border:1px solid #e5e7eb;
    }

    .info-card{
        background:linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        color:white;
        padding:20px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 8px 20px rgba(37,99,235,0.3);
    }

    .footer-login{
        text-align:center;
        margin-top:30px;
        color:#64748b;
    }

    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
<style>

[data-testid="stSidebar"]{
    display:none;
}

/* Background Halaman */
.stApp{
    background: linear-gradient(
        135deg,
        #dbeafe,
        #bfdbfe,
        #93c5fd
    );
}

/* Card Login */
.login-card{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #2563eb
    );
    padding:35px;
    border-radius:25px;
    box-shadow:0 15px 40px rgba(37,99,235,0.30);
    color:white;
}

/* Statistik */
.info-card{
    background:linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );
    color:white;
    padding:20px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 8px 20px rgba(37,99,235,0.25);
}

/* Footer */
.footer-login{
    text-align:center;
    margin-top:30px;
    color:#1e3a8a;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

    

    st.markdown("<br>", unsafe_allow_html=True)

    # FORM LOGIN

    kiri, tengah, kanan = st.columns([1.2,1.6,1.2])

    with tengah:

        st.markdown("""
        <div class="login-card">
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs([
            "🔑 Login",
            "📝 Register"
        ])

        # LOGIN

        with tab1:

            st.markdown("""
<div style="
    text-align:center;
    margin-bottom:25px;
">
    <h2 style="
        color:#0f172a;
        margin-bottom:5px;
        font-weight:800;
    ">
        SELAMAT DATANG
    </h2>
</div>
""", unsafe_allow_html=True)

            username = st.text_input(
                "Username",
                key="login_user"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_pass"
            )

            if st.button(
                "🔓 Login",
                use_container_width=True
            ):

                if (
                    username in st.session_state.users
                    and
                    st.session_state.users[username]
                    == password
                ):

                    st.session_state.logged_in = True
                    st.session_state.username = username

                    st.success(
                        f"Selamat datang {username}"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Username atau Password salah"
                    )

        # REGISTER

        with tab2:

            st.markdown("""
<div style="
    text-align:center;
    margin-bottom:25px;
">
    <h2 style="
        color:#0f172a;
        margin-bottom:5px;
        font-weight:800;
    ">
        BUAT AKUN BARU
    </h2>
</div>
""", unsafe_allow_html=True)

            new_user = st.text_input(
                "Username Baru"
            )

            new_pass = st.text_input(
                "Password Baru",
                type="password"
            )

            confirm_pass = st.text_input(
                "Konfirmasi Password",
                type="password"
            )

            if st.button(
                "📝 Register",
                use_container_width=True
            ):

                if new_user == "":

                    st.warning(
                        "Username wajib diisi"
                    )

                elif new_user in st.session_state.users:

                    st.error(
                        "Username sudah digunakan"
                    )

                elif new_pass != confirm_pass:

                    st.error(
                        "Password tidak sama"
                    )

                else:

                    st.session_state.users[
                        new_user
                    ] = new_pass

                    st.success(
                        "Registrasi berhasil"
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    

    st.stop()


    

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
<div style="
    text-align:center;
    padding:15px;
    border-radius:12px;
    background:rgba(255,255,255,0.1);
    margin-bottom:20px;
">
    <h2 style="
        color:white;
        margin:0;
        font-size:24px;
    ">
        HALAMAN MENU
    </h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="
    background:rgba(255,255,255,0.15);
    padding:15px;
    border-radius:15px;
    text-align:center;
    margin-bottom:15px;
">
    👤 <b>{st.session_state.username}</b>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Beranda",
        "Prediksi Gambar",
        "Riwayat Prediksi",
        "Tentang Model"
    ]
)

st.sidebar.markdown("""
<style>

/* Tombol Logout */
div[data-testid="stSidebar"] .stButton > button{
    background: linear-gradient(
        135deg,
        #ef4444,
        #dc2626
    );
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    height: 45px;
    transition: 0.3s;
}

div[data-testid="stSidebar"] .stButton > button:hover{
    background: linear-gradient(
        135deg,
        #dc2626,
        #b91c1c
    );
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(220,38,38,0.3);
}

</style>
""", unsafe_allow_html=True)
if st.sidebar.button(
    "Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.session_state.username = ""

    st.rerun()

if menu == "Beranda":

    # ==========================================
    # HERO SECTION
    # ==========================================

    st.markdown("""
    <div style="
        background: linear-gradient(
        180deg,
        #0f172a,
        #1e3a8a
    );
        padding:40px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0 8px 25px rgba(0,0,0,0.15);
        margin-bottom:25px;
    ">
        <h1>SISTEM KLASIFIKASI PENYAKIT KULIT</h1>
        <h3>Deep Learning menggunakan Arsitektur Xception</h3>
        <p>
        Sistem cerdas untuk mengklasifikasikan jenis penyakit kulit
        berdasarkan citra digital secara otomatis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # STATISTIK
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h1>5</h1>
            <p>Jumlah Kelas</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h1>Xception</h1>
            <p>Model CNN</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h1>224×224</h1>
            <p>Ukuran Input</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h1>AI</h1>
            <p>Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # DESKRIPSI SISTEM
    # ==========================================

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        <div class="result-card">
            <h3>TENTANG SISTEM</h3>

            Sistem ini dirancang untuk membantu proses
            identifikasi penyakit kulit menggunakan
            teknologi Deep Learning berbasis
            Transfer Learning Xception.

            Pengguna dapat mengunggah satu gambar
            maupun satu dataset gambar sekaligus,
            kemudian sistem akan melakukan proses
            klasifikasi secara otomatis dan
            menampilkan hasil prediksi beserta
            tingkat confidence dari model.

            Sistem ini dikembangkan untuk penelitian
            klasifikasi penyakit kulit menggunakan
            citra digital dengan akurasi tinggi.
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="result-card">
            <h3>KELAS PENYAKIT</h3>

            ✅ Nail Psoriasis
            ✅ SJS-TEN
            ✅ Vitiligo
            ✅ Acne
            ✅ Hyperpigmentation

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # ALUR SISTEM
    # ==========================================

    st.subheader("ALUR PENGGUNAAN SISTEM")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("""
        📤

        Upload
        Gambar
        """)

    with c2:
        st.info("""
        🧠

        Preprocessing
        Citra
        """)

    with c3:
        st.info("""
        🤖

        Klasifikasi
        Xception
        """)

    with c4:
        st.info("""
        📊

        Hasil
        Klasifikasi
        """)

    st.markdown("---")

    # ==========================================
    # FITUR SISTEM
    # ==========================================

    st.subheader("FITUR SISTEM")

    fitur1, fitur2, fitur3 = st.columns(3)

    with fitur1:
        st.success("""
        🔍 Klasifikasi Satu Gambar

        Mengklasifikasikan penyakit kulit
        dari satu gambar yang
        diunggah pengguna.
        """)

    with fitur2:
        st.success("""
        📁 Klasifikasi Dataset

        Mengklasifikasikan
        ratusan gambar sekaligus
        melalui file ZIP.
        """)

    with fitur3:
        st.success("""
        📜 Riwayat Klasifikasi

        Menyimpan hasil klasifikasi
        yang telah dilakukan.
        """)

    st.markdown("""
    <div class="footer">
        © 2026 Sistem Klasifikasi Penyakit Kulit
        <br>
        Deep Learning - Xception Transfer Learning
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PREDIKSI GAMBAR
# =====================================================

elif menu == "Prediksi Gambar":

    st.markdown("""
    <div style="
        background: linear-gradient(
        180deg,
        #0f172a,
        #1e3a8a
    );
        padding:40px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0 8px 25px rgba(0,0,0,0.15);
        margin-bottom:25px;
    ">
        <h1>PREDIKSI GAMBAR PENYAKIT KULIT</h1>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Gambar Input",
                use_container_width=True
            )

        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(
            img_array
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        pred_idx = np.argmax(
            prediction
        )

        confidence = (
            np.max(prediction) * 100
        )

        # ==========================
        # SIMPAN RIWAYAT
        # ==========================

        history_data = {
            "Nama File": uploaded_file.name,
            "Prediksi": classes[pred_idx],
            "Confidence (%)": round(confidence, 2),
            "Gambar": image.copy()
        }

        if (
            len(st.session_state.history) == 0
            or
            st.session_state.history[-1] != history_data
        ):
            st.session_state.history.append(
                history_data
            )

        with col2:

            st.success(
                f"Hasil Prediksi : {classes[pred_idx]}"
            )

            st.info(
                f"Confidence : {confidence:.2f}%"
            )

            st.subheader(
                "Probabilitas Kelas"
            )

            for i, label in enumerate(classes):

                st.write(
                    f"{label} : "
                    f"{prediction[0][i] * 100:.2f}%"
                )


# =====================================================
# RIWAYAT PREDIKSI
# =====================================================

elif menu == "Riwayat Prediksi":

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0f172a,#1e40af);
        padding:35px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0 10px 30px rgba(0,0,0,0.15);
        margin-bottom:25px;
    ">
        <h1 style="margin:0;">
            RIWAYAT PREDIKSI GAMBAR
        </h1>
        <p style="margin-top:10px;">
            Data hasil klasifikasi penyakit kulit
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # CUSTOM CSS
    # ==========================================

    st.markdown("""
    <style>

    .metric-card{
        background: linear-gradient(135deg,#0f172a,#1e3a8a);
        padding:20px;
        border-radius:20px;
        color:white;
        text-align:center;
        box-shadow:0 6px 20px rgba(0,0,0,0.12);
    }

    .history-card{
        background:white;
        border-radius:20px;
        padding:20px;
        box-shadow:0 8px 20px rgba(0,0,0,0.08);
        border-left:6px solid #2563eb;
    }

    .badge{
        background:#dbeafe;
        color:#1e40af;
        padding:8px 15px;
        border-radius:30px;
        font-weight:bold;
        display:inline-block;
    }

    .high{
        color:#16a34a;
        font-weight:bold;
        font-size:18px;
    }

    .medium{
        color:#ca8a04;
        font-weight:bold;
        font-size:18px;
    }

    .low{
        color:#dc2626;
        font-weight:bold;
        font-size:18px;
    }

    </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # MEMBERSIHKAN DATA LAMA
    # ==========================================

    cleaned_history = []

    for item in st.session_state.history:

        if not isinstance(item, dict):
            continue

        cleaned_history.append({
            "Nama File": item.get("Nama File", "-"),
            "Prediksi": item.get("Prediksi", "-"),
            "Confidence (%)": item.get("Confidence (%)", 0),
            "Gambar": item.get("Gambar", None)
        })

    st.session_state.history = cleaned_history

    # ==========================================
    # CEK DATA
    # ==========================================

    if len(st.session_state.history) == 0:

        st.warning(
            "Belum ada riwayat prediksi."
        )

    else:

        # ======================================
        # DATAFRAME
        # ======================================

        history_df = pd.DataFrame([
            {
                "Nama File": item.get(
                    "Nama File", "-"
                ),
                "Prediksi": item.get(
                    "Prediksi", "-"
                ),
                "Confidence (%)": item.get(
                    "Confidence (%)", 0
                )
            }
            for item in st.session_state.history
        ])

        # ======================================
        # METRIC CARD
        # ======================================

        st.markdown("### ✅ STATISTIK PREDIKSI")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <h1>{len(history_df)}</h1>
                <p>Total Prediksi</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="metric-card">
                <h1>
                {history_df['Prediksi'].nunique()}
                </h1>
                <p>Jumlah Kelas</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class="metric-card">
                <h1>
                {history_df['Confidence (%)'].mean():.1f}%
                </h1>
                <p>Rata-rata Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ======================================
        # DAFTAR RIWAYAT
        # ======================================

        st.markdown(
            "### ✅ DAFTAR RIWAYAT PREDIKSI"
        )

        for no, item in enumerate(
            reversed(st.session_state.history),
            start=1
        ):

            confidence = float(
                item.get(
                    "Confidence (%)",
                    0
                )
            )

            if confidence >= 80:
                conf_class = "high"
            elif confidence >= 60:
                conf_class = "medium"
            else:
                conf_class = "low"

            col1, col2 = st.columns(
                [1, 3]
            )

            with col1:

                gambar = item.get(
                    "Gambar",
                    None
                )

                if gambar is not None:

                    st.image(
                        gambar,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Tidak ada gambar"
                    )

            with col2:

                st.markdown(
                    f"""
                    <div class="history-card">

                    <b>
                    {item.get('Nama File','-')}
                    </b>
                    <br><br>

                    <b>Prediksi :
                    {item.get('Prediksi','-')}
                    </b>

                    <b>Confidence : <span class="{conf_class}">
                    {confidence:.2f}%
                    </span></b>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

        # ======================================
        # EXPORT EXCEL
        # ======================================

        export_df = pd.DataFrame([
            {
                "Nama File": item.get(
                    "Nama File", "-"
                ),
                "Prediksi": item.get(
                    "Prediksi", "-"
                ),
                "Confidence (%)": item.get(
                    "Confidence (%)", 0
                )
            }
            for item in st.session_state.history
        ])

        excel_file = (
            "riwayat_prediksi.xlsx"
        )

        export_df.to_excel(
            excel_file,
            index=False
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            with open(
                excel_file,
                "rb"
            ) as file:

                st.download_button(
                    label="📥 Download Riwayat Excel",
                    data=file,
                    file_name="riwayat_prediksi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        with col2:

            if st.button(
                "🗑 Hapus Semua Riwayat",
                use_container_width=True
            ):

                st.session_state.history = []

                st.rerun()
# =====================================================
# UPLOAD DATASET SUPER FAST
# =====================================================

elif menu == "Upload Dataset":

    import cv2
    import tensorflow as tf

    st.markdown("""
    <div style="
        background: linear-gradient(
        180deg,
        #0f172a,
        #1e3a8a
    );
        padding:40px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0 8px 25px rgba(0,0,0,0.15);
        margin-bottom:25px;
    ">
        <h1>KLASIFIKASI DATASET GAMBAR PENYAKIT KULIT</h1>
    </div>
    """, unsafe_allow_html=True)

    zip_file = st.file_uploader(
        "Upload ZIP Dataset",
        type=["zip"]
    )

    if zip_file is not None:

        tf.config.optimizer.set_jit(True)

        with st.spinner(
            "Membaca dataset..."
        ):

            image_arrays = []
            filenames = []

            with zipfile.ZipFile(
                zip_file,
                "r"
            ) as z:

                image_list = [

                    file for file in z.namelist()

                    if file.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png"
                        )
                    )
                ]

                total_images = len(
                    image_list
                )

                if total_images == 0:

                    st.error(
                        "Tidak ditemukan gambar."
                    )

                    st.stop()

                st.success(
                    f"{total_images} gambar ditemukan"
                )

                progress = st.progress(0)

                for idx, file in enumerate(
                    image_list
                ):

                    try:

                        image_bytes = z.read(
                            file
                        )

                        npimg = np.frombuffer(
                            image_bytes,
                            np.uint8
                        )

                        img = cv2.imdecode(
                            npimg,
                            cv2.IMREAD_COLOR
                        )

                        img = cv2.cvtColor(
                            img,
                            cv2.COLOR_BGR2RGB
                        )

                        img = cv2.resize(
                            img,
                            (224,224),
                            interpolation=cv2.INTER_AREA
                        )

                        image_arrays.append(
                            img
                        )

                        filenames.append(
                            os.path.basename(
                                file
                            )
                        )

                        # update tiap 50 gambar
                        if idx % 50 == 0:

                            progress.progress(
                                (idx+1)
                                /
                                total_images
                            )

                    except:

                        continue

        # ==========================
        # NUMPY ARRAY
        # ==========================

        with st.spinner(
            "Preprocessing..."
        ):

            images = np.asarray(
                image_arrays,
                dtype=np.float32
            )

            images = preprocess_input(
                images
            )

        # ==========================
        # TF DATASET SUPER CEPAT
        # ==========================

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(images)
            .batch(256)
            .prefetch(
                tf.data.AUTOTUNE
            )
        )

        # ==========================
        # PREDIKSI
        # ==========================

        with st.spinner(
            "Melakukan klasifikasi..."
        ):

            predictions = model.predict(
                dataset,
                verbose=0
            )

        # ==========================
        # HASIL
        # ==========================

        pred_index = np.argmax(
            predictions,
            axis=1
        )

        confidence = (
            np.max(
                predictions,
                axis=1
            ) * 100
        )

        hasil_df = pd.DataFrame({

            "Nama File":
            filenames,

            "Prediksi":
            [
                classes[i]
                for i in pred_index
            ],

            "Confidence (%)":
            np.round(
                confidence,
                2
            )

        })

        st.success(
            "✅ Analisis selesai!"
        )

        # ==========================
        # METRIC
        # ==========================

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "Total Gambar",
            len(hasil_df)
        )

        col2.metric(
            "Jumlah Kelas",
            hasil_df[
                "Prediksi"
            ].nunique()
        )

        col3.metric(
            "Rata-rata Confidence",
            f"{hasil_df['Confidence (%)'].mean():.2f}%"
        )

        col4.metric(
            "Kelas Dominan",
            hasil_df[
                "Prediksi"
            ].mode()[0]
        )

        # ==========================
        # TABEL
        # ==========================

        st.subheader(
            "Hasil Klasifikasi"
        )

        st.dataframe(
            hasil_df,
            use_container_width=True,
            height=500
        )

        # ==========================
        # GRAFIK
        # ==========================

        st.subheader(
            "Distribusi Prediksi"
        )

        distribusi = (
            hasil_df[
                "Prediksi"
            ]
            .value_counts()
        )

        st.bar_chart(
            distribusi
        )

        # ==========================
        # EXPORT EXCEL
        # ==========================

        excel_file = (
            "hasil_klasifikasi.xlsx"
        )

        hasil_df.to_excel(
            excel_file,
            index=False
        )

        with open(
            excel_file,
            "rb"
        ) as f:

            st.download_button(
                label="📥 Download Hasil Excel",
                data=f,
                file_name="hasil_klasifikasi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# =====================================================
# TENTANG MODEL
# =====================================================

elif menu == "Tentang Model":

    st.markdown("""
    <style>

    .about-header{
        background:linear-gradient(
            135deg,
            #0f172a,
            #1e3a8a,
            #2563eb
        );
        padding:40px;
        border-radius:30px;
        text-align:center;
        color:white;
        box-shadow:0 15px 35px rgba(37,99,235,0.25);
        margin-bottom:30px;
    }

    .about-header h1{
        margin:0;
        font-size:42px;
        font-weight:800;
    }

    .about-header p{
        margin-top:10px;
        color:#dbeafe;
        font-size:18px;
    }

    .info-card{
        background:white;
        border-radius:25px;
        padding:25px;
        box-shadow:0 8px 20px rgba(0,0,0,0.08);
        border:1px solid #e5e7eb;
        margin-bottom:20px;
    }

    .metric-card{
        background:linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        padding:25px;
        border-radius:25px;
        color:white;
        text-align:center;
        box-shadow:0 10px 25px rgba(37,99,235,0.25);
    }

    .metric-card h2{
        margin:0;
        font-size:35px;
        font-weight:800;
    }

    .metric-card p{
        margin-top:10px;
        color:#dbeafe;
    }

    .disease-card{
        background:#f8fafc;
        border-left:5px solid #2563eb;
        padding:15px;
        border-radius:15px;
        margin-bottom:10px;
        font-size:16px;
        font-weight:600;
    }

    .framework{
        background:#eff6ff;
        padding:12px;
        border-radius:12px;
        margin-bottom:10px;
        font-weight:600;
        color:#1e40af;
    }

    </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # HEADER
    # ==========================================

    st.markdown("""
    <div class="about-header">
        <h1>TENTANG MODEL AI</h1>
        <p>
        Sistem Klasifikasi Penyakit Kulit Menggunakan
        Deep Learning Xception Transfer Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # STATISTIK MODEL
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>224×224</h2>
            <p>Ukuran Input</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>5</h2>
            <p>Kelas Penyakit</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Xception</h2>
            <p>Arsitektur Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # INFORMASI MODEL
    # ==========================================

    st.markdown("""
    <div class="info-card">

    <h3>📌 INFORMASI MODEL</h3>

    <hr>

    <b>Arsitektur Model :</b><br>
    Xception Transfer Learning

    <br><br>

    <b>Ukuran Input :</b><br>
    224 × 224 Pixel

    <br><br>

    <b>Jumlah Kelas :</b><br>
    5 Kelas Penyakit Kulit

    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # KELAS PENYAKIT
    # ==========================================

    st.markdown("### 🩺 KELAS PENYAKIT YANG DI KLASIFIKASIKAN")

    penyakit = [
        "Nail Psoriasis",
        "SJS-TEN",
        "Vitiligo",
        "Acne",
        "Hyperpigmentation"
    ]

    for item in penyakit:
        st.markdown(
            f"""
            <div class="disease-card">
            🔬 {item}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # FRAMEWORK
    # ==========================================

    st.markdown("### ⚙️ TEKNOLOGI YANG DIGUNAKAN")

    frameworks = [
        "TensorFlow",
        "Keras",
        "Streamlit"
    ]

    for fw in frameworks:
        st.markdown(
            f"""
            <div class="framework">
            🚀 {fw}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # STATUS MODEL
    # ==========================================

    st.markdown("""
    <div style="
        background:linear-gradient(
            135deg,
            #16a34a,
            #22c55e
        );
        padding:20px;
        border-radius:20px;
        color:white;
        text-align:center;
        font-size:18px;
        font-weight:bold;
        box-shadow:0 8px 20px rgba(34,197,94,0.25);
    ">
        ✅ Model Deep Learning Berhasil Dimuat dan Siap Digunakan
    </div>
    """, unsafe_allow_html=True)
