import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model_def import SimpleCNN
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Bahasa Indonesia",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom untuk meniru tampilan HTML Anda
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 16px;
        background-color: #f8fafc;
        margin-top: 20px;
        border-left: 4px solid;
    }
    .token-pill {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 9999px;
        margin: 4px;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .token-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .floating-btn {
        background: linear-gradient(to right, #4f46e5, #6366f1);
        color: white;
        padding: 12px 24px;
        border-radius: 9999px;
        font-weight: 600;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
        border: none;
        transition: all 0.3s ease;
    }
    .floating-btn:hover {
        box-shadow: 0 15px 30px -5px rgba(79, 70, 229, 0.5);
        transform: translateY(-2px);
    }
    .progress-container {
        height: 20px;
        background-color: #e5e7eb;
        border-radius: 9999px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    .progress-bar {
        height: 100%;
        border-radius: 9999px;
        text-align: right;
        padding-right: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        transition: width 1s ease-in-out;
    }
    .positive { background-color: #10b981; }
    .neutral { background-color: #f59e0b; }
    .negative { background-color: #ef4444; }
    
    /* Animasi */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    .animate-float {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 1.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# Header dengan banner (dalam Streamlit kita menggunakan columns dan markdown)
st.markdown("""
<div class="animate-fade-in">
    <div style="background: linear-gradient(to right, #4f46e5, #6366f1); color: white; padding: 3rem 1rem; border-radius: 0 0 24px 24px; text-align: center; margin-bottom: 2rem;">
        <div class="animate-float" style="font-size: 4rem;">📊</div>
        <h1 style="font-size: 3rem; font-weight: 800; margin: 0.5rem 0;">Dashboard Analisis Sentimen</h1>
        <p style="font-size: 1.25rem; opacity: 0.9;">Lihat hasil analisis sentimen secara langsung dengan visual yang interaktif dan mudah dipahami</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model dan resources (dengan caching untuk performa)
@st.cache_resource
def load_models():
    """Muat semua model dan tokenizer yang diperlukan"""
    # Muat tokenizer dan model BERT
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
    
    # Muat model CNN
    model = SimpleCNN(kernel_size=3)
    model.load_state_dict(torch.load("cnn_model_k3.pth", map_location="cpu"))
    model.eval()
    
    # Initialize stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    return tokenizer, bert_model, model, stemmer

# Fungsi preprocessing
def preprocess_text(text, stemmer):
    """Lakukan preprocessing teks: cleaning, case folding, tokenization, stemming"""
    # Cleaning
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Case folding
    case_folded = cleaned_text.lower()
    
    # Tokenization
    tokens = case_folded.split()
    
    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return {
        'original': text,
        'cleaned': cleaned_text,
        'case_folded': case_folded,
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'stemmed_text': ' '.join(stemmed_tokens)
    }

# Fungsi untuk mendapatkan embedding BERT
def get_bert_embedding(text, tokenizer, bert_model):
    """Dapatkan embedding dari IndoBERT untuk teks input"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state

# Fungsi klasifikasi
def classify_text(text, tokenizer, bert_model, model):
    """Lakukan klasifikasi sentimen pada teks"""
    embedding = get_bert_embedding(text, tokenizer, bert_model)
    output = model(embedding)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs).item()
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    
    processed_embedding = torch.mean(embedding, dim=1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    token_embeddings = embedding.squeeze(0).tolist()
    
    return label_map[pred_class], probs.squeeze().tolist(), processed_embedding, tokens, token_embeddings

# Muat model sekali saja
try:
    tokenizer, bert_model, model, stemmer = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Container untuk form input
with st.container():
    st.markdown("""
    <div style="background: white; border-radius: 24px; padding: 2rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
        <h2 class="main-header">Analisis Sentimen Program Naturalisasi Timnas Indonesia</h2>
    """, unsafe_allow_html=True)
    
    # Input teks dari pengguna
    input_text = st.text_area(
        "Masukkan teks dalam Bahasa Indonesia...",
        height=150,
        max_chars=1000,
        key="text_input",
        help="Maksimal 1000 karakter"
    )
    
    # Character count
    char_count = len(input_text)
    st.caption(f"{char_count}/1000 karakter")
    
    # Tombol untuk melakukan prediksi
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button(
            "Analisis Sentimen", 
            type="primary", 
            use_container_width=True,
            disabled=char_count == 0
        )

# Tampilkan spinner saat memproses
if analyze_btn and input_text.strip():
    with st.spinner("Sedang menganalisis..."):
        # Preprocessing
        preprocessing_steps = preprocess_text(input_text, stemmer)
        
        # Classification
        label, probs, embedding, tokens, token_embeddings = classify_text(
            preprocessing_steps['stemmed_text'], tokenizer, bert_model, model
        )
        
        # Simpan hasil di session state
        st.session_state.result = {
            'label': label,
            'probs': probs,
            'embedding': embedding,
            'tokens': tokens,
            'token_embeddings': token_embeddings,
            'preprocessing': preprocessing_steps
        }

# Tampilkan hasil jika ada
if 'result' in st.session_state:
    result = st.session_state.result
    label = result['label']
    probs = result['probs']
    
    # Container untuk hasil analisis
    with st.container():
        st.markdown("""
        <div style="background: white; border-radius: 24px; padding: 2rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
            <h2 style="font-size: 1.5rem; font-weight: 700; color: #4f46e5; border-bottom: 2px solid #eef2ff; padding-bottom: 1rem; margin-bottom: 1.5rem;">
                📊 Hasil Analisis Sentimen
            </h2>
        """, unsafe_allow_html=True)
        
        # Tampilkan label sentimen
        emoji = "😠" if label == "Negatif" else "😐" if label == "Netral" else "😊"
        bg_color = "bg-red-100" if label == "Negatif" else "bg-yellow-100" if label == "Netral" else "bg-green-100"
        text_color = "text-red-800" if label == "Negatif" else "text-yellow-800" if label == "Netral" else "text-green-800"
        border_color = "border-red-500" if label == "Negatif" else "border-yellow-500" if label == "Netral" else "border-green-500"
        
        st.markdown(f"""
        <div class="result-box {bg_color} {text_color} {border_color}">
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                <span style="font-size: 2rem;">{emoji}</span>
                <span style="font-size: 1.5rem; font-weight: 700;">{label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan progress bar untuk setiap sentimen
        st.markdown("### Probabilitas Sentimen")
        
        labels = ["Negatif", "Netral", "Positif"]
        colors = ["negative", "neutral", "positive"]
        
        for i, (lbl, color) in enumerate(zip(labels, colors)):
            prob_percent = probs[i] * 100
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="progress-container">
                    <div class="progress-bar {color}" style="width: {prob_percent}%">
                        {prob_percent:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{lbl}**")
        
        # Expander untuk detail preprocessing
        with st.expander("🔍 Lihat Detail Preprocessing", expanded=False):
            st.markdown("### Proses Preprocessing Teks")
            
            st.markdown("**Teks Asli**")
            st.code(result['preprocessing']['original'], language='text')
            
            st.markdown("**Cleaning**")
            st.code(result['preprocessing']['cleaned'], language='text')
            
            st.markdown("**Case Folding**")
            st.code(result['preprocessing']['case_folded'], language='text')
            
            st.markdown("**Tokenisasi**")
            st.write(f"Total Token: {len(result['preprocessing']['tokens'])}")
            tokens_html = "".join([f'<span class="token-pill">{token}</span>' for token in result['preprocessing']['tokens']])
            st.markdown(f'<div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">{tokens_html}</div>', unsafe_allow_html=True)
            
            st.markdown("**Stemming**")
            st.write(f"Stemmed Text: \"{result['preprocessing']['stemmed_text']}\"")
            stemmed_tokens_html = "".join([f'<span class="token-pill" style="background-color: #dcfce7; color: #166534;">{token}</span>' for token in result['preprocessing']['stemmed_tokens']])
            st.markdown(f'<div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">{stemmed_tokens_html}</div>', unsafe_allow_html=True)
        
        # Expander untuk embedding details
        with st.expander("🧠 Lihat Detail Embedding", expanded=False):
            st.markdown("### Embedding IndoBERT")
            st.markdown("Representasi vektor 768 dimensi (average pooling)")
            
            st.json(result['embedding'])
            
            st.markdown("### Token dan Embedding")
            st.markdown("Menampilkan 10 dimensi pertama dari masing-masing token")
            
            # Tampilkan tabel token dan embedding
            data = []
            for i, (token, emb) in enumerate(zip(result['tokens'], result['token_embeddings'])):
                if i >= 10:  # Batasi jumlah token yang ditampilkan
                    break
                short_emb = [round(float(x), 4) for x in emb[:10]]
                data.append({"Token": token, "Embedding (10 Dimensi Pertama)": str(short_emb) + " ..."})
            
            st.table(data)
    
    # Footer
    st.markdown("---")
    current_year = datetime.now().year
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 1rem;">
        <p>© {current_year} Sentiment Analysis Dashboard</p>
        <p>Dibuat oleh <span style="color: #4f46e5; font-weight: 600;">Arya Vidyananta</span></p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar dengan informasi tentang model
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(to bottom, #4f46e5, #6366f1); color: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Tentang Model</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    - **Arsitektur**: SimpleCNN dengan kernel size 3
    - **Embedding**: IndoBERT Base P1
    - **Kelas**: Negatif, Netral, Positif
    - **Preprocessing**: Cleaning, case folding, tokenization, dan stemming
    """)
    
    st.markdown("---")
    
    st.markdown("**Detail Teknis:**")
    st.markdown("- Framework: PyTorch")
    st.markdown("- Library: Transformers, Sastrawi")
    st.markdown("- Deployment: Streamlit")
    
    st.markdown("---")
    
    # Tampilkan resource usage
    st.markdown("**Penggunaan Resource:**")
    if 'result' in st.session_state:
        st.success("Model berhasil dimuat dan siap digunakan")
    else:
        st.warning("Model sedang dimuat...")