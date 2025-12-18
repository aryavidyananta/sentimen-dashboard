import asyncio
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoModel, AutoTokenizer

# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

class AppConfig:
    BERT_MODEL_NAME = "indobenchmark/indobert-base-p1"
    CNN_MODEL_PATH = "cnn_best_kernelX.pth"
    CNN_KERNEL_SIZE = 3
    CNN_NUM_FILTERS = 64
    CNN_DROPOUT = 0.5
    CNN_NUM_CLASSES = 3
    MAX_SEQ_LEN = 64


# =========================================================
# 2. DEFINISI MODEL CNN
# =========================================================

class SimpleCNN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        num_filters: int = 64,
        dropout_rate: float = 0.5,
        num_classes: int = 3,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=768,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, 768]
        x = x.permute(0, 2, 1)          # -> [batch, 768, seq_len]
        x = self.relu(self.conv(x))     # -> [batch, num_filters, seq_len]
        x = self.pool(x).squeeze(-1)    # -> [batch, num_filters]
        x = self.dropout(x)
        return self.fc(x)               # -> [batch, num_classes]


# =========================================================
# 3. DATA CLASS UNTUK RESOURCE MODEL
# =========================================================

@dataclass
class ModelResources:
    tokenizer: AutoTokenizer
    bert_model: AutoModel
    cnn_model: SimpleCNN
    stemmer: object


# =========================================================
# 4. SERVICE: LOADING RESOURCES (CACHED)
# =========================================================

@st.cache_resource
def load_resources() -> ModelResources:
    # Load tokenizer dan IndoBERT
    tokenizer = AutoTokenizer.from_pretrained(AppConfig.BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(AppConfig.BERT_MODEL_NAME)

    # Load SimpleCNN
    cnn_model = SimpleCNN(
        kernel_size=AppConfig.CNN_KERNEL_SIZE,
        num_filters=AppConfig.CNN_NUM_FILTERS,
        dropout_rate=AppConfig.CNN_DROPOUT,
        num_classes=AppConfig.CNN_NUM_CLASSES,
    )
    state_dict = torch.load(AppConfig.CNN_MODEL_PATH, map_location="cpu")
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    # Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return ModelResources(
        tokenizer=tokenizer,
        bert_model=bert_model,
        cnn_model=cnn_model,
        stemmer=stemmer,
    )


# =========================================================
# 5. SERVICE: PREPROCESSING TEKS
# =========================================================

class TextPreprocessor:
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def preprocess(self, text: str) -> Dict:
        # Cleaning
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        # Case folding
        case_folded = cleaned_text.lower()

        # Tokenization
        tokens = case_folded.split()

        # Stemming
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

        return {
            "original": text,
            "cleaned": cleaned_text,
            "case_folded": case_folded,
            "tokens": tokens,
            "stemmed_tokens": stemmed_tokens,
            "stemmed_text": " ".join(stemmed_tokens),
        }


# =========================================================
# 6. SERVICE: BERT EMBEDDING
# =========================================================

class BertEmbeddingService:
    def __init__(self, tokenizer: AutoTokenizer, bert_model: AutoModel):
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=AppConfig.MAX_SEQ_LEN,
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state  # [batch, seq_len, hidden]


# =========================================================
# 7. SERVICE: SENTIMENT ANALYZER (SYNC + ASYNC)
# =========================================================

class SentimentAnalyzer:
    LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

    def __init__(self, resources: ModelResources):
        self.resources = resources
        self.bert_service = BertEmbeddingService(
            tokenizer=resources.tokenizer,
            bert_model=resources.bert_model,
        )
        self.cnn_model = resources.cnn_model

    def _classify_sync(
        self, text: str
    ) -> Tuple[str, List[float], List[float], List[str], List[List[float]]]:
        """
        Fungsi sinkron untuk klasifikasi sentimen.
        Dipisah agar bisa dipanggil dari versi async.
        """
        embedding = self.bert_service.get_embedding(text)  # [1, seq_len, 768]
        output = self.cnn_model(embedding)                 # [1, num_classes]
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs).item()
        label = self.LABEL_MAP[pred_class]

        # Embedding rata-rata untuk ditampilkan di UI
        processed_embedding = torch.mean(embedding, dim=1).squeeze().tolist()

        # Token dan token-level embedding (opsional, untuk visualisasi)
        token_ids = self.resources.tokenizer.encode(text)
        tokens = self.resources.tokenizer.convert_ids_to_tokens(token_ids)
        token_embeddings = embedding.squeeze(0).tolist()

        return label, probs.squeeze().tolist(), processed_embedding, tokens, token_embeddings

    async def classify_async(
        self, text: str
    ) -> Tuple[str, List[float], List[float], List[str], List[List[float]]]:
        """
        Versi async: menjalankan _classify_sync di thread pool sehingga
        tidak memblokir event loop (berguna untuk skenario multi-user).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._classify_sync, text)


# =========================================================
# 8. LAYER UI: RENDERING KOMPONEN
# =========================================================

def setup_page():
    st.set_page_config(
        page_title="Analisis Sentimen Bahasa Indonesia",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS custom
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="animate-fade-in">
            <div style="background: linear-gradient(to right, #4f46e5, #6366f1); color: white; padding: 3rem 1rem; border-radius: 0 0 24px 24px; text-align: center; margin-bottom: 2rem;">
                <div class="animate-float" style="font-size: 4rem;">📊</div>
                <h1 style="font-size: 3rem; font-weight: 800; margin: 0.5rem 0;">Dashboard Analisis Sentimen</h1>
                <p style="font-size: 1.25rem; opacity: 0.9;">Lihat hasil analisis sentimen secara langsung dengan visual yang interaktif dan mudah dipahami</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_form() -> Tuple[str, bool]:
    with st.container():
        st.markdown(
            """
            <div style="background: white; border-radius: 24px; padding: 2rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
                <h2 class="main-header">Analisis Sentimen Program Naturalisasi Timnas Indonesia</h2>
            """,
            unsafe_allow_html=True,
        )

        input_text = st.text_area(
            "Masukkan teks dalam Bahasa Indonesia...",
            height=150,
            max_chars=1000,
            key="text_input",
            help="Maksimal 1000 karakter",
        )

        char_count = len(input_text)
        st.caption(f"{char_count}/1000 karakter")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button(
                "Analisis Sentimen",
                type="primary",
                use_container_width=True,
                disabled=char_count == 0,
            )

    return input_text, analyze_btn


def render_result_section(result: Dict):
    label = result["label"]
    probs = result["probs"]

    with st.container():
        st.markdown(
            """
            <div style="background: white; border-radius: 24px; padding: 2rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
                <h2 style="font-size: 1.5rem; font-weight: 700; color: #4f46e5; border-bottom: 2px solid #eef2ff; padding-bottom: 1rem; margin-bottom: 1.5rem;">
                    📊 Hasil Analisis Sentimen
                </h2>
            """,
            unsafe_allow_html=True,
        )

        emoji = "😠" if label == "Negatif" else "😐" if label == "Netral" else "😊"
        bg_color = (
            "bg-red-100"
            if label == "Negatif"
            else "bg-yellow-100"
            if label == "Netral"
            else "bg-green-100"
        )
        text_color = (
            "text-red-800"
            if label == "Negatif"
            else "text-yellow-800"
            if label == "Netral"
            else "text-green-800"
        )
        border_color = (
            "border-red-500"
            if label == "Negatif"
            else "border-yellow-500"
            if label == "Netral"
            else "border-green-500"
        )

        st.markdown(
            f"""
            <div class="result-box {bg_color} {text_color} {border_color}">
                <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                    <span style="font-size: 2rem;">{emoji}</span>
                    <span style="font-size: 1.5rem; font-weight: 700;">{label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Probabilitas Sentimen")

        labels = ["Negatif", "Netral", "Positif"]
        colors = ["negative", "neutral", "positive"]

        for i, (lbl, color) in enumerate(zip(labels, colors)):
            prob_percent = probs[i] * 100
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"""
                    <div class="progress-container">
                        <div class="progress-bar {color}" style="width: {prob_percent}%">
                            {prob_percent:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(f"**{lbl}**")

        with st.expander("🔍 Lihat Detail Preprocessing", expanded=False):
            pre = result["preprocessing"]
            st.markdown("### Proses Preprocessing Teks")

            st.markdown("**Teks Asli**")
            st.code(pre["original"], language="text")

            st.markdown("**Cleaning**")
            st.code(pre["cleaned"], language="text")

            st.markdown("**Case Folding**")
            st.code(pre["case_folded"], language="text")

            st.markdown("**Tokenisasi**")
            st.write(f"Total Token: {len(pre['tokens'])}")
            tokens_html = "".join(
                f'<span class="token-pill">{token}</span>'
                for token in pre["tokens"]
            )
            st.markdown(
                f'<div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">{tokens_html}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Stemming**")
            st.write(f'Stemmed Text: "{pre["stemmed_text"]}"')
            stemmed_tokens_html = "".join(
                f'<span class="token-pill" style="background-color: #dcfce7; color: #166534;">{token}</span>'
                for token in pre["stemmed_tokens"]
            )
            st.markdown(
                f'<div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">{stemmed_tokens_html}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🧠 Lihat Detail Embedding", expanded=False):
            st.markdown("### Embedding IndoBERT")
            st.markdown("Representasi vektor 768 dimensi (average pooling)")
            st.json(result["embedding"])

            st.markdown("### Token dan Embedding")
            st.markdown("Menampilkan 10 dimensi pertama dari masing-masing token")

            data = []
            for i, (token, emb) in enumerate(
                zip(result["tokens"], result["token_embeddings"])
            ):
                if i >= 10:
                    break
                short_emb = [round(float(x), 4) for x in emb[:10]]
                data.append(
                    {
                        "Token": token,
                        "Embedding (10 Dimensi Pertama)": str(short_emb) + " ...",
                    }
                )
            st.table(data)

    st.markdown("---")
    current_year = datetime.now().year
    st.markdown(
        f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 1rem;">
            <p>© {current_year} Sentiment Analysis Dashboard</p>
            <p>Dibuat oleh <span style="color: #4f46e5; font-weight: 600;">Arya Vidyananta</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="background: linear-gradient(to bottom, #4f46e5, #6366f1); color: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem;">
                <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Tentang Model</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            """
            - Arsitektur: SimpleCNN dengan kernel size 3
            - Embedding: IndoBERT Base P1
            - Kelas: Negatif, Netral, Positif
            - Preprocessing: Cleaning, case folding, tokenization, dan stemming
            """
        )

        st.markdown("---")
        st.markdown("**Detail Teknis:**")
        st.markdown("- Framework: PyTorch")
        st.markdown("- Library: Transformers, Sastrawi")
        st.markdown("- Deployment: Streamlit")
        st.markdown("---")

        if "result" in st.session_state:
            st.success("Model berhasil dimuat dan siap digunakan")
        else:
            st.warning("Model sedang menunggu input...")


# =========================================================
# 9. MAIN APP
# =========================================================

def main():
    setup_page()
    render_sidebar()

    try:
        resources = load_resources()
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return

    preprocessor = TextPreprocessor(resources.stemmer)
    analyzer = SentimentAnalyzer(resources)

    input_text, analyze_btn = render_input_form()

    if analyze_btn and input_text.strip():
        with st.spinner("Sedang menganalisis..."):
            preprocessed = preprocessor.preprocess(input_text)

            # Jalankan analisis secara async (tetap dipanggil sinkron dari Streamlit)
            result_tuple = asyncio.run(
                analyzer.classify_async(preprocessed["stemmed_text"])
            )
            label, probs, embedding, tokens, token_embeddings = result_tuple

            st.session_state.result = {
                "label": label,
                "probs": probs,
                "embedding": embedding,
                "tokens": tokens,
                "token_embeddings": token_embeddings,
                "preprocessing": preprocessed,
            }

    if "result" in st.session_state:
        render_result_section(st.session_state.result)


if __name__ == "__main__":
    main()
