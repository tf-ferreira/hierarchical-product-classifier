"""
Demo interativo (Streamlit) do classificador hierárquico de produtos.

Deploy no Streamlit Community Cloud (gratuito):
    1. share.streamlit.io -> New app -> este repositório, branch main,
       main file path: app/streamlit_app.py.
    2. As dependências vêm do requirements.txt da raiz do repo (torch CPU).
    3. O modelo NÃO vive no Git: o app baixa o export do run vencedor do
       Hugging Face Hub (repo de modelo público) e cacheia entre sessões.

Como o app roda de dentro do próprio repositório clonado, o pacote hierclf
é importado direto de src/, sem instalação.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
from PIL import Image

from hierclf.taxonomy import LEVELS

HF_MODEL_REPO = "thf-thiago/hierarchical-product-classifier"
GITHUB_URL = "https://github.com/tf-ferreira/hierarchical-product-classifier"

NOMES_PT = {
    "masterCategory": "Categoria principal",
    "subCategory": "Subcategoria",
    "articleType": "Tipo de produto",
}

st.set_page_config(page_title="Classificador hierárquico de produtos", page_icon="🛍️")


@st.cache_resource(show_spinner="Baixando e carregando o modelo (só na primeira visita)...")
def carregar_predictor():
    """Baixa os artefatos do HF Hub e monta o predictor (cacheado no processo)."""
    from huggingface_hub import snapshot_download

    from hierclf.inference import HierarchicalPredictor

    run_dir = snapshot_download(HF_MODEL_REPO, allow_patterns=["export_*.pkl", "taxonomy.json"])
    return HierarchicalPredictor.from_run(run_dir)


st.title("Classificador hierárquico de produtos de e-commerce")
st.markdown(
    f"""
    Envie a foto de um produto (idealmente em fundo claro, estilo catálogo)
    e o modelo prevê **três níveis** da taxonomia: categoria principal,
    subcategoria e tipo de produto.

    Modelo: ConvNeXt-Tiny com fine-tuning (fastai + timm) no dataset
    Fashion Product Images — variante *flat + taxonomia*, com consistência
    hierárquica garantida por construção. Código, experimentos e limitações:
    [repositório no GitHub]({GITHUB_URL}) ·
    [modelo no Hugging Face](https://huggingface.co/{HF_MODEL_REPO}).
    """
)

arquivo = st.file_uploader("Foto do produto", type=["jpg", "jpeg", "png", "webp"])

if arquivo is not None:
    imagem = Image.open(arquivo).convert("RGB")
    col_img, col_pred = st.columns([1, 2])
    col_img.image(imagem, caption="Imagem enviada", use_container_width=True)

    with st.spinner("Classificando..."):
        resultado = carregar_predictor().predict(imagem)

    with col_pred:
        for level in LEVELS:
            rotulo = resultado[level]["label"]
            confianca = float(resultado[level]["confidence"])
            st.markdown(f"**{NOMES_PT[level]}:** {rotulo}")
            st.progress(confianca, text=f"confiança: {confianca:.1%}")

        if resultado["consistente"]:
            st.success("Predição hierarquicamente consistente com a taxonomia do catálogo.")
        else:
            st.warning(
                "As predições formam uma combinação que não existe na taxonomia "
                "(limitação da abordagem multi-head, discutida no README)."
            )
else:
    st.info("Envie uma imagem para começar.")
