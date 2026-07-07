"""
Demo interativo (Gradio) do classificador hierárquico de produtos.

Deploy no Hugging Face Spaces:
    1. Crie um Space (SDK: Gradio).
    2. Suba este app.py, o requirements.txt desta pasta e os artefatos do
       run vencedor (export_*.pkl + taxonomy.json) em uma pasta run/.
    3. O Space instala as dependências e serve o app automaticamente.

Repare que este arquivo NÃO sabe qual abordagem (flat ou multi-head) está
servindo: ele fala apenas com o HierarchicalPredictor, cujo contrato de
saída é idêntico para as duas. Trocar o modelo = trocar a pasta do run.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from hierclf.inference import HierarchicalPredictor
from hierclf.taxonomy import LEVELS

# Diretório do run servível: variável de ambiente (usada pelo Makefile)
# com fallback para o layout padrão do Space (pasta run/ ao lado do app).
RUN_DIR = Path(os.environ.get("RUN_DIR", Path(__file__).parent / "run"))

predictor = HierarchicalPredictor.from_run(RUN_DIR)

NOMES_PT = {
    "masterCategory": "Categoria principal",
    "subCategory": "Subcategoria",
    "articleType": "Tipo de produto",
}


def classificar(imagem):
    """Callback do Gradio: imagem PIL -> três painéis de rótulo + nota."""
    resultado = predictor.predict(imagem)

    # gr.Label espera {rótulo: confiança}; um painel por nível deixa a
    # estrutura hierárquica visualmente óbvia para quem testa o demo.
    paineis = [
        {resultado[level]["label"]: resultado[level]["confidence"]}
        for level in LEVELS
    ]

    if resultado["consistente"]:
        nota = "Predição hierarquicamente consistente com a taxonomia do catálogo."
    else:
        nota = (
            "Atenção: as três cabeças produziram uma combinação que não existe "
            "na taxonomia (limitação conhecida da abordagem multi-head, discutida no README)."
        )
    return (*paineis, nota)


with gr.Blocks(title="Classificador hierárquico de produtos") as demo:
    gr.Markdown(
        """
        # Classificador hierárquico de produtos de e-commerce
        Envie a foto de um produto (idealmente em fundo claro, estilo catálogo)
        e o modelo prevê **três níveis** da taxonomia: categoria principal,
        subcategoria e tipo de produto.

        Modelo: ConvNeXt-Tiny com fine-tuning (fastai + timm) no dataset
        Fashion Product Images. Código, experimentos e limitações:
        [repositório no GitHub](https://github.com/tf-ferreira/hierarchical-product-classifier).
        """
    )
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", label="Foto do produto")
            botao = gr.Button("Classificar", variant="primary")
        with gr.Column():
            saidas = [gr.Label(num_top_classes=1, label=NOMES_PT[level]) for level in LEVELS]
            nota = gr.Markdown()

    botao.click(classificar, inputs=entrada, outputs=[*saidas, nota])

if __name__ == "__main__":
    demo.launch()
