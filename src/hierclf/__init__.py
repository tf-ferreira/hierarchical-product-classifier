"""
hierclf: classificação hierárquica de produtos de e-commerce com fastai.

Módulos (do dado ao deploy):
    taxonomy   fonte única da verdade da hierarquia (Python puro, testável)
    tracking   logging próprio de experimentos (JSON por run + runs.csv)
    data       download, validação com relatório e DataLoaders
    model      backbone timm, cabeça multi-head, loss combinada, métricas
    train      orquestração: seeds, lr_find, fine-tuning, callbacks, export
    evaluate   acurácia por nível, consistência hierárquica, top confusões
    inference  contrato de predição estável consumido pelo app Gradio
"""
__version__ = "0.1.0"
