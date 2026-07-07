"""
Arquiteturas e funções de perda das duas abordagens.

ABORDAGEM FLAT
    vision_learner padrão do fastai com backbone timm (ConvNeXt-Tiny) e uma
    única cabeça de classificação sobre articleType. Os níveis superiores
    são derivados via Taxonomy na inferência (consistência por construção).

ABORDAGEM MULTI-HEAD (multi-task learning)
    Um único backbone compartilhado + três cabeças lineares independentes,
    uma por nível. A perda é a soma ponderada de três cross-entropies:

        L = w_m·CE(master) + w_s·CE(sub) + w_a·CE(article)

    Intuição: o backbone é forçado a aprender representações úteis para os
    três níveis ao mesmo tempo. A supervisão nos níveis grossos atua como
    regularizador do nível fino (sinal mais fácil e menos ruidoso), e o
    custo de inferência é praticamente o mesmo do flat, já que as cabeças
    são três camadas lineares baratas sobre o mesmo embedding.

    O preço: as três predições são independentes, então nada garante
    consistência hierárquica. Medimos isso em evaluate.py.

POR QUE ConvNeXt-Tiny?
    Arquitetura convolucional moderna (Liu et al., 2022) que incorpora
    lições dos Vision Transformers mantendo a eficiência de CNNs. Na faixa
    de ~28M parâmetros, supera ResNet50 em acurácia no ImageNet com custo
    comparável, e treina confortavelmente em uma única GPU modesta (ex.: T4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .taxonomy import LEVELS, Taxonomy

DEFAULT_ARCH = "convnext_tiny"


# ---------------------------------------------------------------------- #
# Modelo multi-head                                                       #
# ---------------------------------------------------------------------- #
class MultiHeadNet(nn.Module):
    """Backbone timm compartilhado + uma cabeça linear por nível hierárquico.

    forward(x) retorna uma TUPLA de logits na ordem canônica de LEVELS:
    (logits_master, logits_sub, logits_article). O fastai propaga essa tupla
    para a loss e para as métricas junto com os três alvos.
    """

    def __init__(self, n_classes: dict[str, int], arch: str = DEFAULT_ARCH, pretrained: bool = True):
        super().__init__()
        import timm  # import local: mantém o módulo importável sem timm p/ docs

        # num_classes=0 + global_pool="avg": o timm remove a cabeça original
        # e devolve o embedding pós-pooling, exatamente o que queremos para
        # pendurar nossas cabeças.
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
        n_features = self.backbone.num_features

        self.levels = list(LEVELS)  # ordem fixa e explícita
        self.heads = nn.ModuleDict(
            {level: nn.Linear(n_features, n_classes[level]) for level in self.levels}
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        return tuple(self.heads[level](features) for level in self.levels)


def multihead_splitter(model: MultiHeadNet):
    """Divide os parâmetros em grupos para discriminative learning rates.

    Grupo 0: backbone (pré-treinado, taxas menores / congelável)
    Grupo 1: cabeças (inicializadas do zero, taxas maiores)

    É isso que faz learn.freeze() congelar só o backbone e fine_tune()
    aplicar o cronograma clássico de transfer learning: primeiro treina as
    cabeças com o backbone congelado, depois descongela tudo com taxas
    discriminativas (menores nas camadas iniciais, que carregam features
    genéricas de baixo nível, maiores nas finais).
    """
    from fastai.torch_core import params

    return [params(model.backbone), params(model.heads)]


# ---------------------------------------------------------------------- #
# Loss multi-tarefa                                                       #
# ---------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    """Soma ponderada de cross-entropies, uma por nível.

    Pesos padrão (1.0, 1.0, 1.0): sem evidência prévia de que um nível deva
    dominar, começamos neutros; os pesos ficam no YAML justamente para virar
    objeto de experimento (ex.: aumentar o peso do articleType, que é a
    tarefa mais difícil). Label smoothing leve (0.1) porque catálogos reais
    contêm erros de rotulagem e o smoothing reduz o excesso de confiança do
    modelo nesses rótulos.
    """

    def __init__(self, weights: tuple[float, float, float] = (1.0, 1.0, 1.0), label_smoothing: float = 0.1):
        super().__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing

    def forward(self, preds: tuple[torch.Tensor, ...], *targets: torch.Tensor) -> torch.Tensor:
        # preds: tupla de logits (master, sub, article) vinda do MultiHeadNet
        # targets: três tensores de índices de classe, na mesma ordem
        assert len(preds) == len(targets) == len(self.weights), (
            f"Esperava {len(self.weights)} saídas/alvos, "
            f"recebi {len(preds)} saídas e {len(targets)} alvos"
        )
        total = preds[0].new_zeros(())
        for w, p, t in zip(self.weights, preds, targets):
            total = total + w * F.cross_entropy(p, t, label_smoothing=self.label_smoothing)
        return total


# ---------------------------------------------------------------------- #
# Métricas por nível                                                      #
# ---------------------------------------------------------------------- #
def accuracy_at_level(index: int, name: str):
    """Fábrica de métricas: acurácia da cabeça `index` da tupla de saídas.

    No fastai, uma métrica-função recebe (saída_do_modelo, *alvos). Como o
    modelo devolve tupla e há três alvos, indexamos ambos pela posição do
    nível. O atributo __name__ define o rótulo da coluna no log de treino.
    """

    def _accuracy(preds: tuple[torch.Tensor, ...], *targets: torch.Tensor) -> torch.Tensor:
        return (preds[index].argmax(dim=1) == targets[index]).float().mean()

    _accuracy.__name__ = name
    return _accuracy


def multihead_metrics():
    """Uma métrica de acurácia por nível, na ordem canônica de LEVELS."""
    short = {"masterCategory": "master", "subCategory": "sub", "articleType": "article"}
    return [accuracy_at_level(i, f"acc_{short[level]}") for i, level in enumerate(LEVELS)]


# ---------------------------------------------------------------------- #
# Fábricas de Learner                                                     #
# ---------------------------------------------------------------------- #
def create_flat_learner(dls, arch: str = DEFAULT_ARCH):
    """Learner padrão do fastai para a abordagem flat.

    vision_learner resolve tudo: baixa o backbone timm pré-treinado, monta a
    cabeça com o nº de classes do dls e configura o splitter para
    discriminative LR. É a base de comparação: o mínimo bem feito.
    """
    from fastai.vision.all import CrossEntropyLossFlat, accuracy, vision_learner

    return vision_learner(
        dls,
        arch,
        metrics=accuracy,
        # Mesmo label smoothing da MultiTaskLoss: mais uma variável isolada
        # para a comparação flat vs multi-head ser justa.
        loss_func=CrossEntropyLossFlat(label_smoothing=0.1),
    )


def create_multihead_learner(dls, taxonomy: Taxonomy, arch: str = DEFAULT_ARCH,
                             loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """Learner com o MultiHeadNet, loss combinada e métricas por nível."""
    from fastai.learner import Learner

    n_classes = {level: len(taxonomy.vocab(level)) for level in LEVELS}
    model = MultiHeadNet(n_classes=n_classes, arch=arch, pretrained=True)

    learner = Learner(
        dls,
        model,
        loss_func=MultiTaskLoss(weights=loss_weights),
        metrics=multihead_metrics(),
        splitter=multihead_splitter,
    )
    return learner
