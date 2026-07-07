"""
Contrato de inferência: a interface estável entre o modelo e o mundo.

O app Gradio (e qualquer consumidor futuro, uma API, um batch job) não deve
saber se o modelo é flat ou multi-head. Ele chama:

    predictor = HierarchicalPredictor.from_run("experiments/multihead-...")
    predictor.predict("caminho/da/imagem.jpg")

e recebe SEMPRE o mesmo formato:

    {
      "masterCategory": {"label": "Apparel",  "confidence": 0.99},
      "subCategory":    {"label": "Topwear",  "confidence": 0.97},
      "articleType":    {"label": "Tshirts",  "confidence": 0.91},
      "consistente": true,
      "approach": "multihead",
    }

Esse desacoplamento é o que permite trocar a abordagem vencedora do
experimento sem tocar em uma linha do app.
"""
from __future__ import annotations

from pathlib import Path

from .taxonomy import LEVELS, Taxonomy


class HierarchicalPredictor:
    """Encapsula load_learner + taxonomia e uniformiza a saída."""

    def __init__(self, learner, taxonomy: Taxonomy, approach: str):
        self.learner = learner
        self.taxonomy = taxonomy
        self.approach = approach

    # ------------------------------------------------------------------ #
    @classmethod
    def from_run(cls, run_dir: str | Path) -> "HierarchicalPredictor":
        """Carrega o predictor a partir dos artefatos de um run de treino."""
        from fastai.learner import load_learner

        run_dir = Path(run_dir)
        exports = list(run_dir.glob("export_*.pkl"))
        if len(exports) != 1:
            raise FileNotFoundError(f"Esperava 1 export_*.pkl em {run_dir}, achei {len(exports)}")
        approach = exports[0].stem.replace("export_", "")
        learner = load_learner(exports[0])
        taxonomy = Taxonomy.from_json(run_dir / "taxonomy.json")
        return cls(learner, taxonomy, approach)

    # ------------------------------------------------------------------ #
    def predict(self, image) -> dict:
        """Prediz a hierarquia completa para uma imagem (caminho ou PIL).

        learn.predict aplica os MESMOS transforms determinísticos do treino
        (resize + normalize), garantia dada pelo export do pipeline inteiro.
        """
        import torch

        if self.approach == "flat":
            # decoded = rótulo de articleType; probs = distribuição softmax
            _, pred_idx, probs = self.learner.predict(image)
            article = self.learner.dls.vocab[int(pred_idx)]
            confidence = float(probs[int(pred_idx)])
            sub, master = self.taxonomy.parents_of(article)
            # Nos níveis derivados, reportamos a MESMA confiança do nível
            # fino: a derivação é determinística, não há distribuição
            # própria. Deixar isso explícito evita interpretação errada.
            result = {
                "masterCategory": {"label": master, "confidence": confidence},
                "subCategory": {"label": sub, "confidence": confidence},
                "articleType": {"label": article, "confidence": confidence},
                "consistente": True,
            }
        else:  # multihead
            _, _, raw = self.learner.predict(image)
            # raw: tupla de tensores de probabilidade, um por cabeça,
            # na ordem canônica de LEVELS.
            result = {}
            labels = {}
            for level, probs in zip(LEVELS, raw):
                probs = torch.as_tensor(probs)
                idx = int(probs.argmax())
                label = self.taxonomy.vocab(level)[idx]
                labels[level] = label
                result[level] = {"label": label, "confidence": float(probs[idx])}
            result["consistente"] = self.taxonomy.is_consistent(
                labels["masterCategory"], labels["subCategory"], labels["articleType"]
            )

        result["approach"] = self.approach
        return result
