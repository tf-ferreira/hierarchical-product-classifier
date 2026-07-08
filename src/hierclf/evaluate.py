"""
Avaliação e análise de erros. Ponto de entrada:

    python -m hierclf.evaluate --run experiments/multihead-<run_id>

Um treino sem análise de erros é meio treino. Este módulo produz:

1.  Acurácia por nível (master, sub, article), na MESMA validação para as
    duas abordagens (o split é função da seed, ver data.py).

2.  Taxa de consistência hierárquica: fração das predições cuja tripla
    (master, sub, article) respeita a taxonomia.
        - Flat: 100% por construção (níveis superiores são derivados).
        - Multi-head: medida empiricamente; é O custo da abordagem e
          precisa aparecer na tabela comparativa, não só a acurácia.

3.  Matriz de confusão do nível articleType restrita aos pares mais
    confundidos (com dezenas de classes, a matriz completa é ilegível;
    reportar o top-K de confusões comunica mais).

Sobre "acurácia derivada" no flat: a predição de master/sub do flat é
parents_of(articleType_previsto). Se o articleType estiver certo, os pais
estão certos; se estiver errado, os pais ainda podem acertar (confundir
Tshirts com Shirts mantém master=Apparel). Por isso a acurácia derivada
dos níveis grossos costuma ser ALTA no flat, e a comparação com o
multi-head nesses níveis é genuinamente informativa.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_dataloaders, download_dataset, load_and_validate
from .taxonomy import LEVELS, Taxonomy


# ---------------------------------------------------------------------- #
def _decode(preds: "np.ndarray", vocab: list[str]) -> list[str]:
    """argmax de logits -> rótulos legíveis, via vocabulário explícito."""
    return [vocab[i] for i in preds.argmax(axis=1)]


def evaluate_run(run_dir: str | Path) -> dict:
    """Avalia um run exportado e retorna um dict de métricas (também salvo
    em <run_dir>/evaluation.json, para o README consolidar)."""
    from fastai.learner import load_learner

    run_dir = Path(run_dir)
    taxonomy = Taxonomy.from_json(run_dir / "taxonomy.json")

    # O nome do export codifica a abordagem (ver train.py).
    exports = list(run_dir.glob("export_*.pkl"))
    if len(exports) != 1:
        raise FileNotFoundError(f"Esperava exatamente 1 export_*.pkl em {run_dir}, achei {len(exports)}")
    export_path = exports[0]
    approach = export_path.stem.replace("export_", "")

    # learn.export NÃO carrega dados (o pipeline é salvo vazio, por design do
    # fastai). Reconstruímos o conjunto de validação a partir da config do
    # run: como o split é função determinística de (dados, seed), obtemos
    # EXATAMENTE a validação vista no treino.
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    data_root = download_dataset(version=config["dataset"]["version"])
    df, _, _ = load_and_validate(
        data_root, min_samples_per_class=int(config["dataset"]["min_samples_per_class"])
    )
    dls = build_dataloaders(
        df,
        taxonomy,  # a taxonomia ARTEFATO do run, para vocabulários idênticos
        approach=approach,
        img_size=int(config["dataset"]["img_size"]),
        batch_size=int(config["train"]["batch_size"]),
        valid_pct=float(config["dataset"]["valid_pct"]),
        seed=int(config["seed"]),
    )

    import torch

    # cpu=False quando há GPU: os batches do dls reconstruído vivem no
    # dispositivo padrão (cuda, se disponível) e o modelo precisa acompanhar.
    learn = load_learner(export_path, cpu=not torch.cuda.is_available())
    # get_preds na validação: logits + alvos, sem augmentation (só transforms
    # determinísticos), exatamente como em produção.
    preds, targets = learn.get_preds(dl=dls.valid)

    vocabs = {level: taxonomy.vocab(level) for level in LEVELS}
    results: dict = {"approach": approach, "run_dir": str(run_dir)}

    if approach == "flat":
        # preds: logits de articleType; targets: índices de articleType
        article_pred = _decode(preds.numpy(), vocabs["articleType"])
        article_true = [vocabs["articleType"][i] for i in targets.numpy()]

        # Deriva os níveis superiores da predição E do alvo via taxonomia.
        derived_pred = {"articleType": article_pred}
        derived_true = {"articleType": article_true}
        for key, idx in (("subCategory", 0), ("masterCategory", 1)):
            derived_pred[key] = [taxonomy.parents_of(a)[idx] for a in article_pred]
            derived_true[key] = [taxonomy.parents_of(a)[idx] for a in article_true]

        for level in LEVELS:
            acc = float(np.mean(np.array(derived_pred[level]) == np.array(derived_true[level])))
            results[f"acc_{level}"] = acc
        results["consistencia_hierarquica"] = 1.0  # por construção
        pred_article, true_article = article_pred, article_true

    else:  # multihead
        # preds: tupla de 3 tensores de logits; targets: tupla de 3 tensores
        decoded_pred = {level: _decode(p.numpy(), vocabs[level]) for level, p in zip(LEVELS, preds)}
        decoded_true = {level: [vocabs[level][i] for i in t.numpy()] for level, t in zip(LEVELS, targets)}

        for level in LEVELS:
            acc = float(np.mean(np.array(decoded_pred[level]) == np.array(decoded_true[level])))
            results[f"acc_{level}"] = acc

        consistent = [
            taxonomy.is_consistent(m, s, a)
            for m, s, a in zip(
                decoded_pred["masterCategory"],
                decoded_pred["subCategory"],
                decoded_pred["articleType"],
            )
        ]
        results["consistencia_hierarquica"] = float(np.mean(consistent))
        pred_article, true_article = decoded_pred["articleType"], decoded_true["articleType"]

    # ------------------- top confusões (articleType) ------------------ #
    confusion = (
        pd.DataFrame({"true": true_article, "pred": pred_article})
        .query("true != pred")
        .value_counts()
        .head(15)
        .reset_index(name="n")
    )
    results["top_confusoes"] = confusion.to_dict(orient="records")

    out = run_dir / "evaluation.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[hierclf] avaliação salva em {out}")
    for level in LEVELS:
        print(f"  acc_{level}: {results[f'acc_{level}']:.4f}")
    print(f"  consistência hierárquica: {results['consistencia_hierarquica']:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um run exportado")
    parser.add_argument("--run", required=True, help="diretório do run (experiments/<nome>-<id>)")
    args = parser.parse_args()
    evaluate_run(args.run)


if __name__ == "__main__":
    main()
