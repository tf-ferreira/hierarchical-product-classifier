"""
Testes do ExperimentLogger e do split estratificado.

Nenhum destes testes toca fastai/torch: rodam em segundos em qualquer
máquina (inclusive a sua, sem GPU). O que depende de GPU é validado em
ambiente de treino; o que é lógica pura é validado aqui. Essa divisão é
intencional.
"""
import json

import numpy as np
import pandas as pd
import pytest

from hierclf.data import stratified_valid_indices
from hierclf.tracking import ExperimentLogger, _flatten


# ---------------------------------------------------------------------- #
# ExperimentLogger                                                        #
# ---------------------------------------------------------------------- #
def test_logger_cria_config_imediatamente(tmp_path):
    cfg = {"approach": "flat", "train": {"epochs": 3}}
    logger = ExperimentLogger(tmp_path, run_name="flat", config=cfg)
    saved = json.loads((logger.dir / "config.json").read_text())
    # A config completa está no snapshot ANTES de qualquer época treinar.
    assert saved["config"] == cfg
    assert saved["run_name"] == "flat"


def test_logger_epocas_em_jsonl(tmp_path):
    logger = ExperimentLogger(tmp_path, run_name="x", config={})
    logger.log_epoch(0, train_loss=1.5, valid_loss=1.2)
    logger.log_epoch(1, train_loss=1.1, valid_loss=1.0)
    df = logger.load_epochs()
    assert list(df["epoch"]) == [0, 1]
    assert df["valid_loss"].iloc[-1] == pytest.approx(1.0)


def test_finalize_consolida_runs_csv(tmp_path):
    """Dois runs devem virar duas linhas no MESMO runs.csv, com a config
    achatada em colunas: é daí que sai a tabela comparativa do README."""
    for name, acc in (("flat", 0.90), ("multihead", 0.92)):
        logger = ExperimentLogger(tmp_path, run_name=name, config={"train": {"epochs": 8}})
        logger.finalize(acc_article=acc)
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert len(runs) == 2
    assert set(runs["run_name"]) == {"flat", "multihead"}
    assert "cfg.train.epochs" in runs.columns
    assert runs["acc_article"].max() == pytest.approx(0.92)


def test_flatten():
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    assert _flatten(nested) == {"a": 1, "b.c": 2, "b.d.e": 3}


# ---------------------------------------------------------------------- #
# Split estratificado                                                     #
# ---------------------------------------------------------------------- #
@pytest.fixture
def df_desbalanceado() -> pd.DataFrame:
    """100 exemplos da classe A, 30 da B e 5 da C (cauda longa em miniatura)."""
    labels = ["A"] * 100 + ["B"] * 30 + ["C"] * 5
    return pd.DataFrame({"articleType": labels, "image_path": [f"{i}.jpg" for i in range(135)]})


def test_split_e_deterministico(df_desbalanceado):
    """Mesma seed => mesmo split. É a garantia de que flat e multi-head
    são avaliados no MESMO conjunto de validação."""
    a = stratified_valid_indices(df_desbalanceado, valid_pct=0.2, seed=42)
    b = stratified_valid_indices(df_desbalanceado, valid_pct=0.2, seed=42)
    assert np.array_equal(a, b)
    c = stratified_valid_indices(df_desbalanceado, valid_pct=0.2, seed=7)
    assert not np.array_equal(a, c)  # e seed diferente => split diferente


def test_split_estratifica_por_classe(df_desbalanceado):
    idx = stratified_valid_indices(df_desbalanceado, valid_pct=0.2, seed=42)
    valid = df_desbalanceado.loc[idx, "articleType"].value_counts()
    # ~20% de cada classe, com pelo menos 1 exemplo da classe rara.
    assert valid["A"] == 20
    assert valid["B"] == 6
    assert valid["C"] >= 1


def test_split_sem_vazamento(df_desbalanceado):
    """Nenhum índice de validação pode se repetir (vazamento treino/valid)."""
    idx = stratified_valid_indices(df_desbalanceado, valid_pct=0.2, seed=42)
    assert len(idx) == len(set(idx))
    assert set(idx).issubset(set(df_desbalanceado.index))
