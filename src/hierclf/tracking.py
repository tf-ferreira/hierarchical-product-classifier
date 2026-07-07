"""
Tracking de experimentos, implementação própria e sem dependências externas.

Filosofia: um experimento só existe se for reproduzível e comparável.
Para isso, cada run gera:

    experiments/
    └── <run_name>-<run_id>/
        ├── config.json      # config COMPLETA do run (hiperparâmetros, seed,
        │                    # dataset, arquitetura). Reproduzir = reler isto.
        ├── metrics.jsonl    # uma linha JSON por época (formato append-only,
        │                    # robusto a interrupções: nada se perde se o
        │                    # ambiente cair no meio do treino)
        └── ...              # checkpoints/figuras que o treino quiser salvar

E, ao finalizar, anexa UMA linha em experiments/runs.csv, a tabela
consolidada de onde saem os gráficos comparativos do README.

Por que JSONL por época e CSV consolidado?
    - JSONL é append-only: cada época é um write atômico de uma linha.
    - CSV consolidado é o formato natural para pandas -> tabela do README.

Este módulo é Python puro (stdlib + pandas), sem fastai/torch, logo é
testável em qualquer ambiente. A ponte com o fastai (Callback que chama
log_epoch a cada época) vive em train.py, mantendo este módulo agnóstico
ao framework de treino.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten(d: dict, prefix: str = "") -> dict:
    """Achata dict aninhado: {"model": {"arch": "x"}} -> {"model.arch": "x"}.

    Necessário porque a config YAML é aninhada, mas uma linha de CSV é plana.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


class ExperimentLogger:
    """Logger de um único run de experimento.

    Uso típico (ver train.py):

        logger = ExperimentLogger("experiments", run_name="multihead", config=cfg)
        for epoch in ...:
            logger.log_epoch(epoch, train_loss=..., valid_loss=..., acc_article=...)
        logger.finalize(acc_article=0.93, acc_master=0.99)
    """

    def __init__(self, base_dir: str | Path, run_name: str, config: dict):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.config = config

        # run_id legível + sufixo aleatório: ordenável por data e sem colisão
        # mesmo que dois runs comecem no mesmo segundo.
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = self.base_dir / f"{run_name}-{self.run_id}"
        self.dir.mkdir(parents=True, exist_ok=False)

        # Config salva IMEDIATAMENTE, antes de qualquer treino: se o run
        # falhar, ainda sabemos exatamente o que foi tentado.
        snapshot = {
            "run_id": self.run_id,
            "run_name": run_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "config": config,
        }
        (self.dir / "config.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self._metrics_path = self.dir / "metrics.jsonl"
        self._t0 = time.time()

    # ------------------------------------------------------------------ #
    def log_epoch(self, epoch: int, **metrics: float) -> None:
        """Anexa as métricas de uma época (uma linha JSON, write atômico)."""
        row = {"epoch": int(epoch), "elapsed_s": round(time.time() - self._t0, 1)}
        # Converte tensores/np.float para float nativo antes de serializar.
        row.update({k: (float(v) if v is not None else None) for k, v in metrics.items()})
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    def finalize(self, **final_metrics: float) -> Path:
        """Fecha o run: anexa a linha-resumo em runs.csv e a retorna.

        runs.csv é a tabela consolidada de TODOS os runs. Colunas: metadados
        do run + config achatada + métricas finais. Se runs futuros tiverem
        colunas novas, o pandas alinha por nome (colunas ausentes viram NaN).
        """
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": round(time.time() - self._t0, 1),
        }
        row.update(_flatten(self.config, prefix="cfg."))
        row.update({k: (float(v) if v is not None else None) for k, v in final_metrics.items()})

        runs_csv = self.base_dir / "runs.csv"
        new = pd.DataFrame([row])
        if runs_csv.exists():
            # Concatena com o histórico para alinhar colunas por nome,
            # depois regrava. Com dezenas de runs isso é barato; se um dia
            # forem milhares, migrar para SQLite seria o próximo passo.
            old = pd.read_csv(runs_csv)
            pd.concat([old, new], ignore_index=True).to_csv(runs_csv, index=False)
        else:
            new.to_csv(runs_csv, index=False)
        return runs_csv

    # ------------------------------------------------------------------ #
    def load_epochs(self) -> pd.DataFrame:
        """Carrega o histórico de épocas do run como DataFrame (para plots)."""
        if not self._metrics_path.exists():
            return pd.DataFrame()
        rows = [
            json.loads(line)
            for line in self._metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)
