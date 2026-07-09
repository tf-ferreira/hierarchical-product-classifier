"""
Orquestração do treino. Ponto de entrada:

    python -m hierclf.train --config configs/multihead.yaml

Tudo que varia entre experimentos vive no YAML; este módulo só executa.
Essa separação (config declarativa vs. código executor) é o que permite
comparar runs com rigor: o diff entre dois experimentos é o diff entre
dois arquivos YAML, legível em segundos.

Cronograma de treino (transfer learning clássico, via learn.fine_tune):
    Fase 1 (freeze_epochs): backbone CONGELADO, treina só a(s) cabeça(s).
        Motivo: as cabeças começam com pesos aleatórios; se o backbone
        estivesse solto, os gradientes ruidosos das cabeças destruiriam as
        features pré-treinadas nas primeiras iterações.
    Fase 2 (epochs): descongela tudo com discriminative learning rates,
        taxas menores nas camadas iniciais (features genéricas: bordas,
        texturas) e maiores nas finais (features específicas da tarefa).

Callbacks:
    - SaveModelCallback: mantém o checkpoint da MELHOR época (não da
      última), critério = valid_loss.
    - EarlyStoppingCallback: aborta se a valid_loss não melhorar por
      `patience` épocas; evita desperdiçar tempo de GPU à toa.
    - TrackingCallback (nosso): a cada época, envia as métricas ao
      ExperimentLogger. É a ponte fina entre o fastai e o tracking.py,
      que permanece agnóstico a framework.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import build_dataloaders, download_dataset, load_and_validate
from .model import create_flat_learner, create_multihead_learner
from .tracking import ExperimentLogger


# ---------------------------------------------------------------------- #
# Ponte fastai -> ExperimentLogger                                        #
# ---------------------------------------------------------------------- #
def make_tracking_callback(logger: ExperimentLogger):
    """Cria o callback dentro de uma função para adiar o import do fastai."""
    from fastai.callback.core import Callback

    class TrackingCallback(Callback):
        # order alto: roda depois do Recorder, quando as métricas da época
        # já foram computadas e estão em recorder.values/metric_names.
        order = 60

        def after_epoch(self):
            names = list(self.recorder.metric_names[1:-1])  # tira 'epoch' e 'time'
            values = [float(v) for v in self.recorder.values[-1]]
            logger.log_epoch(self.epoch, **dict(zip(names, values)))

    return TrackingCallback()


# ---------------------------------------------------------------------- #
# Treino de um experimento                                                #
# ---------------------------------------------------------------------- #
def run_training(config: dict, experiments_dir: str | Path = "experiments") -> Path:
    """Executa um experimento completo a partir de um dict de config.

    Retorna o diretório do run (com config, métricas, modelo exportado e
    taxonomy.json). Ver configs/*.yaml para o schema da config.
    """
    from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
    from fastai.torch_core import set_seed

    approach = config["approach"]
    seed = int(config["seed"])

    # Reprodutibilidade: fixa seeds de python/numpy/torch e ativa o modo
    # determinístico do cudnn. reproducible=True custa um pouco de
    # velocidade; em experimento comparativo, determinismo > throughput.
    set_seed(seed, reproducible=True)

    logger = ExperimentLogger(experiments_dir, run_name=approach, config=config)
    print(f"[hierclf] run {logger.run_id} -> {logger.dir}")

    # ------------------------- dados --------------------------------- #
    data_root = download_dataset(version=config["dataset"]["version"])
    df, taxonomy, report = load_and_validate(
        data_root, min_samples_per_class=int(config["dataset"]["min_samples_per_class"])
    )
    print(f"[hierclf] dados: {report.resumo()}")
    if report.conflitos_taxonomia:
        print(f"[hierclf] atenção: {len(report.conflitos_taxonomia)} conflito(s) de taxonomia (resolvidos por voto majoritário)")

    # A taxonomia é um ARTEFATO do run: a inferência e o app Gradio a
    # carregam deste JSON, sem depender do CSV original.
    taxonomy.to_json(logger.dir / "taxonomy.json")

    dls = build_dataloaders(
        df,
        taxonomy,
        approach=approach,
        img_size=int(config["dataset"]["img_size"]),
        batch_size=int(config["train"]["batch_size"]),
        valid_pct=float(config["dataset"]["valid_pct"]),
        seed=seed,
    )
    print(f"[hierclf] {len(dls.train_ds)} treino / {len(dls.valid_ds)} validação")

    # ------------------------- modelo -------------------------------- #
    arch = config["model"]["arch"]
    if approach == "flat":
        learn = create_flat_learner(dls, arch=arch)
    else:
        weights = tuple(float(w) for w in config["model"]["loss_weights"])
        learn = create_multihead_learner(dls, taxonomy, arch=arch, loss_weights=weights)

    # ------------------------- learning rate ------------------------- #
    # lr do YAML se definido; caso contrário, lr_find com a heurística
    # 'valley' (ponto de descida estável da curva loss vs lr). O valor
    # usado é sempre logado, escolha de LR é decisão documentada, não magia.
    base_lr = config["train"].get("base_lr")
    if base_lr is None:
        suggestion = learn.lr_find(show_plot=False)
        base_lr = float(suggestion.valley)
        print(f"[hierclf] lr_find (valley): {base_lr:.2e}")
    else:
        base_lr = float(base_lr)
    logger.log_epoch(-1, base_lr=base_lr)  # época -1 = registro pré-treino

    # ------------------------- treino -------------------------------- #
    callbacks = [
        SaveModelCallback(monitor="valid_loss", fname=f"best_{approach}"),
        EarlyStoppingCallback(monitor="valid_loss", patience=int(config["train"]["patience"])),
        make_tracking_callback(logger),
    ]
    learn.fine_tune(
        int(config["train"]["epochs"]),
        base_lr=base_lr,
        freeze_epochs=int(config["train"]["freeze_epochs"]),
        cbs=callbacks,
    )

    # ------------------------- exportação ---------------------------- #
    # learn.export salva o pipeline completo (transforms + vocab + pesos):
    # é o artefato que a inferência e o app Gradio consomem via load_learner.
    export_path = logger.dir / f"export_{approach}.pkl"
    learn.export(export_path)
    print(f"[hierclf] modelo exportado: {export_path}")

    # Métricas finais da melhor época (SaveModelCallback já recarregou o
    # melhor checkpoint ao fim do treino).
    final = learn.validate()
    # learn.metrics embrulha as funções em AvgMetric, que expõe .name
    # (ex.: "accuracy"), não __name__; str(m) seria o repr do objeto.
    metric_names = ["valid_loss"] + [
        getattr(m, "name", None) or getattr(m, "__name__", str(m)) for m in learn.metrics
    ]
    logger.finalize(**dict(zip(metric_names, [float(v) for v in final])))

    return logger.dir


# ---------------------------------------------------------------------- #
# CLI                                                                      #
# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um experimento do hierarchical-product-classifier")
    parser.add_argument("--config", required=True, help="caminho do YAML de configuração")
    parser.add_argument("--experiments-dir", default="experiments", help="raiz dos artefatos de experimentos")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_training(config, experiments_dir=args.experiments_dir)


if __name__ == "__main__":
    main()
