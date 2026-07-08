"""
Camada de dados: download, validação e construção dos DataLoaders.

Decisões de design documentadas:

1.  VALIDAÇÃO EXPLÍCITA. O Fashion Product Images tem imperfeições
    conhecidas: linhas malformadas no styles.csv e alguns IDs sem imagem
    correspondente. Em vez de esconder isso, cada etapa de limpeza CONTA e
    RETORNA o que removeu (ver DataReport), e os números vão para o README.
    Dado sujo tratado silenciosamente é bug em produção esperando para
    acontecer.

2.  SPLIT ESTRATIFICADO COM SEED. O dataset tem cauda longa (articleTypes
    com pouquíssimos exemplos). Um split aleatório simples pode deixar
    classes raras sem representação na validação, tornando a métrica
    enganosa. Estratificamos por articleType (o nível mais fino, que induz
    estratificação nos níveis superiores) e fixamos a seed para que flat e
    multi-head vejam EXATAMENTE o mesmo split, pré-requisito da comparação.

3.  IMPORTS DO FASTAI DENTRO DA FUNÇÃO. Somente build_dataloaders importa
    fastai. Todo o resto é pandas puro, então os testes de validação/split
    rodam em qualquer máquina, sem GPU e sem instalar torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .taxonomy import LEVELS, Taxonomy

# Slugs do Kaggle. A versão "small" (~80x60 px) baixa em minutos e permite
# iterar rápido; a "full" (2400x1600 px) melhora o teto de acurácia ao custo
# de ~25 GB. A escolha é feita no YAML (dataset.version), não no código.
KAGGLE_SLUGS = {
    "small": "paramaggarwal/fashion-product-images-small",
    "full": "paramaggarwal/fashion-product-images-dataset",
}


# ---------------------------------------------------------------------- #
# Relatório de qualidade dos dados                                        #
# ---------------------------------------------------------------------- #
@dataclass
class DataReport:
    """Contabilidade de tudo que a limpeza removeu (transparência total)."""

    linhas_brutas: int = 0
    linhas_malformadas: int = 0          # linhas puladas na leitura do CSV
    linhas_sem_rotulo: int = 0           # NaN em algum dos 3 níveis
    imagens_ausentes: int = 0            # id no CSV sem arquivo .jpg no disco
    classes_raras_removidas: int = 0     # articleTypes abaixo do mínimo
    exemplos_de_classes_raras: int = 0   # linhas removidas por classe rara
    linhas_reetiquetadas: int = 0        # pais reescritos p/ o par majoritário
    linhas_finais: int = 0
    conflitos_taxonomia: list = field(default_factory=list)

    def resumo(self) -> str:
        return (
            f"Linhas brutas: {self.linhas_brutas} | "
            f"malformadas: {self.linhas_malformadas} | "
            f"sem rótulo: {self.linhas_sem_rotulo} | "
            f"imagens ausentes: {self.imagens_ausentes} | "
            f"classes raras removidas: {self.classes_raras_removidas} "
            f"({self.exemplos_de_classes_raras} linhas) | "
            f"reetiquetadas pela taxonomia: {self.linhas_reetiquetadas} | "
            f"finais: {self.linhas_finais}"
        )


# ---------------------------------------------------------------------- #
# Download                                                                 #
# ---------------------------------------------------------------------- #
def download_dataset(version: str = "small") -> Path:
    """Baixa o dataset via kagglehub e retorna a raiz local.

    kagglehub cuida de cache: chamadas repetidas não baixam de novo.
    Exige as credenciais do Kaggle configuradas (arquivo ~/.kaggle/kaggle.json,
    gerado em Kaggle > Settings > API > Create New Token).
    """
    import kagglehub  # import local: só é necessário quando de fato baixamos

    if version not in KAGGLE_SLUGS:
        raise ValueError(f"version deve ser um de {list(KAGGLE_SLUGS)}, veio {version!r}")
    path = Path(kagglehub.dataset_download(KAGGLE_SLUGS[version]))

    # A estrutura interna varia entre as versões do dataset no Kaggle
    # (às vezes há um nível extra de diretório). Procuramos o styles.csv
    # em vez de assumir o layout.
    candidates = list(path.rglob("styles.csv"))
    if not candidates:
        raise FileNotFoundError(f"styles.csv não encontrado sob {path}")
    return candidates[0].parent


# ---------------------------------------------------------------------- #
# Carga e validação                                                       #
# ---------------------------------------------------------------------- #
def load_and_validate(
    data_root: str | Path,
    min_samples_per_class: int = 20,
) -> tuple[pd.DataFrame, Taxonomy, DataReport]:
    """Pipeline completo de preparação tabular.

    Passos (cada um contabilizado no DataReport):
      1. Lê styles.csv pulando linhas malformadas (e contando quantas).
      2. Remove linhas com rótulo ausente em qualquer nível.
      3. Resolve o caminho da imagem e remove ids sem arquivo no disco.
      4. Remove articleTypes com menos de `min_samples_per_class` exemplos.
         Justificativa: com <20 exemplos, o split estratificado deixa ~4
         imagens na validação, e qualquer métrica sobre isso é ruído. Essa
         é uma decisão de escopo declarada, não uma limpeza escondida.
      5. Constrói a Taxonomy (com relato de conflitos).
      6. CANONICALIZA os níveis superiores segundo a Taxonomy: linhas cujo
         par (subCategory, masterCategory) perdeu o voto majoritário são
         reescritas para o par vencedor (e contadas). Sem isto, um rótulo
         minoritário (ex.: subCategory "Perfumes" quando o par majoritário
         do articleType é outro) não existiria no vocabulário das cabeças
         do multi-head e quebraria o treino com KeyError no DataLoader.

    Retorna (df pronto, taxonomia, relatório).
    """
    data_root = Path(data_root)
    report = DataReport()

    csv_path = data_root / "styles.csv"
    # Conta linhas brutas do arquivo para sabermos quantas o parser pulou.
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        raw_lines = sum(1 for _ in f) - 1  # -1 do cabeçalho
    report.linhas_brutas = raw_lines

    # on_bad_lines="skip": o styles.csv tem descrições com vírgulas sem
    # aspas em algumas linhas. Pular é aceitável aqui porque são poucas e
    # o custo de recuperá-las não paga o benefício; o número exato fica
    # registrado no relatório.
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    report.linhas_malformadas = raw_lines - len(df)

    # Rótulos ausentes em qualquer um dos três níveis.
    before = len(df)
    df = df.dropna(subset=list(LEVELS)).copy()
    report.linhas_sem_rotulo = before - len(df)

    # Caminho da imagem: images/<id>.jpg. Checamos existência no disco.
    images_dir = data_root / "images"
    df["image_path"] = df["id"].astype(int).map(lambda i: str(images_dir / f"{i}.jpg"))
    exists = df["image_path"].map(lambda p: Path(p).exists())
    report.imagens_ausentes = int((~exists).sum())
    df = df[exists].copy()

    # Cauda longa: remove classes com poucos exemplos (decisão de escopo).
    counts = df["articleType"].value_counts()
    rare = counts[counts < min_samples_per_class].index
    report.classes_raras_removidas = len(rare)
    report.exemplos_de_classes_raras = int(df["articleType"].isin(rare).sum())
    df = df[~df["articleType"].isin(rare)].copy()

    taxonomy, conflicts = Taxonomy.from_dataframe(df)
    report.conflitos_taxonomia = conflicts

    # Canonicalização: a Taxonomy é a fonte única da verdade, então o df de
    # treino deve refletir exatamente o mapeamento dela. Isso também torna os
    # alvos do multi-head consistentes com a derivação usada pelo flat na
    # avaliação (comparação justa entre abordagens).
    pares = [taxonomy.parents_of(a) for a in df["articleType"]]
    canon_sub = pd.Series([p[0] for p in pares], index=df.index)
    canon_master = pd.Series([p[1] for p in pares], index=df.index)
    report.linhas_reetiquetadas = int(
        ((df["subCategory"] != canon_sub) | (df["masterCategory"] != canon_master)).sum()
    )
    df["subCategory"] = canon_sub
    df["masterCategory"] = canon_master

    report.linhas_finais = len(df)

    return df.reset_index(drop=True), taxonomy, report


# ---------------------------------------------------------------------- #
# Split estratificado                                                     #
# ---------------------------------------------------------------------- #
def stratified_valid_indices(
    df: pd.DataFrame,
    valid_pct: float = 0.2,
    seed: int = 42,
    stratify_col: str = "articleType",
) -> np.ndarray:
    """Índices de validação estratificados por classe, com seed fixa.

    Implementação com pandas puro (sem sklearn) para manter as dependências
    mínimas: amostramos `valid_pct` de cada grupo. max(1, ...) garante ao
    menos 1 exemplo de validação por classe.
    """
    rng = np.random.default_rng(seed)
    valid_idx: list[int] = []
    for _, group in df.groupby(stratify_col):
        n_valid = max(1, int(round(len(group) * valid_pct)))
        chosen = rng.choice(group.index.to_numpy(), size=n_valid, replace=False)
        valid_idx.extend(chosen.tolist())
    return np.sort(np.array(valid_idx))


# ---------------------------------------------------------------------- #
# DataLoaders (única função que toca o fastai)                            #
# ---------------------------------------------------------------------- #
def build_dataloaders(
    df: pd.DataFrame,
    taxonomy: Taxonomy,
    approach: str,
    img_size: int = 224,
    batch_size: int = 64,
    valid_pct: float = 0.2,
    seed: int = 42,
):
    """Constrói os DataLoaders do fastai para uma das duas abordagens.

    approach="flat":
        blocks = (ImageBlock, CategoryBlock)          -> 1 alvo: articleType
    approach="multihead":
        blocks = (ImageBlock, CB, CB, CB), n_inp=1    -> 3 alvos, na ordem
        canônica de LEVELS (master, sub, article)

    Pontos importantes:
    - Vocabulários EXPLÍCITOS vindos da Taxonomy: mesmo mapeamento
      índice->classe em todos os runs (comparabilidade e inferência estável).
    - IndexSplitter com o split estratificado: flat e multi-head veem o
      MESMO conjunto de validação.
    - Augmentation moderada e justificada: produtos de catálogo são fotos
      padronizadas (fundo branco, produto centralizado), então flips
      horizontais + pequenas variações de zoom/iluminação simulam variação
      realista de catálogo; rotações agressivas ou warp de perspectiva
      criariam exemplos fora da distribuição real.
    - Normalize com estatísticas do ImageNet: obrigatório em transfer
      learning, o backbone pré-treinado espera entradas nessa distribuição.
    """
    # Import local proposital: o resto do módulo funciona sem fastai/torch.
    from fastai.vision.all import (
        CategoryBlock,
        ColReader,
        DataBlock,
        ImageBlock,
        IndexSplitter,
        Normalize,
        Resize,
        aug_transforms,
        imagenet_stats,
    )

    valid_idx = stratified_valid_indices(df, valid_pct=valid_pct, seed=seed)
    splitter = IndexSplitter(valid_idx)

    common = dict(
        get_x=ColReader("image_path"),
        splitter=splitter,
        # Resize no item (CPU, uma imagem por vez) para uniformizar tamanhos;
        # augmentations no batch (GPU, mais rápido).
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(
                do_flip=True,          # flip horizontal: um tênis espelhado ainda é um tênis
                flip_vert=False,       # produto de cabeça para baixo não ocorre em catálogo
                max_rotate=5.0,        # leve tolerância de enquadramento
                max_zoom=1.1,
                max_lighting=0.2,
                max_warp=0.0,          # sem warp: fotos de estúdio não têm perspectiva variável
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    if approach == "flat":
        block = DataBlock(
            blocks=(ImageBlock, CategoryBlock(vocab=taxonomy.vocab("articleType"))),
            get_y=ColReader("articleType"),
            **common,
        )
    elif approach == "multihead":
        block = DataBlock(
            blocks=(
                ImageBlock,
                CategoryBlock(vocab=taxonomy.vocab("masterCategory")),
                CategoryBlock(vocab=taxonomy.vocab("subCategory")),
                CategoryBlock(vocab=taxonomy.vocab("articleType")),
            ),
            n_inp=1,  # 1 entrada (imagem), o resto são alvos
            get_y=[ColReader(level) for level in LEVELS],
            **common,
        )
    else:
        raise ValueError(f"approach deve ser 'flat' ou 'multihead', veio {approach!r}")

    return block.dataloaders(df, bs=batch_size)
