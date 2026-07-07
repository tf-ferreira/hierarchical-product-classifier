"""
Taxonomia hierárquica do catálogo de produtos.

Este módulo é a FONTE ÚNICA DA VERDADE do mapeamento hierárquico:

    articleType  ->  subCategory  ->  masterCategory
    (ex.: Tshirts ->  Topwear     ->  Apparel)

Por que um módulo separado?
    1. A abordagem "flat" treina apenas no nível mais fino (articleType) e
       DERIVA os níveis superiores por lookup aqui. Consistência garantida
       por construção.
    2. A abordagem "multi-head" prevê os três níveis de forma independente,
       então PODE produzir combinações impossíveis (ex.: master=Footwear com
       article=Tshirts). Este módulo fornece a função que MEDE essa
       inconsistência, uma das métricas centrais da comparação.
    3. Por ser Python puro (sem fastai/torch), é trivialmente testável com
       pytest e roda em qualquer ambiente.

Observação sobre os dados: no Fashion Product Images, o mapeamento
articleType -> (subCategory, masterCategory) é essencialmente funcional
(cada articleType tem um único pai), mas construímos a taxonomia por VOTO
MAJORITÁRIO e reportamos conflitos, em vez de assumir a limpeza dos dados.
Nunca confie em catálogo de e-commerce sem verificar.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Ordem canônica dos níveis, do mais grosso ao mais fino.
# Toda a base de código referencia esta tupla, nunca strings soltas.
LEVELS: tuple[str, str, str] = ("masterCategory", "subCategory", "articleType")


class Taxonomy:
    """Mapeamento imutável articleType -> (subCategory, masterCategory)."""

    def __init__(self, article_to_parents: dict[str, tuple[str, str]]):
        # dict: {"Tshirts": ("Topwear", "Apparel"), ...}
        self.article_to_parents = dict(article_to_parents)

    # ------------------------------------------------------------------ #
    # Construção                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> tuple["Taxonomy", list[dict]]:
        """Constrói a taxonomia a partir do styles.csv já carregado.

        Retorna (taxonomia, conflitos). Um "conflito" é um articleType que
        aparece com mais de um par (subCategory, masterCategory) nos dados.
        Resolvemos por voto majoritário e devolvemos o registro do conflito
        para que o chamador logue/reporte (transparência > silêncio).
        """
        conflicts: list[dict] = []
        mapping: dict[str, tuple[str, str]] = {}

        cols = ["subCategory", "masterCategory"]
        for article, group in df.groupby("articleType")[cols]:
            counts = group.value_counts()  # pares (sub, master) ordenados por freq.
            sub, master = counts.index[0]  # par majoritário
            mapping[str(article)] = (str(sub), str(master))
            if len(counts) > 1:
                conflicts.append(
                    {
                        "articleType": str(article),
                        "pares_observados": {str(k): int(v) for k, v in counts.items()},
                        "par_escolhido": (str(sub), str(master)),
                    }
                )
        return cls(mapping), conflicts

    # ------------------------------------------------------------------ #
    # Consulta                                                            #
    # ------------------------------------------------------------------ #
    def parents_of(self, article: str) -> tuple[str, str]:
        """Retorna (subCategory, masterCategory) do articleType dado.

        Levanta KeyError para articleType desconhecido: preferimos falhar
        alto e cedo a devolver None e propagar erro silencioso.
        """
        return self.article_to_parents[article]

    def is_consistent(self, master: str, sub: str, article: str) -> bool:
        """Verifica se a tripla respeita a hierarquia da taxonomia.

        Usada na avaliação do multi-head: para cada exemplo, as três
        predições independentes formam uma tripla; a taxa de triplas
        consistentes é reportada no README.
        """
        expected = self.article_to_parents.get(article)
        if expected is None:
            return False
        return expected == (sub, master)

    @property
    def articles(self) -> list[str]:
        return sorted(self.article_to_parents)

    def vocab(self, level: str) -> list[str]:
        """Vocabulário ordenado (determinístico) de um nível.

        Passamos vocabulários EXPLÍCITOS aos CategoryBlocks do fastai em vez
        de deixá-lo inferir: garante o mesmo mapeamento índice->classe entre
        runs e entre as duas abordagens (pré-requisito da comparação justa).
        """
        if level == "articleType":
            return self.articles
        idx = 0 if level == "subCategory" else 1
        if level not in LEVELS:
            raise ValueError(f"Nível desconhecido: {level!r}. Use um de {LEVELS}.")
        return sorted({parents[idx] for parents in self.article_to_parents.values()})

    # ------------------------------------------------------------------ #
    # Persistência                                                        #
    # ------------------------------------------------------------------ #
    def to_json(self, path: str | Path) -> None:
        """Serializa como artefato do treino.

        O app Gradio e o módulo de inferência carregam este JSON para
        derivar os níveis superiores sem depender do CSV original.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {a: list(p) for a, p in self.article_to_parents.items()}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Taxonomy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({a: (p[0], p[1]) for a, p in raw.items()})
