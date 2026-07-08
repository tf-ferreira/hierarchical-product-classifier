"""
Testes do pipeline de preparação de dados (load_and_validate).

O teste de canonicalização é regressão direta de um bug real: linhas cujo
par (subCategory, masterCategory) perdia o voto majoritário mantinham o
rótulo minoritário no DataFrame, rótulo que não existe no vocabulário dos
CategoryBlocks (derivado da Taxonomy) e explodia o treino multi-head com
KeyError dentro do DataLoader (ex.: "Label 'Perfumes' was not included in
the training dataset").

Como o resto da suíte, nada aqui toca fastai/torch: o dataset é sintético,
escrito em tmp_path com imagens vazias (só a EXISTÊNCIA do arquivo importa
para load_and_validate).
"""
import pandas as pd
import pytest

from hierclf.data import load_and_validate
from hierclf.taxonomy import LEVELS


def _make_dataset(tmp_path, rows, missing_images=()):
    """Escreve um styles.csv + images/<id>.jpg sintéticos e retorna a raiz."""
    df = pd.DataFrame(rows, columns=["id", "masterCategory", "subCategory", "articleType"])
    df.to_csv(tmp_path / "styles.csv", index=False)
    images = tmp_path / "images"
    images.mkdir()
    for i in df["id"]:
        if i not in missing_images:
            (images / f"{i}.jpg").write_bytes(b"")
    return tmp_path


@pytest.fixture
def root_com_conflito(tmp_path):
    """3 Tshirts com pais majoritários + 1 com par minoritário (conflito),
    mais uma segunda classe para a taxonomia não ser trivial."""
    rows = [
        (1, "Apparel", "Topwear", "Tshirts"),
        (2, "Apparel", "Topwear", "Tshirts"),
        (3, "Apparel", "Topwear", "Tshirts"),
        (4, "Apparel", "Perfumes", "Tshirts"),  # par minoritário (erro de catálogo)
        (5, "Footwear", "Shoes", "Casual Shoes"),
        (6, "Footwear", "Shoes", "Casual Shoes"),
    ]
    return _make_dataset(tmp_path, rows)


def test_canonicaliza_pais_pelo_voto_majoritario(root_com_conflito):
    df, taxonomy, report = load_and_validate(root_com_conflito, min_samples_per_class=1)

    # A linha minoritária foi REETIQUETADA para o par vencedor, e contada.
    assert report.linhas_reetiquetadas == 1
    assert set(df.loc[df["articleType"] == "Tshirts", "subCategory"]) == {"Topwear"}
    # E o conflito continua reportado (transparência preservada).
    assert len(report.conflitos_taxonomia) == 1


def test_todos_os_rotulos_existem_no_vocabulario(root_com_conflito):
    """A propriedade que quebrou no treino: TODO rótulo presente no df deve
    existir no vocabulário do nível correspondente (senão, KeyError no
    DataLoader do multi-head)."""
    df, taxonomy, _ = load_and_validate(root_com_conflito, min_samples_per_class=1)
    for level in LEVELS:
        assert set(df[level]) <= set(taxonomy.vocab(level)), level


def test_imagem_ausente_contada_e_removida(tmp_path):
    rows = [
        (1, "Apparel", "Topwear", "Tshirts"),
        (2, "Apparel", "Topwear", "Tshirts"),
        (3, "Apparel", "Topwear", "Tshirts"),
    ]
    root = _make_dataset(tmp_path, rows, missing_images={3})
    df, _, report = load_and_validate(root, min_samples_per_class=1)
    assert report.imagens_ausentes == 1
    assert report.linhas_finais == len(df) == 2


def test_classe_rara_removida_e_contada(tmp_path):
    rows = [(i, "Apparel", "Topwear", "Tshirts") for i in range(1, 6)] + [
        (10, "Footwear", "Shoes", "Casual Shoes"),  # classe com 1 exemplo só
    ]
    root = _make_dataset(tmp_path, rows)
    df, taxonomy, report = load_and_validate(root, min_samples_per_class=2)
    assert report.classes_raras_removidas == 1
    assert report.exemplos_de_classes_raras == 1
    assert set(df["articleType"]) == {"Tshirts"}
    # A taxonomia é construída APÓS o corte: classe fora do escopo, fora do vocab.
    assert "Casual Shoes" not in taxonomy.vocab("articleType")
