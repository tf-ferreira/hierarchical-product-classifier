"""
Testes da Taxonomy, o módulo mais crítico do projeto.

Por que "mais crítico"? Porque tanto a inferência do flat (derivação dos
pais) quanto a métrica central do multi-head (consistência hierárquica)
dependem dele. Um bug aqui contamina TODA a comparação entre abordagens.
Por ser Python puro, testar é barato; não há desculpa para não testar.
"""
import pandas as pd
import pytest

from hierclf.taxonomy import LEVELS, Taxonomy


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Catálogo mínimo com hierarquia limpa + um conflito proposital."""
    rows = [
        # articleType, subCategory, masterCategory
        ("Tshirts", "Topwear", "Apparel"),
        ("Tshirts", "Topwear", "Apparel"),
        ("Tshirts", "Topwear", "Apparel"),
        # Conflito: 1 linha de Tshirts com pai errado (simula erro de catálogo).
        ("Tshirts", "Bottomwear", "Apparel"),
        ("Casual Shoes", "Shoes", "Footwear"),
        ("Casual Shoes", "Shoes", "Footwear"),
        ("Handbags", "Bags", "Accessories"),
    ]
    return pd.DataFrame(rows, columns=["articleType", "subCategory", "masterCategory"])


def test_voto_majoritario_resolve_conflito(toy_df):
    tax, conflicts = Taxonomy.from_dataframe(toy_df)
    # O par majoritário de Tshirts é (Topwear, Apparel): 3 votos contra 1.
    assert tax.parents_of("Tshirts") == ("Topwear", "Apparel")
    # E o conflito foi REPORTADO, não engolido.
    assert len(conflicts) == 1
    assert conflicts[0]["articleType"] == "Tshirts"


def test_parents_of_desconhecido_falha_alto(toy_df):
    tax, _ = Taxonomy.from_dataframe(toy_df)
    with pytest.raises(KeyError):
        tax.parents_of("Naves Espaciais")


def test_is_consistent(toy_df):
    tax, _ = Taxonomy.from_dataframe(toy_df)
    assert tax.is_consistent("Apparel", "Topwear", "Tshirts") is True
    # Tripla impossível: sapato que é Topwear.
    assert tax.is_consistent("Apparel", "Topwear", "Casual Shoes") is False
    # articleType desconhecido nunca é consistente.
    assert tax.is_consistent("Apparel", "Topwear", "Naves") is False


def test_vocab_ordenado_e_deterministico(toy_df):
    tax, _ = Taxonomy.from_dataframe(toy_df)
    assert tax.vocab("articleType") == ["Casual Shoes", "Handbags", "Tshirts"]
    assert tax.vocab("subCategory") == ["Bags", "Shoes", "Topwear"]
    assert tax.vocab("masterCategory") == ["Accessories", "Apparel", "Footwear"]
    with pytest.raises(ValueError):
        tax.vocab("nivelInexistente")


def test_roundtrip_json(tmp_path, toy_df):
    """Serializar e recarregar deve preservar o mapeamento exato:
    é o contrato entre o treino (que salva) e o app (que carrega)."""
    tax, _ = Taxonomy.from_dataframe(toy_df)
    path = tmp_path / "taxonomy.json"
    tax.to_json(path)
    reloaded = Taxonomy.from_json(path)
    assert reloaded.article_to_parents == tax.article_to_parents


def test_ordem_dos_niveis_e_canonica():
    """Guard rail: se alguém mudar LEVELS, muita coisa quebra em cascata
    (ordem das cabeças, dos alvos, dos pesos da loss). Este teste transforma
    essa mudança silenciosa em falha explícita."""
    assert LEVELS == ("masterCategory", "subCategory", "articleType")
