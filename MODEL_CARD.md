# Model Card: hierarchical-product-classifier

Documento de transparência sobre o que este modelo é, para que serve e,
principalmente, **onde ele falha**. Formato inspirado em Mitchell et al.,
"Model Cards for Model Reporting" (2019).

## Descrição

Classificador de imagens de produtos de e-commerce em três níveis de uma
taxonomia de moda (`masterCategory → subCategory → articleType`), treinado por
fine-tuning de um ConvNeXt-Tiny pré-treinado no ImageNet, sobre o dataset
Fashion Product Images (~44 mil produtos de um catálogo real).

Duas variantes (ver README): `flat` (1 cabeça + derivação taxonômica) e
`multihead` (3 cabeças independentes).

## Uso pretendido

- Sugestão de categoria no cadastro de produtos de catálogo (human-in-the-loop).
- Detecção de produtos possivelmente mal categorizados (triagem, não decisão).
- Apoio a matching de produtos entre catálogos.
- Fins educacionais e de portfólio: comparação controlada entre arquiteturas.

## Usos NÃO pretendidos

- Decisão automática final de categorização sem revisão humana.
- Classificação de fotos de usuários "in the wild" (ver limitação 1).
- Domínios fora de moda/vestuário (a taxonomia é específica deste catálogo).
- Qualquer inferência sobre pessoas presentes nas imagens.

## Limitações conhecidas

1. **Domain gap de catálogo.** O treino usa fotos padronizadas de estúdio
   (fundo claro, produto centralizado, boa iluminação). Fotos de usuário, com
   fundo complexo, oclusão e iluminação variada, estão fora da distribuição de
   treino e a acurácia reportada NÃO se transfere para elas.
2. **Taxonomia congelada e específica.** O mapeamento hierárquico é o do
   catálogo de origem. Categorias novas ou taxonomias de outros marketplaces
   exigem re-treino/re-mapeamento; a abordagem flat, em particular, é incapaz
   de prever um `articleType` que não existia no treino.
3. **Cauda longa truncada.** Classes com menos de `min_samples_per_class`
   exemplos (padrão: 20) foram removidas do escopo, com contagem reportada
   pelo pipeline. O modelo não conhece essas classes.
4. **Inconsistência hierárquica (variante multihead).** As três cabeças são
   independentes; uma fração das predições forma triplas impossíveis. A taxa
   é medida e reportada em `evaluation.json`, use-a antes de escolher a
   variante para qualquer aplicação.
5. **Vieses do catálogo de origem.** O dataset reflete o sortimento, o mercado
   e as convenções de fotografia de um único e-commerce de moda (recorte
   geográfico e temporal específico). Frequências de classe e aparência dos
   produtos carregam esse viés.
6. **Resolução.** A versão "small" do dataset tem imagens reduzidas; o teto de
   acurácia em classes visualmente próximas (fine-grained) é limitado por
   isso. A config aceita a versão em alta resolução como extensão.

## Dados

Fashion Product Images (Kaggle, licença conforme a página do dataset). O
pipeline de preparação remove e contabiliza: linhas malformadas do CSV,
registros sem rótulo, imagens ausentes e classes raras (`DataReport` em
`src/hierclf/data.py`); os números de cada run ficam no log do treino.

## Métricas e avaliação

Acurácia por nível + taxa de consistência hierárquica, sobre validação
estratificada por `articleType` (20%, seed 42), idêntica entre variantes.
Resultados no README e em `experiments/runs.csv`.
