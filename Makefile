# Comandos canônicos do projeto. `make <alvo>`.
.PHONY: setup test train-flat train-multihead evaluate app lint

setup:            ## instala o pacote em modo editável + dependências de dev
	pip install -e ".[dev]"

test:             ## roda a suíte de testes (não exige GPU nem fastai)
	pytest tests/ -v

train-flat:       ## treina o experimento flat (GPU recomendada)
	python -m hierclf.train --config configs/flat.yaml

train-multihead:  ## treina o experimento multi-head
	python -m hierclf.train --config configs/multihead.yaml

evaluate:         ## avalia um run: make evaluate RUN=experiments/multihead-<id>
	python -m hierclf.evaluate --run $(RUN)

app:              ## roda o demo Gradio localmente: make app RUN=experiments/<...>
	RUN_DIR=$(RUN) python app/app.py

lint:             ## checagem estática de estilo
	ruff check src/ tests/ app/
