# AI / Open Data / Scorecards Públicos — MVP

## Objetivo
Transformar datasets públicos (ex.: 3-1-1, orçamentos) em scorecards, previsões e relatórios públicos para gerar insights e material para vídeos ou parcerias com prefeituras.

## Estrutura do Projeto
~/proj-van/
├─ data_raw/ # Dados brutos baixados (não versionar grandes dumps)
├─ data_processed/ # Dados processados e normalizados
├─ notebooks/ # Notebooks de exploração e experimentação
├─ src/
│ ├─ ingest/ # Scripts para download de datasets
│ ├─ normalize/ # Mapeamento e normalização de taxonomia
│ ├─ features/ # Construção de features para modelos
│ ├─ models/ # Treinamento e avaliação de modelos
│ ├─ reports/ # Geração de relatórios PDF/HTML
│ └─ utils/ # Configurações e funções utilitárias
├─ artifacts/
│ ├─ models/ # Modelos treinados (.pkl)
│ ├─ plots/ # Gráficos gerados
│ └─ pdfs/ # Relatórios PDF gerados
├─ deploy/ # Scripts de publicação em OCI
├─ requirements.txt # Dependências Python
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
# Próximos Passos
* Normalização de dados com src/normalize/taxonomy_map.py

* Treinamento de modelos com src/models/train_forecast.py

* Geração de relatórios com src/reports/render_report.py

* Publicação de artefatos com deploy/publish_to_oci.sh