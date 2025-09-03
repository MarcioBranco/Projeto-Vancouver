#!/bin/bash
set -e

# Criar estrutura inicial do projeto
mkdir -p ~/proj-van/{data_raw,data_processed,notebooks,src/{ingest,normalize,features,models,reports,utils},artifacts/{models,plots,pdfs},deploy}

# Criar arquivos vazios básicos
touch ~/proj-van/requirements.txt
touch ~/proj-van/Dockerfile
touch ~/proj-van/docker-compose.yml
touch ~/proj-van/README.md
touch ~/proj-van/deploy/publish_to_oci.sh

echo "Estrutura inicial criada em ~/proj-van/"
