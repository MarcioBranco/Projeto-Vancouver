#!/usr/bin/env bash
set -euo pipefail

LOCAL_PUBLIC="$HOME/proj-van/public_ready"
mkdir -p "$LOCAL_PUBLIC"

# Copiar artefatos para diretório público
rsync -av --delete \
  ~/proj-van/artifacts/pdfs/ \
  ~/proj-van/artifacts/plots/ \
  ~/proj-van/artifacts/html/ \
  "$LOCAL_PUBLIC/"

# Publicação via rsync para OCI
: "${OCI_HOST:?set OCI_HOST}"
: "${OCI_USER:?set OCI_USER}"
: "${OCI_PATH:?set OCI_PATH}"

rsync -av --delete "$LOCAL_PUBLIC/" "${OCI_USER}@${OCI_HOST}:${OCI_PATH}/"
echo "Published to ${OCI_USER}@${OCI_HOST}:${OCI_PATH}/"
