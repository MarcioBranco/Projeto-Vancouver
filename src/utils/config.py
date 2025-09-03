from pathlib import Path

# Diretórios principais
ROOT = Path.home() / "proj-van"
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_processed"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_MODELS = ARTIFACTS / "models"
ARTIFACTS_PLOTS = ARTIFACTS / "plots"
ARTIFACTS_PDFS = ARTIFACTS / "pdfs"
DEPLOY_DIR = ROOT / "deploy"

# URLs de fontes (podem ser configuradas via variável de ambiente)
VANCOUVER_311_CSV_URL = None
VANCOUVER_BUDGET_CSV_URL = None

# Criar diretórios se não existirem
for d in [DATA_RAW, DATA_PROCESSED, ARTIFACTS_MODELS, ARTIFACTS_PLOTS, ARTIFACTS_PDFS, DEPLOY_DIR]:
    d.mkdir(parents=True, exist_ok=True)
