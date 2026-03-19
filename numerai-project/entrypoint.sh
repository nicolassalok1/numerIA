#!/bin/bash
set -euo pipefail

echo "============================================"
echo "  Numerai Pipeline - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# -----------------------------------------------
# 0. Load secrets from AWS Secrets Manager (if running on AWS)
# -----------------------------------------------
if [ -n "${AWS_REGION:-}" ] && [ -n "${NUMERAI_SECRET_ARN:-}" ]; then
    echo "[1/6] Loading credentials from AWS Secrets Manager..."
    SECRETS=$(python -c "
import boto3, json
client = boto3.client('secretsmanager', region_name='${AWS_REGION}')
resp = client.get_secret_value(SecretId='${NUMERAI_SECRET_ARN}')
print(resp['SecretString'])
")
    export NUMERAI_PUBLIC_ID=$(echo "$SECRETS" | python -c "import sys,json; print(json.load(sys.stdin)['NUMERAI_PUBLIC_ID'])")
    export NUMERAI_SECRET_KEY=$(echo "$SECRETS" | python -c "import sys,json; print(json.load(sys.stdin)['NUMERAI_SECRET_KEY'])")
    export NUMERAI_MODEL_ID=$(echo "$SECRETS" | python -c "import sys,json; print(json.load(sys.stdin)['NUMERAI_MODEL_ID'])")
    echo "  Credentials loaded."
else
    echo "[1/6] Using credentials from environment variables."
fi

# -----------------------------------------------
# 1. Download data from Numerai
# -----------------------------------------------
echo "[2/6] Downloading Numerai data..."
python -c "
from pathlib import Path
from numerapi import NumerAPI
napi = NumerAPI()
Path('data').mkdir(exist_ok=True)
print('  Downloading training data...')
napi.download_dataset('v5.0/train.parquet', 'data/numerai_training_data.parquet')
print('  Downloading live data...')
napi.download_dataset('v5.0/live.parquet', 'data/numerai_tournament_data.parquet')
print('  Data download complete.')
"

# -----------------------------------------------
# 2. Optuna optimization (optional, controlled by env var)
# -----------------------------------------------
PARAMS_FILE="${PARAMS_FILE:-config/program_input_params.yaml}"

if [ "${RUN_OPTUNA:-0}" = "1" ]; then
    OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
    echo "[3/6] Running Optuna optimization ($OPTUNA_TRIALS trials)..."
    python src/optimize.py --n-trials "$OPTUNA_TRIALS"
    PARAMS_FILE="config/optimized_params.yaml"
    echo "  Using optimized params: $PARAMS_FILE"
else
    echo "[3/6] Skipping Optuna (set RUN_OPTUNA=1 to enable)."
fi

# -----------------------------------------------
# 3. Train models
# -----------------------------------------------
echo "[4/6] Training models..."
python src/train.py \
    --config config/training.yaml \
    --params "$PARAMS_FILE" \
    --features config/features.yaml

# -----------------------------------------------
# 4. Generate predictions
# -----------------------------------------------
echo "[5/6] Generating predictions..."
python src/predict.py \
    --config config/training.yaml \
    --params "$PARAMS_FILE" \
    --features config/features.yaml

# -----------------------------------------------
# 5. Upload to Numerai
# -----------------------------------------------
if [ -n "${NUMERAI_PUBLIC_ID:-}" ] && [ -n "${NUMERAI_SECRET_KEY:-}" ] && [ -n "${NUMERAI_MODEL_ID:-}" ]; then
    echo "[6/6] Uploading submission to Numerai..."
    python -c "
import os
from numerapi import NumerAPI
napi = NumerAPI(os.environ['NUMERAI_PUBLIC_ID'], os.environ['NUMERAI_SECRET_KEY'])
resp = napi.upload_predictions('submission.csv', model_id=os.environ['NUMERAI_MODEL_ID'])
print('  Upload response:', resp)
"
    echo "  Submission uploaded."
else
    echo "[6/6] SKIP upload: Numerai credentials not set."
fi

# -----------------------------------------------
# 6. Upload artifacts to S3 (optional)
# -----------------------------------------------
if [ -n "${S3_BUCKET:-}" ]; then
    TIMESTAMP=$(date -u '+%Y%m%d_%H%M%S')
    echo "Uploading artifacts to s3://${S3_BUCKET}/runs/${TIMESTAMP}/..."
    python -c "
import boto3, os, glob
from pathlib import Path
s3 = boto3.client('s3')
bucket = os.environ['S3_BUCKET']
ts = '${TIMESTAMP}'
for f in ['submission.csv', 'models/metrics.json', 'models/model_spec.json', 'models/feature_columns.json']:
    p = Path(f)
    if p.exists():
        key = f'runs/{ts}/{p.name}'
        s3.upload_file(str(p), bucket, key)
        print(f'  Uploaded {f} -> s3://{bucket}/{key}')
# Upload all model pkl files
for f in glob.glob('models/**/*.pkl', recursive=True):
    key = f'runs/{ts}/models/{Path(f).name}'
    s3.upload_file(f, bucket, key)
    print(f'  Uploaded {f} -> s3://{bucket}/{key}')
print('  Artifacts uploaded.')
"
fi

echo ""
echo "============================================"
echo "  Pipeline complete - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# If running on EC2 with auto-shutdown, stop the instance
if [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
    echo "Auto-shutdown enabled. Stopping instance in 60 seconds..."
    sleep 60
    sudo shutdown -h now
fi
