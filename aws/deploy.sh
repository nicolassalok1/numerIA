#!/bin/bash
set -euo pipefail

# =============================================================
# Numerai AWS Deployment Script
#
# Usage:
#   ./deploy.sh setup     # First time: create AWS infra + build + push
#   ./deploy.sh build     # Build and push Docker image only
#   ./deploy.sh run       # Trigger a manual run now
#   ./deploy.sh destroy   # Tear down all AWS resources
#   ./deploy.sh logs      # Show latest pipeline logs
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/numerai-project"
AWS_DIR="$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

check_prereqs() {
    command -v aws >/dev/null 2>&1     || err "AWS CLI not installed. Install: https://aws.amazon.com/cli/"
    command -v terraform >/dev/null 2>&1 || err "Terraform not installed. Install: https://terraform.io/downloads"
    command -v docker >/dev/null 2>&1    || err "Docker not installed."
    aws sts get-caller-identity >/dev/null 2>&1 || err "AWS CLI not configured. Run: aws configure"
}

get_ecr_url() {
    cd "$AWS_DIR"
    terraform output -raw ecr_repository_url 2>/dev/null
}

get_region() {
    cd "$AWS_DIR"
    terraform output -raw 2>/dev/null aws_region || aws configure get region
}

# ----- SETUP: Create infra + build + push -----
cmd_setup() {
    check_prereqs

    log "Step 1/4: Initializing Terraform..."
    cd "$AWS_DIR"
    if [ ! -f terraform.tfvars ]; then
        err "terraform.tfvars not found. Copy terraform.tfvars.example and fill in your values."
    fi
    terraform init

    log "Step 2/4: Creating AWS infrastructure..."
    terraform apply -auto-approve

    log "Step 3/4: Building Docker image..."
    cmd_build

    log "Step 4/4: Setup complete!"
    echo ""
    terraform output
    echo ""
    log "To trigger a manual run: ./deploy.sh run"
    log "The pipeline will also run automatically on schedule."
}

# ----- BUILD: Build and push Docker image -----
cmd_build() {
    check_prereqs

    ECR_URL=$(get_ecr_url)
    REGION=$(get_region)
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

    log "Logging in to ECR..."
    aws ecr get-login-password --region "$REGION" | \
        docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

    log "Building Docker image..."
    cd "$PROJECT_DIR"
    docker build -t numerai-pipeline .

    log "Tagging and pushing to ECR..."
    docker tag numerai-pipeline:latest "${ECR_URL}:latest"
    docker push "${ECR_URL}:latest"

    log "Image pushed: ${ECR_URL}:latest"
}

# ----- RUN: Trigger a manual run now -----
cmd_run() {
    check_prereqs

    cd "$AWS_DIR"
    LAMBDA_NAME=$(terraform output -raw lambda_function)

    log "Triggering pipeline via Lambda: $LAMBDA_NAME"
    RESPONSE=$(aws lambda invoke \
        --function-name "$LAMBDA_NAME" \
        --payload '{}' \
        --cli-binary-format raw-in-base64-out \
        /dev/stdout 2>/dev/null)

    echo "$RESPONSE"
    log "EC2 spot instance starting. It will auto-stop after the pipeline completes."
    log "Check logs: ./deploy.sh logs"
}

# ----- DESTROY: Tear down everything -----
cmd_destroy() {
    check_prereqs
    warn "This will destroy ALL AWS resources for this project."
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi

    cd "$AWS_DIR"
    terraform destroy -auto-approve
    log "All resources destroyed."
}

# ----- LOGS: Show recent pipeline logs -----
cmd_logs() {
    check_prereqs
    cd "$AWS_DIR"
    S3_BUCKET=$(terraform output -raw s3_bucket 2>/dev/null || echo "")

    if [ -n "$S3_BUCKET" ]; then
        log "Latest runs in S3:"
        aws s3 ls "s3://${S3_BUCKET}/runs/" --recursive | tail -20
    fi

    log "Recent EC2 instances:"
    aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=numerai" \
        --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Launch:LaunchTime}" \
        --output table
}

# ----- Main -----
case "${1:-help}" in
    setup)   cmd_setup ;;
    build)   cmd_build ;;
    run)     cmd_run ;;
    destroy) cmd_destroy ;;
    logs)    cmd_logs ;;
    *)
        echo "Usage: $0 {setup|build|run|destroy|logs}"
        echo ""
        echo "  setup   - First time: create AWS infra + build + push Docker image"
        echo "  build   - Build and push Docker image to ECR"
        echo "  run     - Trigger a manual pipeline run now"
        echo "  destroy - Tear down all AWS resources"
        echo "  logs    - Show latest pipeline logs and runs"
        ;;
esac
