param(
    [Parameter(Position=0)]
    [ValidateSet("setup", "build", "run", "destroy", "logs", "help")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path (Split-Path -Parent $ScriptDir) "numerai-project"
$AwsDir = $ScriptDir

function Log   { param([string]$msg) Write-Host "[+] $msg" -ForegroundColor Green }
function Warn  { param([string]$msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Err   { param([string]$msg) Write-Error $msg; exit 1 }

function Check-Prereqs {
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
        Err "AWS CLI not installed. Install: https://aws.amazon.com/cli/"
    }
    if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
        Err "Terraform not installed. Install: https://terraform.io/downloads"
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Err "Docker not installed."
    }
    try { aws sts get-caller-identity | Out-Null }
    catch { Err "AWS CLI not configured. Run: aws configure" }
}

function Get-EcrUrl {
    Set-Location $AwsDir
    return (terraform output -raw ecr_repository_url 2>$null)
}

function Get-AwsRegion {
    Set-Location $AwsDir
    $r = terraform output -raw aws_region 2>$null
    if (-not $r) { $r = aws configure get region }
    return $r
}

# ----- SETUP -----
function Cmd-Setup {
    Check-Prereqs

    Set-Location $AwsDir
    if (-not (Test-Path "terraform.tfvars")) {
        Err "terraform.tfvars not found. Copy terraform.tfvars.example and fill in your values."
    }

    Log "Step 1/4: Initializing Terraform..."
    terraform init

    Log "Step 2/4: Creating AWS infrastructure..."
    terraform apply -auto-approve

    Log "Step 3/4: Building Docker image..."
    Cmd-Build

    Log "Step 4/4: Setup complete!"
    Write-Host ""
    terraform output
    Write-Host ""
    Log "To trigger a manual run: .\deploy.ps1 run"
    Log "The pipeline will also run automatically on schedule."
}

# ----- BUILD -----
function Cmd-Build {
    Check-Prereqs

    $EcrUrl = Get-EcrUrl
    $Region = Get-AwsRegion
    $AccountId = (aws sts get-caller-identity --query Account --output text)

    Log "Logging in to ECR..."
    $password = aws ecr get-login-password --region $Region
    $password | docker login --username AWS --password-stdin "${AccountId}.dkr.ecr.${Region}.amazonaws.com"

    Log "Building Docker image..."
    Set-Location $ProjectDir
    docker build -t numerai-pipeline .

    Log "Tagging and pushing to ECR..."
    docker tag numerai-pipeline:latest "${EcrUrl}:latest"
    docker push "${EcrUrl}:latest"

    Log "Image pushed: ${EcrUrl}:latest"
}

# ----- RUN -----
function Cmd-Run {
    Check-Prereqs

    Set-Location $AwsDir
    $LambdaName = (terraform output -raw lambda_function)

    Log "Triggering pipeline via Lambda: $LambdaName"
    $outFile = [System.IO.Path]::GetTempFileName()
    aws lambda invoke `
        --function-name $LambdaName `
        --payload '{}' `
        --cli-binary-format raw-in-base64-out `
        $outFile | Out-Null

    Get-Content $outFile
    Remove-Item $outFile -ErrorAction SilentlyContinue

    Log "EC2 spot instance starting. It will auto-stop after the pipeline completes."
    Log "Check logs: .\deploy.ps1 logs"
}

# ----- DESTROY -----
function Cmd-Destroy {
    Check-Prereqs
    Warn "This will destroy ALL AWS resources for this project."
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Cancelled."
        return
    }

    Set-Location $AwsDir
    terraform destroy -auto-approve
    Log "All resources destroyed."
}

# ----- LOGS -----
function Cmd-Logs {
    Check-Prereqs

    Set-Location $AwsDir
    $S3Bucket = terraform output -raw s3_bucket 2>$null

    if ($S3Bucket) {
        Log "Latest runs in S3:"
        aws s3 ls "s3://${S3Bucket}/runs/" --recursive | Select-Object -Last 20
    }

    Log "Recent EC2 instances:"
    aws ec2 describe-instances `
        --filters "Name=tag:Project,Values=numerai" `
        --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Launch:LaunchTime}" `
        --output table
}

# ----- MAIN -----
switch ($Command) {
    "setup"   { Cmd-Setup }
    "build"   { Cmd-Build }
    "run"     { Cmd-Run }
    "destroy" { Cmd-Destroy }
    "logs"    { Cmd-Logs }
    default {
        Write-Host ""
        Write-Host "Usage: .\deploy.ps1 {setup|build|run|destroy|logs}" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  setup   - First time: create AWS infra + build + push Docker image"
        Write-Host "  build   - Build and push Docker image to ECR"
        Write-Host "  run     - Trigger a manual pipeline run now"
        Write-Host "  destroy - Tear down all AWS resources"
        Write-Host "  logs    - Show latest pipeline logs and runs"
        Write-Host ""
    }
}
