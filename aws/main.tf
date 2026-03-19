terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  name       = var.project_name
}

# =============================================================
# ECR - Container Registry
# =============================================================
resource "aws_ecr_repository" "numerai" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "numerai" {
  repository = aws_ecr_repository.numerai.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

# =============================================================
# S3 - Artifacts storage
# =============================================================
resource "aws_s3_bucket" "artifacts" {
  bucket        = "${local.name}-artifacts-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "cleanup-old-runs"
    status = "Enabled"
    expiration {
      days = 90
    }
  }
}

# =============================================================
# Secrets Manager - Numerai API keys
# =============================================================
resource "aws_secretsmanager_secret" "numerai_keys" {
  name                    = "${local.name}-api-keys"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "numerai_keys" {
  secret_id = aws_secretsmanager_secret.numerai_keys.id
  secret_string = jsonencode({
    NUMERAI_PUBLIC_ID  = var.numerai_public_id
    NUMERAI_SECRET_KEY = var.numerai_secret_key
    NUMERAI_MODEL_ID   = var.numerai_model_id
  })
}

# =============================================================
# IAM - EC2 instance role
# =============================================================
resource "aws_iam_role" "ec2_role" {
  name = "${local.name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ec2_policy" {
  name = "${local.name}-ec2-policy"
  role = aws_iam_role.ec2_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.numerai_keys.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["ec2:StopInstances"]
        Resource = "*"
        Condition = {
          StringEquals = { "ec2:ResourceTag/Project" = local.name }
        }
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${local.name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# =============================================================
# Security Group
# =============================================================
resource "aws_security_group" "numerai" {
  name        = "${local.name}-sg"
  description = "Numerai pipeline - outbound only"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = var.ssh_key_name != "" ? [1] : []
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  tags = { Project = local.name }
}

# =============================================================
# EC2 Spot GPU Instance
# =============================================================

# Latest Deep Learning AMI (Amazon Linux 2, NVIDIA drivers pre-installed)
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch * (Amazon Linux 2) *"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "aws_launch_template" "numerai" {
  name_prefix   = "${local.name}-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.ec2_instance_type
  key_name      = var.ssh_key_name != "" ? var.ssh_key_name : null

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_profile.name
  }

  vpc_security_group_ids = [aws_security_group.numerai.id]

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price          = var.spot_max_price
      spot_instance_type = "one-time"
    }
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${local.name}-gpu"
      Project = local.name
    }
  }

  user_data = base64encode(<<-USERDATA
#!/bin/bash
set -euo pipefail
exec > /var/log/numerai-pipeline.log 2>&1

echo "=== Numerai Pipeline Start: $(date -u) ==="

# Login to ECR
aws ecr get-login-password --region ${local.region} | \
  docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${local.region}.amazonaws.com

# Pull latest image
IMAGE="${aws_ecr_repository.numerai.repository_url}:latest"
docker pull "$IMAGE"

# Run pipeline
docker run --rm --gpus all \
  -e AWS_REGION="${local.region}" \
  -e NUMERAI_SECRET_ARN="${aws_secretsmanager_secret.numerai_keys.arn}" \
  -e S3_BUCKET="${aws_s3_bucket.artifacts.bucket}" \
  -e RUN_OPTUNA="${var.run_optuna}" \
  -e AUTO_SHUTDOWN=0 \
  "$IMAGE"

echo "=== Numerai Pipeline Done: $(date -u) ==="

# Self-terminate: stop the instance to save cost
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region ${local.region}
  USERDATA
  )
}

# =============================================================
# IAM - Lambda role (to start EC2)
# =============================================================
resource "aws_iam_role" "lambda_role" {
  name = "${local.name}-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${local.name}-lambda-policy"
  role = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ec2:RunInstances", "ec2:CreateTags", "iam:PassRole"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

# =============================================================
# Lambda - Starts EC2 spot instance
# =============================================================
resource "aws_lambda_function" "start_pipeline" {
  function_name = "${local.name}-start-pipeline"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  runtime       = "python3.12"
  timeout       = 30

  filename = "${path.module}/lambda.zip"

  environment {
    variables = {
      LAUNCH_TEMPLATE_ID = aws_launch_template.numerai.id
    }
  }

  depends_on = [data.archive_file.lambda]
}

data "archive_file" "lambda" {
  type        = "zip"
  output_path = "${path.module}/lambda.zip"
  source {
    content  = <<-PYTHON
import boto3
import os

def handler(event, context):
    ec2 = boto3.client('ec2')
    template_id = os.environ['LAUNCH_TEMPLATE_ID']

    resp = ec2.run_instances(
        LaunchTemplate={'LaunchTemplateId': template_id, 'Version': '$Latest'},
        MinCount=1,
        MaxCount=1,
    )

    instance_id = resp['Instances'][0]['InstanceId']
    print(f'Started instance: {instance_id}')
    return {'instanceId': instance_id}
    PYTHON
    filename = "index.py"
  }
}

# =============================================================
# EventBridge - Weekly schedule
# =============================================================
resource "aws_cloudwatch_event_rule" "weekly" {
  name                = "${local.name}-weekly-run"
  description         = "Trigger Numerai pipeline weekly"
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "lambda" {
  rule = aws_cloudwatch_event_rule.weekly.name
  arn  = aws_lambda_function.start_pipeline.arn
}

resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.start_pipeline.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly.arn
}
