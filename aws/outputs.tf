output "ecr_repository_url" {
  description = "ECR repository URL - push your Docker image here"
  value       = aws_ecr_repository.numerai.repository_url
}

output "s3_bucket" {
  description = "S3 bucket for artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}

output "lambda_function" {
  description = "Lambda function to manually trigger a run"
  value       = aws_lambda_function.start_pipeline.function_name
}

output "schedule" {
  description = "EventBridge schedule expression"
  value       = var.schedule_expression
}

output "estimated_cost_per_run" {
  description = "Estimated cost per weekly run"
  value       = "~$0.40-0.80 (g4dn.xlarge spot x 2-3h + S3 + ECR)"
}
