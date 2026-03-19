variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  default     = "numerai"
}

variable "numerai_public_id" {
  description = "Numerai API public ID"
  type        = string
  sensitive   = true
}

variable "numerai_secret_key" {
  description = "Numerai API secret key"
  type        = string
  sensitive   = true
}

variable "numerai_model_id" {
  description = "Numerai model ID"
  type        = string
  sensitive   = true
}

variable "ec2_instance_type" {
  description = "EC2 instance type (GPU). g4dn.xlarge = T4 16GB, ~$0.16/hr spot"
  default     = "g4dn.xlarge"
}

variable "spot_max_price" {
  description = "Max spot price per hour (USD). Set to 0 for on-demand."
  default     = "0.25"
}

variable "schedule_expression" {
  description = "EventBridge schedule for weekly runs. Default: every Saturday at 18:00 UTC"
  default     = "cron(0 18 ? * SAT *)"
}

variable "run_optuna" {
  description = "Run Optuna optimization (1=yes, 0=no)"
  default     = "0"
}

variable "ssh_key_name" {
  description = "EC2 SSH key pair name (optional, for debugging)"
  default     = ""
}
