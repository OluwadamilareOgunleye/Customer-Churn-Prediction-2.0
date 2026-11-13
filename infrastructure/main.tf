# Infrastructure as Code (Terraform) - Telco Churn Prediction System
# Targeted for AWS ECS Fargate, S3 Artifact Storage, and Application Load Balancer

terraform {
  required_version = ">= 1.5.0"
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

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

# 1. Dedicated S3 Bucket for Versioned Model Bundles
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "telco-churn-models-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts_encryption" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 2. Elastic Container Registry (ECR) Repositories
resource "aws_ecr_repository" "api_repo" {
  name                 = "telco-churn-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "frontend_repo" {
  name                 = "telco-churn-frontend"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# 3. ECS Fargate Cluster
resource "aws_ecs_cluster" "app_cluster" {
  name = "telco-churn-cluster-${var.environment}"
}

# 4. CloudWatch Log Group for Application Telemetry
resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/ecs/telco-churn-api"
  retention_in_days = 30
}
