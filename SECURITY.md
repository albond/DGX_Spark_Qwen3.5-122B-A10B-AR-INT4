# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly by opening a GitHub issue.

Since this project contains inference optimization scripts and Docker configurations (not a web service), most security concerns would relate to:

- Dockerfile misconfigurations
- Unsafe default settings in launch scripts
- Dependency vulnerabilities in Docker images

## Scope

This project provides Docker build files and launch scripts for local inference on NVIDIA DGX Spark. It does not include any network-facing services beyond the local vLLM API server.
