terraform {
  required_version = ">= 1.6"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

# Auth comes from ~/.oci/config (created by `oci setup config`).
provider "oci" {
  region              = var.region
  config_file_profile = var.oci_profile
}
