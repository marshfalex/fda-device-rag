variable "region" {
  description = "OCI home region, e.g. us-ashburn-1. Always Free A1 capacity only exists in your home region."
  type        = string
}

variable "oci_profile" {
  description = "Profile name in ~/.oci/config."
  type        = string
  default     = "DEFAULT"
}

variable "compartment_ocid" {
  description = "Compartment to build everything in (the tenancy OCID works for a personal account)."
  type        = string
}

variable "ssh_public_key_path" {
  description = "Public key installed on every VM."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "admin_cidrs" {
  description = "Public IPs allowed to SSH to the bastion, in CIDR form (curl ifconfig.me then append /32). One per place you work from, e.g. home and school."
  type        = list(string)
}

variable "vcn_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  type    = string
  default = "10.0.0.0/24"
}

variable "private_subnet_cidr" {
  type    = string
  default = "10.0.1.0/24"
}

# Always Free Ampere budget: 4 OCPU + 24 GB RAM total across all A1 VMs.
variable "server_ocpus" {
  type    = number
  default = 2
}

variable "server_memory_gb" {
  type    = number
  default = 12
}

variable "agent_count" {
  type    = number
  default = 2
}

variable "agent_ocpus" {
  type    = number
  default = 1
}

variable "agent_memory_gb" {
  type    = number
  default = 6
}

variable "rke2_version" {
  description = "Pin RKE2 so rebuilds are reproducible. See https://github.com/rancher/rke2/releases"
  type        = string
  default     = "v1.31.4+rke2r1"
}

# Always Free quirks: the E2.1.Micro is only allowed in ONE availability domain
# of the home region, and A1 capacity varies per AD. Pick each independently
# (1-based; run `oci iam availability-domain list` to see how many you have).
variable "bastion_ad_number" {
  type    = number
  default = 1
}

variable "nodes_ad_number" {
  type    = number
  default = 1
}
