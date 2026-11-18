data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_ocid
}

locals {
  ad             = data.oci_identity_availability_domains.ads.availability_domains[0].name
  ssh_public_key = file(pathexpand(var.ssh_public_key_path))
}

# Latest Ubuntu 22.04 images for ARM (nodes) and x86 (bastion micro).
data "oci_core_images" "ubuntu_arm" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

data "oci_core_images" "ubuntu_x86" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.Standard.E2.1.Micro"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "random_password" "rke2_token" {
  length  = 48
  special = false
}

# ---- Bastion (Always Free E2.1.Micro, public subnet) ----
resource "oci_core_instance" "bastion" {
  compartment_id      = var.compartment_ocid
  availability_domain = local.ad
  display_name        = "bastion"
  shape               = "VM.Standard.E2.1.Micro"
  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
    hostname_label   = "bastion"
    nsg_ids          = [oci_core_network_security_group.bastion.id]
  }
  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu_x86.images[0].id
  }
  metadata = {
    ssh_authorized_keys = local.ssh_public_key
  }
}

# ---- RKE2 server (control plane + etcd) ----
resource "oci_core_instance" "server" {
  compartment_id      = var.compartment_ocid
  availability_domain = local.ad
  display_name        = "rke2-server"
  shape               = "VM.Standard.A1.Flex"
  shape_config {
    ocpus         = var.server_ocpus
    memory_in_gbs = var.server_memory_gb
  }
  create_vnic_details {
    subnet_id        = oci_core_subnet.private.id
    assign_public_ip = false
    hostname_label   = "rke2-server"
    private_ip       = cidrhost(var.private_subnet_cidr, 10) # stable IP agents can join
    nsg_ids          = [oci_core_network_security_group.nodes.id]
  }
  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu_arm.images[0].id
    boot_volume_size_in_gbs = 50
  }
  metadata = {
    ssh_authorized_keys = local.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/templates/rke2-server.sh.tftpl", {
      token        = random_password.rke2_token.result
      rke2_version = var.rke2_version
      server_ip    = cidrhost(var.private_subnet_cidr, 10)
    }))
  }
}

# ---- RKE2 agents (workers) ----
resource "oci_core_instance" "agent" {
  count               = var.agent_count
  compartment_id      = var.compartment_ocid
  availability_domain = local.ad
  display_name        = "rke2-agent-${count.index}"
  shape               = "VM.Standard.A1.Flex"
  shape_config {
    ocpus         = var.agent_ocpus
    memory_in_gbs = var.agent_memory_gb
  }
  create_vnic_details {
    subnet_id        = oci_core_subnet.private.id
    assign_public_ip = false
    hostname_label   = "rke2-agent-${count.index}"
    nsg_ids          = [oci_core_network_security_group.nodes.id]
  }
  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu_arm.images[0].id
    boot_volume_size_in_gbs = 50
  }
  metadata = {
    ssh_authorized_keys = local.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/templates/rke2-agent.sh.tftpl", {
      token        = random_password.rke2_token.result
      rke2_version = var.rke2_version
      server_ip    = cidrhost(var.private_subnet_cidr, 10)
    }))
  }
  depends_on = [oci_core_instance.server]
}
