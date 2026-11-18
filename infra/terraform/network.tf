# ---------------------------------------------------------------------------
# VCN: 10.0.0.0/16 leaves room for future subnets (e.g. a DB subnet) without
# re-addressing. Each /24 gives 251 usable IPs (OCI reserves 3 per subnet + network/broadcast).
# Pod (10.42.0.0/16) and Service (10.43.0.0/16) CIDRs are RKE2 defaults and
# deliberately do NOT overlap the VCN, so routing stays unambiguous.
# ---------------------------------------------------------------------------
resource "oci_core_vcn" "main" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = [var.vcn_cidr]
  display_name   = "rag-vcn"
  dns_label      = "ragvcn" # => hosts resolve as <host>.<subnet>.ragvcn.oraclevcn.com
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "rag-igw"
  enabled        = true
}

resource "oci_core_nat_gateway" "nat" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "rag-nat"
}

# Public subnet: default route -> Internet Gateway (two-way internet).
resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "rt-public"
  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

# Private subnet: default route -> NAT (outbound only: pull images, apt, HF models).
resource "oci_core_route_table" "private" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "rt-private"
  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_nat_gateway.nat.id
  }
}

# Security lists are left minimal (egress only); real rules live in NSGs so
# they attach to specific VNICs instead of whole subnets.
resource "oci_core_security_list" "egress_only" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "sl-egress-only"
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }
}

resource "oci_core_subnet" "public" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.main.id
  cidr_block        = var.public_subnet_cidr
  display_name      = "public"
  dns_label         = "pub"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.egress_only.id]
}

resource "oci_core_subnet" "private" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.main.id
  cidr_block                 = var.private_subnet_cidr
  display_name               = "private"
  dns_label                  = "priv"
  route_table_id             = oci_core_route_table.private.id
  security_list_ids          = [oci_core_security_list.egress_only.id]
  prohibit_public_ip_on_vnic = true # nodes can never get a public IP
}

# ---------------------------- NSGs -----------------------------------------
resource "oci_core_network_security_group" "bastion" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "nsg-bastion"
}

resource "oci_core_network_security_group" "lb" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "nsg-lb"
}

resource "oci_core_network_security_group" "nodes" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "nsg-k8s-nodes"
}

# Bastion: SSH only from the admin's IPs (one rule per location).
resource "oci_core_network_security_group_security_rule" "bastion_ssh_in" {
  for_each                  = toset(var.admin_cidrs)
  network_security_group_id = oci_core_network_security_group.bastion.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = each.value
  source_type               = "CIDR_BLOCK"
  tcp_options {
    destination_port_range {
      min = 22
      max = 22
    }
  }
}

resource "oci_core_network_security_group_security_rule" "bastion_egress" {
  network_security_group_id = oci_core_network_security_group.bastion.id
  direction                 = "EGRESS"
  protocol                  = "all"
  destination               = "0.0.0.0/0"
  destination_type          = "CIDR_BLOCK"
}

# Load balancer: HTTP/HTTPS from anywhere (this is the only public entry point for app traffic).
resource "oci_core_network_security_group_security_rule" "lb_in" {
  for_each                  = toset(["80", "443"])
  network_security_group_id = oci_core_network_security_group.lb.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = "0.0.0.0/0"
  source_type               = "CIDR_BLOCK"
  tcp_options {
    destination_port_range {
      min = tonumber(each.value)
      max = tonumber(each.value)
    }
  }
}

resource "oci_core_network_security_group_security_rule" "lb_to_nodes" {
  network_security_group_id = oci_core_network_security_group.lb.id
  direction                 = "EGRESS"
  protocol                  = "6"
  destination               = oci_core_network_security_group.nodes.id
  destination_type          = "NETWORK_SECURITY_GROUP"
}

# Nodes: everything from other nodes (RKE2 supervisor 9345, apiserver 6443,
# kubelet 10250, etcd 2379-2380, Canal VXLAN 8472/udp, etc.).
resource "oci_core_network_security_group_security_rule" "nodes_intra" {
  network_security_group_id = oci_core_network_security_group.nodes.id
  direction                 = "INGRESS"
  protocol                  = "all"
  source                    = oci_core_network_security_group.nodes.id
  source_type               = "NETWORK_SECURITY_GROUP"
}

# Nodes: SSH + kube-apiserver only from the bastion.
resource "oci_core_network_security_group_security_rule" "nodes_from_bastion" {
  for_each                  = toset(["22", "6443"])
  network_security_group_id = oci_core_network_security_group.nodes.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = oci_core_network_security_group.bastion.id
  source_type               = "NETWORK_SECURITY_GROUP"
  tcp_options {
    destination_port_range {
      min = tonumber(each.value)
      max = tonumber(each.value)
    }
  }
}

# Nodes: ingress-nginx (hostPort 80/443 on every node) only from the LB.
resource "oci_core_network_security_group_security_rule" "nodes_from_lb" {
  for_each                  = toset(["80", "443"])
  network_security_group_id = oci_core_network_security_group.nodes.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = oci_core_network_security_group.lb.id
  source_type               = "NETWORK_SECURITY_GROUP"
  tcp_options {
    destination_port_range {
      min = tonumber(each.value)
      max = tonumber(each.value)
    }
  }
}

resource "oci_core_network_security_group_security_rule" "nodes_egress" {
  network_security_group_id = oci_core_network_security_group.nodes.id
  direction                 = "EGRESS"
  protocol                  = "all"
  destination               = "0.0.0.0/0"
  destination_type          = "CIDR_BLOCK"
}
