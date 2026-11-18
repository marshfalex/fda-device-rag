# Always Free includes one flexible LB at 10 Mbps. It sits in the public
# subnet and forwards 80/443 (TCP passthrough) to ingress-nginx on every node.
resource "oci_load_balancer_load_balancer" "ingress" {
  compartment_id = var.compartment_ocid
  display_name   = "rag-ingress-lb"
  shape          = "flexible"
  shape_details {
    minimum_bandwidth_in_mbps = 10
    maximum_bandwidth_in_mbps = 10
  }
  subnet_ids                 = [oci_core_subnet.public.id]
  network_security_group_ids = [oci_core_network_security_group.lb.id]
  is_private                 = false
}

locals {
  node_ips = concat([oci_core_instance.server.private_ip], oci_core_instance.agent[*].private_ip)
  lb_ports = { http = 80, https = 443 }
}

resource "oci_load_balancer_backend_set" "ingress" {
  for_each         = local.lb_ports
  load_balancer_id = oci_load_balancer_load_balancer.ingress.id
  name             = "bs-${each.key}"
  policy           = "ROUND_ROBIN"
  health_checker {
    protocol = "TCP"
    port     = each.value
  }
}

resource "oci_load_balancer_backend" "nodes" {
  for_each = {
    for pair in setproduct(keys(local.lb_ports), range(length(local.node_ips))) :
    "${pair[0]}-${pair[1]}" => { set = pair[0], idx = pair[1] }
  }
  load_balancer_id = oci_load_balancer_load_balancer.ingress.id
  backendset_name  = oci_load_balancer_backend_set.ingress[each.value.set].name
  ip_address       = local.node_ips[each.value.idx]
  port             = local.lb_ports[each.value.set]
}

resource "oci_load_balancer_listener" "ingress" {
  for_each                 = local.lb_ports
  load_balancer_id         = oci_load_balancer_load_balancer.ingress.id
  name                     = "listener-${each.key}"
  default_backend_set_name = oci_load_balancer_backend_set.ingress[each.key].name
  port                     = each.value
  protocol                 = "TCP"
}
