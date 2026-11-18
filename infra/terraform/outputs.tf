output "bastion_public_ip" {
  value = oci_core_instance.bastion.public_ip
}

output "server_private_ip" {
  value = oci_core_instance.server.private_ip
}

output "agent_private_ips" {
  value = oci_core_instance.agent[*].private_ip
}

output "lb_public_ip" {
  value = oci_load_balancer_load_balancer.ingress.ip_address_details[0].ip_address
}

output "app_hostname" {
  description = "nip.io gives free wildcard DNS: <ip>.nip.io resolves to <ip>. Use this as the Ingress host."
  value       = "rag.${oci_load_balancer_load_balancer.ingress.ip_address_details[0].ip_address}.nip.io"
}

output "ssh_tunnel_command" {
  description = "Forwards the Kubernetes API to localhost:6443 through the bastion."
  value       = "ssh -N -L 6443:${oci_core_instance.server.private_ip}:6443 ubuntu@${oci_core_instance.bastion.public_ip}"
}
