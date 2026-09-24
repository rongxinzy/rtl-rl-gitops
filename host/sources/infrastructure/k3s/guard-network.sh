#!/bin/sh
# Only cluster peers may send VXLAN packets. Does not flush host/Docker rules.
set -eu
iptables -w -N RTL-K3S-VXLAN 2>/dev/null || true
for peer in 172.18.5.188 172.18.4.199 172.18.5.123; do
  iptables -w -C RTL-K3S-VXLAN -s "$peer" -j ACCEPT 2>/dev/null || iptables -w -A RTL-K3S-VXLAN -s "$peer" -j ACCEPT
done
iptables -w -C RTL-K3S-VXLAN -j DROP 2>/dev/null || iptables -w -A RTL-K3S-VXLAN -j DROP
iptables -w -C INPUT -p udp --dport 8472 -j RTL-K3S-VXLAN 2>/dev/null || iptables -w -I INPUT 1 -p udp --dport 8472 -j RTL-K3S-VXLAN

# The L20 uses the control host's API, kubelet, data bridge, and WireGuard.
# Keep these grants scoped to its fixed host address; the WireGuard accept is
# inserted before the existing per-peer drop rule.
iptables -w -C INPUT -s 172.18.5.123 -p udp --dport 51822 -j ACCEPT 2>/dev/null || iptables -w -I INPUT 1 -s 172.18.5.123 -p udp --dport 51822 -j ACCEPT
allow_l20_host() {
  ufw allow from 172.18.5.123 to 172.18.5.188 port "$1" proto "$2" comment "$3" >/dev/null
}
allow_l20_host 6443 tcp rtl-k3s-api
allow_l20_host 10250 tcp rtl-k3s-kubelet
allow_l20_host 15432 tcp rtl-new-api-db-bridge
allow_l20_host 16379 tcp rtl-new-api-redis-bridge
allow_l20_host 51822 udp rtl-l20-wireguard
