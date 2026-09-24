#!/bin/sh
# Only cluster peers may send VXLAN packets. Does not flush host/Docker rules.
set -eu
iptables -w -N RTL-K3S-VXLAN 2>/dev/null || true
for peer in 172.18.5.188 172.18.4.199 172.18.5.123; do
  while iptables -w -C RTL-K3S-VXLAN -s "$peer" -j ACCEPT 2>/dev/null; do
    iptables -w -D RTL-K3S-VXLAN -s "$peer" -j ACCEPT
  done
  iptables -w -I RTL-K3S-VXLAN 1 -s "$peer" -j ACCEPT
done
iptables -w -C RTL-K3S-VXLAN -j DROP 2>/dev/null || iptables -w -A RTL-K3S-VXLAN -j DROP
iptables -w -C INPUT -p udp --dport 8472 -j RTL-K3S-VXLAN 2>/dev/null || iptables -w -I INPUT 1 -p udp --dport 8472 -j RTL-K3S-VXLAN

# Only the control host owns the API, data bridges, and primary WireGuard peer.
if ip -o -4 addr show | grep -q '172.18.5.188/'; then
  iptables -w -C INPUT -s 172.18.5.123 -p udp --dport 51822 -j ACCEPT 2>/dev/null || iptables -w -I INPUT 1 -s 172.18.5.123 -p udp --dport 51822 -j ACCEPT
  ufw allow from 172.18.5.123 to 172.18.5.188 port 6443 proto tcp comment rtl-k3s-api >/dev/null
  ufw allow from 172.18.5.123 to 172.18.5.188 port 10250 proto tcp comment rtl-k3s-kubelet >/dev/null
  ufw allow from 172.18.5.123 to 172.18.5.188 port 15432 proto tcp comment rtl-new-api-db-bridge >/dev/null
  ufw allow from 172.18.5.123 to 172.18.5.188 port 16379 proto tcp comment rtl-new-api-redis-bridge >/dev/null
  ufw allow from 172.18.5.123 to 172.18.5.188 port 51822 proto udp comment rtl-l20-wireguard >/dev/null
fi
