#!/bin/sh
# Only cluster peers may send VXLAN packets. Does not flush host/Docker rules.
set -eu
iptables -w -N RTL-K3S-VXLAN 2>/dev/null || true
for peer in 172.18.5.188 172.18.4.199 172.18.5.123; do
  iptables -w -C RTL-K3S-VXLAN -s "$peer" -j ACCEPT 2>/dev/null || iptables -w -A RTL-K3S-VXLAN -s "$peer" -j ACCEPT
done
iptables -w -C RTL-K3S-VXLAN -j DROP 2>/dev/null || iptables -w -A RTL-K3S-VXLAN -j DROP
iptables -w -C INPUT -p udp --dport 8472 -j RTL-K3S-VXLAN 2>/dev/null || iptables -w -I INPUT 1 -p udp --dport 8472 -j RTL-K3S-VXLAN
