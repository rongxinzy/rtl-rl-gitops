# Restricted SwanLab CONNECT proxy

A dedicated two-replica CPU service on `rtl-control` exposes NodePort 31443, with `externalTrafficPolicy: Local`. L20 Docker training uses the separately created `rtl-swanlab` bridge; neither host default routing nor the existing Kubernetes Pod egress policy changes.

The proxy accepts authenticated CONNECT requests only to these exact names on port 443:

- `api.swanlab.cn`
- `swanlab-beta-private-1301372061.cos.ap-nanjing.myqcloud.com`

The second hostname was observed from SwanLab SDK 0.10.0's real Text resource upload. Wildcard cloud-storage domains are intentionally not allowed. A new service hostname needs explicit review and configuration change. DNS answers must all be public unicast addresses; connections pin the resolved addresses. The TLS ClientHello must name the same host as CONNECT, preventing cross-host SNI routing on shared addresses. The proxy does not decrypt TLS or disable client certificate verification.

Independent proxy authentication is mounted from `proxy-auth/token`. Public manifests contain only the Secret reference. Provision the token out of band. The client reads its private proxy URL from a root-only mounted file and sets HTTPS_PROXY inside the process, never through Docker command arguments, experiment config, logs, or public manifests. The proxy does not log headers, paths, request bodies, signed URLs, credentials or exception text. NetworkPolicy additionally limits ingress to the three lab host addresses and egress to DNS plus public TCP443. Authentication remains mandatory even where NodePort/NAT source identity is ambiguous.

The server limits active tunnels to 64, header and TLS negotiation to ten seconds each, idle time to sixty seconds and total tunnel time to fifteen minutes. Clients must retry interrupted uploads. This is a telemetry path, not an inference fallback or general-purpose proxy. It cannot upload to private endpoints or arbitrary domains.

Build the local image from this directory, import it into the control node's k3s containerd, provision the Secret, then reconcile `clusters/lab/rtl-swanlab-proxy`. This image currently has a local-image prerequisite; the GitOps tree alone does not provide a complete cold-start recovery.

Test with `python3 -B -m unittest discover -s infrastructure/swanlab_proxy -p 'test_*.py'` (in the GitOps repository, prefix the source path with `src/`). Tests cover strict authority matching, auth, mixed/private DNS, HTTP refusal without echoing secrets, TLS SNI match/mismatch and plaintext refusal. Live acceptance must additionally exercise SDK scalar and Text upload from an isolated CPU container on the real L20 Docker bridge, then check cloud results and scan generated artifacts for secret values without printing any matches.
