from __future__ import annotations

import argparse
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed

from suragus_client import SuragusHttpClient, result_to_dict, tcp_port_open


COMMON_PORTS = (80, 443, 8080, 8443)


def iter_hosts(cidr: str, limit: int | None) -> list[str]:
    network = ipaddress.ip_network(cidr, strict=False)
    hosts: list[str] = []
    for host in network.hosts():
        hosts.append(str(host))
        if limit is not None and len(hosts) >= limit:
            break
    return hosts


def scan_host(host: str, connect_timeout: float) -> dict[str, object]:
    open_ports = [port for port in COMMON_PORTS if tcp_port_open(host, port, timeout=connect_timeout)]
    return {"host": host, "open_ports": open_ports}


def probe_host(host: str, ports: list[int], http_timeout: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for port in ports:
        client = SuragusHttpClient(host, port, use_https=(port in (443, 8443)), timeout=http_timeout)
        for result in client.probe_common_paths():
            row = result_to_dict(result)
            row["host"] = host
            row["port"] = port
            output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and probe a SURAGUS sheet resistance device.")
    parser.add_argument("--cidr", default="169.254.181.0/24", help="CIDR to scan for the device.")
    parser.add_argument("--host", help="Probe one specific host instead of scanning a CIDR.")
    parser.add_argument("--limit", type=int, default=512, help="Maximum number of hosts to scan from the CIDR.")
    parser.add_argument("--workers", type=int, default=64, help="Parallel scan workers.")
    parser.add_argument("--connect-timeout", type=float, default=0.2, help="TCP connect timeout in seconds.")
    parser.add_argument("--http-timeout", type=float, default=1.0, help="HTTP probe timeout in seconds.")
    parser.add_argument("--probe", action="store_true", help="Probe common HTTP paths on discovered hosts.")
    args = parser.parse_args()

    if args.host:
        targets = [args.host]
    else:
        targets = iter_hosts(args.cidr, args.limit)

    discovered: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(scan_host, host, args.connect_timeout): host for host in targets
        }
        for future in as_completed(future_map):
            result = future.result()
            if result["open_ports"]:
                discovered.append(result)
                print(f"[open] {result['host']} ports={result['open_ports']}")

    if not discovered:
        print("No hosts with open common web ports were found.")
        return

    if not args.probe:
        return

    for item in discovered:
        host = str(item["host"])
        ports = list(item["open_ports"])
        print(f"\n[probe] {host}")
        for result in probe_host(host, ports, args.http_timeout):
            status = result["status"]
            error = result["error"]
            content_type = result["content_type"]
            preview = str(result["body_preview"]).replace("\n", " ")[:120]
            print(
                f"{result['port']:>5} {result['url']} "
                f"status={status} type={content_type!r} error={error!r} preview={preview!r}"
            )


if __name__ == "__main__":
    main()
