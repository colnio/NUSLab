from __future__ import annotations

import json
import math
import random
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from suragus_client import SuragusHttpClient, tcp_port_open


COMMON_PORTS = (80, 443, 8080, 8443)
COMMON_STATUS_PATHS = ("/api/status", "/status", "/health", "/version", "/")


@dataclass
class DeviceCandidate:
    label: str
    host: str
    port: int
    use_https: bool
    status_path: str
    preview: str


@dataclass
class MeasurementResult:
    timestamp: float
    value: float
    unit: str
    latency_s: float
    raw: Any


def iter_hosts(cidr: str, limit: int | None) -> list[str]:
    import ipaddress

    network = ipaddress.ip_network(cidr, strict=False)
    hosts: list[str] = []
    for host in network.hosts():
        hosts.append(str(host))
        if limit is not None and len(hosts) >= limit:
            break
    return hosts


def discover_candidates(
    cidr: str,
    limit: int = 512,
    connect_timeout: float = 0.2,
    http_timeout: float = 1.0,
    cancel_event: threading.Event | None = None,
) -> list[DeviceCandidate]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    hosts = iter_hosts(cidr, limit)
    results: list[DeviceCandidate] = []

    def scan_one(host: str) -> list[DeviceCandidate]:
        if cancel_event and cancel_event.is_set():
            return []

        candidates: list[DeviceCandidate] = []
        for port in COMMON_PORTS:
            if cancel_event and cancel_event.is_set():
                return []
            if not tcp_port_open(host, port, timeout=connect_timeout):
                continue

            use_https = port in (443, 8443)
            client = SuragusHttpClient(host, port, use_https=use_https, timeout=http_timeout)
            for path in COMMON_STATUS_PATHS:
                if cancel_event and cancel_event.is_set():
                    return []
                probe = client.probe_path(path)
                if probe.ok or probe.status is not None:
                    preview = (probe.body_preview or probe.error or "").replace("\n", " ").strip()
                    preview = preview[:80]
                    label = f"{host}:{port} {'HTTPS' if use_https else 'HTTP'}"
                    if preview:
                        label = f"{label} - {preview}"
                    candidates.append(
                        DeviceCandidate(
                            label=label,
                            host=host,
                            port=port,
                            use_https=use_https,
                            status_path=path,
                            preview=preview,
                        )
                    )
                    break
        return candidates

    with ThreadPoolExecutor(max_workers=64) as executor:
        future_map = {executor.submit(scan_one, host): host for host in hosts}
        for future in as_completed(future_map):
            if cancel_event and cancel_event.is_set():
                break
            try:
                results.extend(future.result())
            except Exception:
                continue

    results.sort(key=lambda item: (item.host, item.port, item.use_https))
    return results


class BaseTransport:
    def connect(self) -> dict[str, Any]:
        raise NotImplementedError

    def measure(self) -> MeasurementResult:
        raise NotImplementedError

    def close(self) -> None:
        return


class MockSheetResistanceTransport(BaseTransport):
    def __init__(self) -> None:
        self._counter = 0

    def connect(self) -> dict[str, Any]:
        return {"mode": "mock", "device": "SURAGUS Mock", "status": "connected"}

    def measure(self) -> MeasurementResult:
        self._counter += 1
        baseline = 124.0 + 4.0 * math.sin(self._counter / 3.0)
        value = baseline + random.uniform(-0.8, 0.8)
        return MeasurementResult(
            timestamp=time.time(),
            value=value,
            unit="ohm/sq",
            latency_s=random.uniform(0.05, 0.12),
            raw={
                "sheet_resistance": value,
                "unit": "ohm/sq",
                "sequence": self._counter,
            },
        )


class RestSheetResistanceTransport(BaseTransport):
    def __init__(
        self,
        host: str,
        port: int,
        use_https: bool = False,
        timeout: float = 3.0,
        status_path: str = "/api/status",
        measure_path: str = "/api/measure",
        measure_method: str = "GET",
        payload_text: str = "",
        value_key: str = "",
        unit_key: str = "unit",
    ) -> None:
        self.host = host
        self.port = port
        self.use_https = use_https
        self.timeout = timeout
        self.status_path = status_path or "/"
        self.measure_path = measure_path or "/"
        self.measure_method = measure_method.upper().strip()
        self.payload_text = payload_text.strip()
        self.value_key = value_key.strip()
        self.unit_key = unit_key.strip()

    @property
    def base_url(self) -> str:
        scheme = "https" if self.use_https else "http"
        return f"{scheme}://{self.host}:{self.port}"

    def _build_opener(self) -> urllib.request.OpenerDirector:
        if not self.use_https:
            return urllib.request.build_opener()

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))

    def _make_url(self, path: str) -> str:
        return urllib.parse.urljoin(f"{self.base_url}/", path.lstrip("/"))

    def _request(self, path: str, method: str = "GET", payload_text: str = "") -> tuple[Any, float]:
        headers = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.8"}
        data = None
        payload_text = payload_text.strip()
        if payload_text:
            headers["Content-Type"] = "application/json"
            data = payload_text.encode("utf-8")
        request = urllib.request.Request(self._make_url(path), data=data, headers=headers, method=method)
        opener = self._build_opener()
        t0 = time.perf_counter()
        try:
            with opener.open(request, timeout=self.timeout) as response:
                raw_bytes = response.read()
                latency_s = time.perf_counter() - t0
                content_type = response.headers.get("Content-Type", "")
                text = raw_bytes.decode("utf-8", errors="replace")
                if "json" in content_type.lower():
                    return json.loads(text), latency_s
                try:
                    return json.loads(text), latency_s
                except Exception:
                    return {"text": text, "status": getattr(response, "status", None)}, latency_s
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {path}: {body[:200]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Request to {path} failed: {exc}") from exc

    def connect(self) -> dict[str, Any]:
        data, _ = self._request(self.status_path, method="GET")
        if isinstance(data, dict):
            return data
        return {"status": "connected", "raw": data}

    def measure(self) -> MeasurementResult:
        data, latency_s = self._request(
            self.measure_path,
            method=self.measure_method,
            payload_text=self.payload_text,
        )
        value = extract_numeric_value(data, self.value_key)
        unit = extract_text_value(data, self.unit_key) if self.unit_key else ""
        return MeasurementResult(
            timestamp=time.time(),
            value=value,
            unit=unit or "a.u.",
            latency_s=latency_s,
            raw=data,
        )


def extract_numeric_value(payload: Any, preferred_key: str = "") -> float:
    if preferred_key:
        value = extract_by_dotted_key(payload, preferred_key)
        return float(value)

    for key in (
        "sheet_resistance",
        "sheetResistance",
        "resistance",
        "value",
        "result.value",
        "measurement.value",
    ):
        try:
            value = extract_by_dotted_key(payload, key)
            return float(value)
        except Exception:
            continue
    raise ValueError("Could not extract a numeric measurement value from the response.")


def extract_text_value(payload: Any, preferred_key: str) -> str:
    if not preferred_key:
        return ""
    value = extract_by_dotted_key(payload, preferred_key)
    return str(value)


def extract_by_dotted_key(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in dotted_key.split("."):
        part = part.strip()
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
            continue
        if isinstance(current, list):
            current = current[int(part)]
            continue
        raise KeyError(dotted_key)
    return current
