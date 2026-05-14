from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


DEFAULT_PATHS = (
    "/",
    "/api",
    "/api/status",
    "/status",
    "/health",
    "/version",
    "/swagger",
    "/swagger/index.html",
    "/openapi.json",
    "/docs",
)


@dataclass
class ProbeResult:
    url: str
    ok: bool
    status: int | None
    content_type: str
    body_preview: str
    error: str | None


class SuragusHttpClient:
    def __init__(self, host: str, port: int, use_https: bool = False, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.use_https = use_https
        self.timeout = timeout

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
        handler = urllib.request.HTTPSHandler(context=context)
        return urllib.request.build_opener(handler)

    def probe_path(self, path: str) -> ProbeResult:
        url = urllib.parse.urljoin(f"{self.base_url}/", path.lstrip("/"))
        request = urllib.request.Request(url, method="GET")
        opener = self._build_opener()
        try:
            with opener.open(request, timeout=self.timeout) as response:
                body = response.read(512).decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
                return ProbeResult(
                    url=url,
                    ok=True,
                    status=getattr(response, "status", None),
                    content_type=content_type,
                    body_preview=body.strip(),
                    error=None,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read(256).decode("utf-8", errors="replace")
            return ProbeResult(
                url=url,
                ok=False,
                status=exc.code,
                content_type=exc.headers.get("Content-Type", ""),
                body_preview=body.strip(),
                error=f"HTTP {exc.code}",
            )
        except Exception as exc:
            return ProbeResult(
                url=url,
                ok=False,
                status=None,
                content_type="",
                body_preview="",
                error=str(exc),
            )

    def probe_common_paths(self) -> list[ProbeResult]:
        return [self.probe_path(path) for path in DEFAULT_PATHS]


def tcp_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def result_to_dict(result: ProbeResult) -> dict[str, object]:
    return {
        "url": result.url,
        "ok": result.ok,
        "status": result.status,
        "content_type": result.content_type,
        "body_preview": result.body_preview,
        "error": result.error,
    }


def results_to_json(results: list[ProbeResult]) -> str:
    return json.dumps([result_to_dict(item) for item in results], indent=2)
