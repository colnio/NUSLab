from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .errors import ServiceError
from .service import LabService


class SessionRequest(BaseModel):
    sample_name: str = Field(min_length=1, max_length=200)
    devices: dict[str, str]
    notes: dict | None = None


class ConfigureRequest(BaseModel):
    source_mode: Literal["voltage", "current"] | None = None
    function: str | None = None
    nplc: float | None = None
    source_range: float | None = None
    sense_autorange: bool | None = None
    sense_range: float | None = None
    compliance: float | None = None
    range_auto: bool | None = None
    range_value: float | None = None


class SetpointRequest(BaseModel):
    value: float


class OutputRequest(BaseModel):
    enabled: bool


class ReadRequest(BaseModel):
    function: str | None = None


class SampleRequest(BaseModel):
    devices: list[str]


class ScpiRequest(BaseModel):
    kind: Literal["write", "query"]
    command: str


class SweepRequest(BaseModel):
    device: str
    voltage_points_v: list[float]
    nplc: float = 1.0
    source_delay_s: float = 0.0
    compliance_a: float | None = None
    sense_range_a: float | None = None
    compliance_action: Literal["stop", "continue"] = "stop"


class RecoverRequest(BaseModel):
    resource: str


def configure_logging(log_dir: Path) -> RotatingFileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("GPIBServer")
    for old in logger.handlers[:]:
        logger.removeHandler(old)
        old.close()
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_dir / "server.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return handler


def create_app(service: LabService | None = None) -> FastAPI:
    lab = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal lab
        if lab is None:
            lab = LabService()
        app.state.lab = lab
        handler = configure_logging(lab.store.recovery_root.parent)
        lab.start()
        try:
            yield
        finally:
            lab.stop()
            logger = logging.getLogger("GPIBServer")
            logger.removeHandler(handler)
            handler.close()

    app = FastAPI(title="NUSLab GPIB Server", version="1.0.0", lifespan=lifespan)
    app.state.lab = lab

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"error": exc.payload()})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request",
                                                                    "message": "Request validation failed",
                                                                    "details": {"fields": jsonable_encoder(exc.errors())}}})

    @app.exception_handler(OSError)
    async def storage_error_handler(request: Request, exc: OSError):
        logging.getLogger("GPIBServer").exception("Filesystem error on %s", request.url.path)
        return JSONResponse(status_code=507, content={"error": {"code": "storage_error",
                                                                    "message": str(exc), "details": {}}})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logging.getLogger("GPIBServer").exception("Unexpected error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                                                                    "message": "Unexpected server error", "details": {}}})

    @app.get("/v1/health")
    def health():
        return lab.health()

    @app.get("/v1/devices")
    def devices():
        found = lab.discover()
        return {"devices": found, "scan_errors": lab.bus.discovery_errors}

    @app.post("/v1/devices/recover")
    def recover(request: RecoverRequest):
        return lab.recover_device(request.resource)

    @app.post("/v1/sessions", status_code=201)
    def create_session(request: SessionRequest):
        return lab.create_session(request.sample_name, request.devices, request.notes)

    @app.get("/v1/sessions/{session_id}")
    def session(session_id: str):
        return lab.session_snapshot(lab.get_session(session_id))

    @app.post("/v1/sessions/{session_id}/heartbeat")
    def heartbeat(session_id: str):
        return lab.heartbeat(session_id)

    @app.delete("/v1/sessions/{session_id}")
    def release(session_id: str):
        return lab.close_session(session_id)

    @app.post("/v1/sessions/{session_id}/devices/{alias}/configure")
    def configure(session_id: str, alias: str, request: ConfigureRequest):
        return lab.configure(session_id, alias, request.model_dump(exclude_none=True))

    @app.get("/v1/sessions/{session_id}/devices/{alias}")
    def device_status(session_id: str, alias: str):
        return lab.device_status(session_id, alias)

    @app.post("/v1/sessions/{session_id}/devices/{alias}/setpoint")
    def setpoint(session_id: str, alias: str, request: SetpointRequest):
        return lab.setpoint(session_id, alias, request.value)

    @app.post("/v1/sessions/{session_id}/devices/{alias}/output")
    def output(session_id: str, alias: str, request: OutputRequest):
        return lab.output(session_id, alias, request.enabled)

    @app.post("/v1/sessions/{session_id}/devices/{alias}/read")
    def read(session_id: str, alias: str, request: ReadRequest | None = None):
        return lab.read(session_id, alias, request.function if request else None)

    @app.post("/v1/sessions/{session_id}/samples")
    def grouped_sample(session_id: str, request: SampleRequest):
        return lab.sample(session_id, request.devices)

    @app.post("/v1/sessions/{session_id}/devices/{alias}/scpi")
    def scpi(session_id: str, alias: str, request: ScpiRequest):
        return lab.raw_scpi(session_id, alias, request.kind, request.command)

    @app.post("/v1/sessions/{session_id}/sweeps", status_code=202)
    def start_sweep(session_id: str, request: SweepRequest):
        return lab.start_sweep(session_id, request.device, request.voltage_points_v, request.nplc,
                               request.source_delay_s, request.compliance_a, request.sense_range_a,
                               request.compliance_action)

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str):
        return lab.get_job(job_id).snapshot()

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return lab.cancel_job(job_id)

    @app.get("/v1/jobs/{job_id}/readings")
    def job_readings(job_id: str, after: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=10000)):
        return lab.job_rows(job_id, after, limit)

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str):
        return lab.store.load(run_id)

    return app


app = create_app()
