from __future__ import annotations


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 500, **details: object):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details

    def payload(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}
