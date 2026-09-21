"""Stdio MCP tools for the configured Telegram recipient."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .notifier import NotifierError, TelegramNotifier


def create_server(factory: Callable[[], TelegramNotifier] = TelegramNotifier.from_config) -> FastMCP:
    server = FastMCP(
        "telegram",
        instructions=(
            "Send Telegram notifications to the configured private recipient, @colnio. "
            "Only send messages or files when the user requests or authorizes sending them. "
            "Use absolute local paths for files. Text is plain text. "
            "Sends are not idempotent; after a timeout check delivery before retrying."
        ),
        log_level="WARNING",
    )
    send_annotations = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    )

    def invoke(action: str, *args) -> dict:
        try:
            with factory() as notifier:
                if action == "status":
                    bot = notifier.get_bot()
                    return {
                        "bot_username": bot["username"],
                        "recipient_username": notifier.username,
                        "chat_id": notifier.chat_id,
                        "linked": notifier.chat_id is not None,
                    }
                result = getattr(notifier, action)(*args)
                return {
                    "message_id": result["message_id"],
                    "chat_id": result["chat"]["id"],
                    "recipient_username": notifier.username,
                }
        except (NotifierError, OSError) as exc:
            raise ToolError(str(exc)) from None

    def absolute_path(path: str) -> Path:
        result = Path(path)
        if not result.is_absolute():
            raise ToolError("Provide an absolute local file path.")
        return result

    @server.tool(annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ))
    def status() -> dict:
        """Check the Telegram bot identity and linked recipient without sending a message."""
        return invoke("status")

    @server.tool(annotations=send_annotations)
    def send_text(text: str) -> dict:
        """Send plain text to @colnio (1–4096 characters). Returns the delivery message ID."""
        return invoke("send_text", text)

    @server.tool(annotations=send_annotations)
    def send_image(path: str, caption: str = "") -> dict:
        """Send an image to @colnio from an absolute local path (10 MB; caption up to 1024 characters)."""
        return invoke("send_image", absolute_path(path), caption)

    @server.tool(annotations=send_annotations)
    def send_document(path: str, caption: str = "") -> dict:
        """Send a document or original image to @colnio from an absolute local path (50 MB; caption up to 1024 characters)."""
        return invoke("send_document", absolute_path(path), caption)

    return server


def main() -> None:
    # HTTP logs include bot tokens in request URLs. Stdout is reserved for MCP.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
