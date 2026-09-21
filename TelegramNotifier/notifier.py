from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
from pathlib import Path

import httpx


DEFAULT_CONFIG = Path(__file__).with_name("config.local.json")


class NotifierError(RuntimeError):
    """Configuration, connection, or Telegram delivery failure."""

    def __init__(self, message: str, *, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise NotifierError(f"Cannot read JSON configuration: {path}") from None
    if not isinstance(result, dict):
        raise NotifierError("Configuration must be a JSON object.")
    return result


class TelegramNotifier:
    """Synchronous sender; use as a context manager to close HTTP connections.

    No automatic retries: a timeout may occur after Telegram accepted a send.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: int | str | None = None,
        *,
        username: str = "colnio",
        transport: httpx.BaseTransport | None = None,
    ):
        if not isinstance(bot_token, str) or not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", bot_token):
            raise NotifierError("Set TELEGRAM_BOT_TOKEN or bot_token in config.local.json.")
        self._token = bot_token
        self.chat_id = self._parse_chat_id(chat_id)
        if not isinstance(username, str) or not re.fullmatch(r"@?[A-Za-z0-9_]+", username):
            raise NotifierError("Set a valid recipient username.")
        self.username = username.lstrip("@").lower()
        self._client = httpx.Client(timeout=httpx.Timeout(60, connect=10), transport=transport)

    @staticmethod
    def _parse_chat_id(value: int | str | None) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
            raise NotifierError("TELEGRAM_CHAT_ID/chat_id must be a positive private-chat ID.")
        return int(value)

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG) -> TelegramNotifier:
        """Environment variables override the module-local JSON configuration."""
        config = _read_config(Path(path))
        return cls(
            os.environ.get("TELEGRAM_BOT_TOKEN", config.get("bot_token", "")),
            os.environ.get("TELEGRAM_CHAT_ID", config.get("chat_id")),
            username=os.environ.get("TELEGRAM_USERNAME", config.get("username", "colnio")),
        )

    def __enter__(self) -> TelegramNotifier:
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, data: dict | None = None, *, files=None):
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            if files is None:
                response = self._client.post(url, json=data or {})
            else:
                response = self._client.post(url, data=data, files=files)
        except httpx.HTTPError:
            # HTTP exception strings contain the credential-bearing URL.
            raise NotifierError(
                f"Telegram {method} connection failed. Delivery may be unknown; check the chat before retrying."
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise NotifierError(f"Telegram returned an invalid response (HTTP {response.status_code}).") from None
        if not isinstance(payload, dict):
            raise NotifierError("Telegram returned an invalid response.")
        if not response.is_success or payload.get("ok") is not True:
            description = str(payload.get("description", "Request failed")).replace(self._token, "[redacted]")
            parameters = payload.get("parameters") or {}
            retry_after = parameters.get("retry_after") if isinstance(parameters, dict) else None
            suffix = f" Retry after {retry_after} seconds." if retry_after is not None else ""
            raise NotifierError(
                f"Telegram {method}: {description}.{suffix}", retry_after=retry_after
            )
        if "result" not in payload:
            raise NotifierError("Telegram response is missing its result.")
        return payload["result"]

    def get_bot(self) -> dict:
        """Validate the token and return the bot's public identity."""
        return self._request("getMe")

    def link(self, *, wait: int = 0, config_path: str | Path = DEFAULT_CONFIG) -> int:
        """Find the configured user's private chat and persist its numeric ID.

        Reads up to 100 pending updates without acknowledging or discarding them.
        Run setup without another getUpdates poller for this bot.
        """
        if not 0 <= wait <= 50:
            raise NotifierError("Link wait must be between 0 and 50 seconds.")
        if self.chat_id is not None:
            chat = self._request("getChat", {"chat_id": self.chat_id})
            if chat.get("type") != "private" or chat.get("username", "").lower() != self.username:
                raise NotifierError(f"Configured chat does not belong to @{self.username}.")
            found = self.chat_id
        else:
            if self._request("getWebhookInfo").get("url"):
                raise NotifierError(
                    "This bot uses a webhook. Set TELEGRAM_CHAT_ID from its private-chat event, then run link."
                )
            updates = self._request("getUpdates", {"timeout": wait, "limit": 100})
            matches = set()
            for update in updates:
                message = update.get("message", {})
                chat = message.get("chat", {})
                sender = message.get("from", {})
                if (
                    chat.get("type") == "private"
                    and sender.get("username", "").lower() == self.username
                    and sender.get("id") == chat.get("id")
                    and not sender.get("is_bot", False)
                ):
                    matches.add(self._parse_chat_id(chat.get("id")))
            matches.discard(None)
            if len(matches) != 1:
                raise NotifierError(
                    f"No unique private chat for @{self.username}. Send /start to the bot and run link again. "
                    "If another app handles updates or the queue is full, set TELEGRAM_CHAT_ID explicitly."
                )
            found = matches.pop()
        path = Path(config_path)
        config = _read_config(path)
        config.update(username=self.username, chat_id=found)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep environment-supplied tokens in the environment; never copy them to disk.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(config, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.chat_id = found
        return found

    def _destination(self) -> dict:
        if self.chat_id is None:
            raise NotifierError("Recipient is not linked. Run: python -m TelegramNotifier link")
        return {"chat_id": self.chat_id}

    def send_text(self, text: str) -> dict:
        """Send plain text (1–4096 characters); return Telegram's Message object."""
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise NotifierError("Text must contain 1–4096 characters and cannot be blank.")
        return self._request("sendMessage", {**self._destination(), "text": text})

    def send_image(self, path: str | Path, caption: str = "") -> dict:
        """Upload an image as a Telegram photo (up to 10 MB)."""
        return self._send_file("sendPhoto", "photo", path, caption, 10)

    def send_document(self, path: str | Path, caption: str = "") -> dict:
        """Upload a document, or an uncompressed image (up to 50 MB)."""
        return self._send_file("sendDocument", "document", path, caption, 50)

    def _send_file(self, method: str, field: str, path: str | Path, caption: str, max_mb: int) -> dict:
        data = self._destination()
        if not isinstance(caption, str) or len(caption) > 1024:
            raise NotifierError("Captions must contain at most 1024 characters.")
        data["caption"] = caption
        path = Path(path)
        if not path.is_file():
            raise NotifierError(f"File not found: {path}")
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if not 0 < size <= max_mb * 1024 * 1024:
                raise NotifierError(f"File must be nonempty and at most {max_mb} MB.")
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return self._request(method, data, files={field: (path.name, handle, media_type)})
