import json

import httpx
import pytest

from TelegramNotifier import NotifierError, TelegramNotifier
from TelegramNotifier.__main__ import main


TOKEN = "123456:test_token"


def ok(result):
    return httpx.Response(200, json={"ok": True, "result": result})


def test_text_is_plain_and_uses_numeric_destination():
    def handler(request):
        assert request.url.path.endswith("/sendMessage")
        assert json.loads(request.content) == {"chat_id": 123, "text": "Done: <sample> _1_ 🧪"}
        return ok({"message_id": 42})

    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(handler)) as notifier:
        assert notifier.send_text("Done: <sample> _1_ 🧪")["message_id"] == 42


@pytest.mark.parametrize("kind,method,field,name", [
    ("send_image", "sendPhoto", "photo", "plot.png"),
    ("send_document", "sendDocument", "document", "results.csv"),
])
def test_files_are_uploaded_as_multipart(tmp_path, kind, method, field, name):
    path = tmp_path / name
    path.write_bytes(b"sample payload")

    def handler(request):
        assert request.url.path.endswith("/" + method)
        assert "multipart/form-data" in request.headers["content-type"]
        body = request.read()
        assert f'name="{field}"; filename="{name}"'.encode() in body
        assert b"sample payload" in body
        assert b'name="caption"' in body and b"My caption" in body
        assert b'name="chat_id"' in body and b"123" in body
        return ok({"message_id": 43})

    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(handler)) as notifier:
        assert getattr(notifier, kind)(path, "My caption")["message_id"] == 43
    # An open upload handle would prevent replacement/deletion on Windows.
    path.unlink()


def test_link_ignores_other_users_and_group_messages(tmp_path):
    config_path = tmp_path / "config.local.json"
    config_path.write_text(json.dumps({"bot_token": TOKEN}))

    def message(chat_id, username, kind="private", sender_id=None):
        return {"message": {"chat": {"id": chat_id, "type": kind},
                            "from": {"id": sender_id or chat_id, "username": username}}}

    def handler(request):
        if request.url.path.endswith("/getWebhookInfo"):
            return ok({"url": ""})
        assert request.url.path.endswith("/getUpdates")
        assert json.loads(request.content) == {"timeout": 0, "limit": 100}
        return ok([
            message(99, "someone_else"), message(-123, "colnio", "group"),
            message(888, "colnio", sender_id=123), message(123, "ColNio"),
        ])

    with TelegramNotifier(TOKEN, transport=httpx.MockTransport(handler)) as notifier:
        assert notifier.link(config_path=config_path) == 123
        assert notifier.chat_id == 123
    saved = json.loads(config_path.read_text())
    assert saved == {"bot_token": TOKEN, "username": "colnio", "chat_id": 123}
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("webhook", [False, True])
def test_unresolved_link_does_not_save_or_delete_webhook(tmp_path, webhook):
    calls = []

    def handler(request):
        calls.append(request.url.path.rsplit("/", 1)[-1])
        return ok({"url": "https://example.com/hook" if webhook else ""}) if calls[-1] == "getWebhookInfo" else ok([])

    with TelegramNotifier(TOKEN, transport=httpx.MockTransport(handler)) as notifier:
        with pytest.raises(NotifierError, match="webhook" if webhook else "Send /start"):
            notifier.link(config_path=tmp_path / "config.local.json")
    assert calls == (["getWebhookInfo"] if webhook else ["getWebhookInfo", "getUpdates"])
    assert not list(tmp_path.iterdir())


def test_existing_chat_must_match_username_before_link(tmp_path):
    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(
        lambda request: ok({"id": 123, "type": "private", "username": "someone_else"})
    )) as notifier:
        with pytest.raises(NotifierError, match="does not belong"):
            notifier.link(config_path=tmp_path / "config.local.json")
    assert not list(tmp_path.iterdir())


def test_environment_override_and_no_token_written(tmp_path, monkeypatch):
    config_path = tmp_path / "config.local.json"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERNAME", "colnio")
    with TelegramNotifier.from_config(config_path) as notifier:
        assert notifier.chat_id == 123
        notifier._client.close()
        notifier._client = httpx.Client(transport=httpx.MockTransport(
            lambda request: ok({"id": 123, "type": "private", "username": "colnio"})
        ))
        notifier.link(config_path=config_path)
    assert "bot_token" not in json.loads(config_path.read_text())


@pytest.mark.parametrize("status,body,match", [
    (401, {"ok": False, "description": "Unauthorized"}, "Unauthorized"),
    (403, {"ok": False, "description": "bot was blocked by the user"}, "blocked"),
    (429, {"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 10}}, "Retry after 10"),
    (500, ["bad response"], "invalid response"),
])
def test_api_failures_are_actionable_and_not_retried(status, body, match):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(handler)) as notifier:
        with pytest.raises(NotifierError, match=match) as error:
            notifier.send_text("test")
    assert len(calls) == 1
    assert TOKEN not in str(error.value)
    if status == 429:
        assert error.value.retry_after == 10


def test_timeout_does_not_expose_token_or_retry():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout(str(request.url), request=request)

    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(handler)) as notifier:
        with pytest.raises(NotifierError, match="Delivery may be unknown") as error:
            notifier.send_text("test")
    assert TOKEN not in str(error.value)
    assert len(calls) == 1


def test_non_json_response_is_sanitized():
    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(
        lambda request: httpx.Response(502, text="Failed URL: " + str(request.url))
    )) as notifier:
        with pytest.raises(NotifierError, match="HTTP 502") as error:
            notifier.send_text("test")
    assert TOKEN not in str(error.value)


def test_invalid_inputs_make_no_requests(tmp_path):
    def handler(request):
        pytest.fail("Invalid input must not reach Telegram")

    with TelegramNotifier(TOKEN, 123, transport=httpx.MockTransport(handler)) as notifier:
        for text in ("", "  ", "x" * 4097):
            with pytest.raises(NotifierError):
                notifier.send_text(text)
        with pytest.raises(NotifierError, match="File not found"):
            notifier.send_document(tmp_path / "missing.csv")
        path = tmp_path / "large.png"
        with path.open("wb") as handle:
            handle.truncate(10 * 1024 * 1024 + 1)
        with pytest.raises(NotifierError, match="10 MB"):
            notifier.send_image(path)
        with pytest.raises(NotifierError, match="1024"):
            notifier.send_image(path, "x" * 1025)
        notifier.chat_id = None
        with pytest.raises(NotifierError, match="not linked"):
            notifier.send_text("test")


def test_cli_unconfigured_exits_cleanly(tmp_path, monkeypatch, capsys):
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_USERNAME"):
        monkeypatch.delenv(name, raising=False)
    assert main(["--config", str(tmp_path / "missing.json"), "text", "hello"]) == 1
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err
