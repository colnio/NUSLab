import asyncio
import json
import pytest

pytest.importorskip("mcp")

from TelegramNotifier import NotifierError
from TelegramNotifier.mcp_server import create_server
from mcp.server.fastmcp.exceptions import ToolError


class FakeNotifier:
    username = "colnio"
    chat_id = 123

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get_bot(self):
        return {"username": "example_bot"}

    def send_text(self, text):
        self.calls.append(("text", text))
        if text == "fail":
            raise NotifierError("bot was blocked by the user")
        return {"message_id": 42, "chat": {"id": 123}}

    def send_image(self, path, caption):
        self.calls.append(("image", path, caption))
        return {"message_id": 43, "chat": {"id": 123}}

    def send_document(self, path, caption):
        self.calls.append(("document", path, caption))
        return {"message_id": 44, "chat": {"id": 123}}


def test_tools_have_correct_schemas_and_side_effect_annotations():
    server = create_server(FakeNotifier)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == {"status", "send_text", "send_image", "send_document"}
    assert tools["status"].annotations.readOnlyHint
    for name in ("send_text", "send_image", "send_document"):
        assert tools[name].annotations.readOnlyHint is False
        assert tools[name].annotations.idempotentHint is False
        assert "chat_id" not in tools[name].inputSchema["properties"]
    assert tools["send_document"].inputSchema["required"] == ["path"]


def test_tools_deliver_arguments_and_summarize_results(tmp_path):
    notifier = FakeNotifier()
    server = create_server(lambda: notifier)

    async def exercise():
        status_content = await server.call_tool("status", {})
        status = json.loads(status_content[0].text)
        assert status == {"bot_username": "example_bot", "recipient_username": "colnio", "chat_id": 123, "linked": True}
        for name, args, message_id in (
            ("send_text", {"text": "hello"}, 42),
            ("send_image", {"path": str(tmp_path / "plot.png"), "caption": "plot"}, 43),
            ("send_document", {"path": str(tmp_path / "data.csv")}, 44),
        ):
            content = await server.call_tool(name, args)
            result = json.loads(content[0].text)
            assert result == {"message_id": message_id, "chat_id": 123, "recipient_username": "colnio"}

    asyncio.run(exercise())
    assert notifier.calls == [("text", "hello"), ("image", tmp_path / "plot.png", "plot"),
                              ("document", tmp_path / "data.csv", "")]


def test_errors_and_relative_paths_are_rejected():
    notifier = FakeNotifier()
    server = create_server(lambda: notifier)

    async def exercise():
        with pytest.raises(ToolError, match="absolute local file path"):
            await server.call_tool("send_document", {"path": "data.csv"})
        assert not notifier.calls
        with pytest.raises(ToolError, match="blocked"):
            await server.call_tool("send_text", {"text": "fail"})

    asyncio.run(exercise())
