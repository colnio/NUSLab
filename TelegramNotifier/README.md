# Telegram notifier

A synchronous Python helper and CLI for sending text, images, and documents to
`@colnio`. No server or background process is needed. The configured bot is
[@notifier_colio_bot](https://t.me/notifier_colio_bot).

## Setup

Run commands from the repository root. The lab `.venv` already includes `httpx`;
for a new environment, install `TelegramNotifier/requirements.txt`.

The supplied token is stored locally in `TelegramNotifier/config.local.json`,
which this package's `.gitignore` excludes. On another machine, copy
`config.example.json` to `config.local.json` and fill in the token, or set
`TELEGRAM_BOT_TOKEN`. Environment variables `TELEGRAM_CHAT_ID` and
`TELEGRAM_USERNAME` also override the file. Keep tokens out of version control
and avoid HTTP client debug logging, which can expose Telegram request URLs.

1. Open https://t.me/notifier_colio_bot as `@colnio` and send `/start`.
2. Link that private chat and check the configuration:

```powershell
.\.venv\Scripts\python.exe -m TelegramNotifier link --wait 30
.\.venv\Scripts\python.exe -m TelegramNotifier check
```

Telegram requires a numeric chat ID for a private user; `@colnio` alone is not
a send destination. Linking matches the sender's username in a private message
and stores their numeric ID for subsequent sends. Once linked, changing the
user's username does not redirect notifications to another account.

Setup reads up to 100 pending updates without acknowledging them. Updates expire
after 24 hours, so send a fresh `/start` if necessary. If this bot already has a
webhook or another polling application, obtain the user's private chat ID from
that application, set `TELEGRAM_CHAT_ID`, and run `link` to verify and save it.
The notifier does not remove webhooks or discard updates. Sending needs no
polling once the recipient is linked.

## Send from PowerShell

```powershell
.\.venv\Scripts\python.exe -m TelegramNotifier text "Measurement complete."
.\.venv\Scripts\python.exe -m TelegramNotifier image "C:\path\plot.png" --caption "IV sweep"
.\.venv\Scripts\python.exe -m TelegramNotifier document "C:\path\results.csv" --caption "Raw data"
```

Use `--config PATH` before the subcommand for a different JSON config file.
The default config path is relative to the module, independent of the current
working directory. The Python package must still be on the import path.

## Send from Python or a notebook

```python
from TelegramNotifier import TelegramNotifier, NotifierError

with TelegramNotifier.from_config() as notifier:
    notifier.send_text("Measurement complete.")
    notifier.send_image("plot.png", caption="IV sweep")
    notifier.send_document("results.csv", caption="Raw data")
```

Each send returns Telegram's Message dictionary, including `message_id`.
The methods block until the request completes; call them after a measurement
or from a worker thread in a GUI. No instrument-control code is changed.

Text is plain text (no Markdown escaping needed), up to 4096 characters.
Captions allow 1024 characters. Photo uploads allow 10 MB; Telegram also limits
the sum of width and height to 10000 and the aspect ratio to 20. Documents allow
50 MB. Send images as documents to preserve the original file. Telegram validates
image format/dimensions and returns a readable error when rejected.

Failures raise `NotifierError` (local filesystem errors may raise `OSError`);
the CLI prints an error and exits with status 1. Rate-limit errors expose
`retry_after` in seconds. Requests have a 10-second connection timeout and a
60-second I/O timeout. There are no automatic retries: if a connection fails
after an upload, check the chat before retrying to avoid duplicate messages.

## Tests

For notifier-only tests, the existing lab environment is sufficient. To include
the MCP adapter tests, install `requirements-mcp.txt` plus `pytest` into a
separate environment (for example `TelegramNotifier/.venv`).

```powershell
.\.venv\Scripts\python.exe -m pytest TelegramNotifier/tests -q
```

Tests use a fake HTTP transport and never send Telegram messages.

## User-wide Codex MCP installation (Windows)

After linking the notifier, run from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TelegramNotifier\install_mcp.ps1
```

The installer copies the notifier and linked local configuration to
`%LOCALAPPDATA%\NUSLab\TelegramMCP`, creates a dedicated Python environment,
and registers the server as `telegram` in the user-wide Codex `config.toml`.
It backs up that config before registration. This makes the server available
across Codex projects for the current Windows user, independent of the repository
location. It does not install a Windows service or configure other users' accounts.
Codex starts and stops the stdio process as needed; no port is opened.

Restart the Codex extension/app after installation. The server provides:

| Tool | Action |
| --- | --- |
| `status` | Check bot identity and linked recipient without sending |
| `send_text` | Send plain text to the configured private recipient |
| `send_image` | Upload an image using an absolute local path |
| `send_document` | Upload a document using an absolute local path |

Example request: “Send this plot to me on Telegram.” The MCP tools use the
linked recipient and do not accept arbitrary destination IDs. Sending tools
are marked as writes and non-idempotent; they do not automatically retry.

The bot token stays in the installed `TelegramNotifier/config.local.json`,
outside Codex's configuration and command-line arguments. Re-running the
installer updates the code while preserving the installed credentials. To
change the installed bot or recipient, update that installed configuration.
The original Python helper continues to use the repository-local configuration.

Check registration with `codex mcp get telegram`. Remove registration with
`codex mcp remove telegram`. Other MCP clients can launch the same server using
`%LOCALAPPDATA%\NUSLab\TelegramMCP\.venv\Scripts\python.exe` with the argument
`%LOCALAPPDATA%\NUSLab\TelegramMCP\TelegramNotifier\run_mcp.py` (expand the paths
to absolute paths in client configuration).

Codex configuration reference:
[MCP servers](https://developers.openai.com/codex/mcp/).

API references: [messages](https://core.telegram.org/bots/api#sendmessage),
[photos](https://core.telegram.org/bots/api#sendphoto),
[documents](https://core.telegram.org/bots/api#senddocument),
[updates and chat linking](https://core.telegram.org/bots/api#getupdates).
