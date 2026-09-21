"""Absolute-path launcher; works regardless of the MCP client's working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from TelegramNotifier.mcp_server import main

if __name__ == "__main__":
    main()
