import argparse
import sys

from .notifier import DEFAULT_CONFIG, NotifierError, TelegramNotifier


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Send Telegram notifications to the configured private chat.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Local JSON configuration path")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check bot credentials without sending a message")
    link = commands.add_parser("link", help="Link @colnio after they send /start to the bot")
    link.add_argument("--wait", type=int, default=0, help="Wait up to 50 seconds for a new message")
    text = commands.add_parser("text", help="Send plain text")
    text.add_argument("text")
    for name in ("image", "document"):
        upload = commands.add_parser(name, help=f"Upload a local {name}")
        upload.add_argument("path")
        upload.add_argument("--caption", default="")
    args = parser.parse_args(argv)
    try:
        with TelegramNotifier.from_config(args.config) as notifier:
            if args.command == "check":
                bot = notifier.get_bot()
                print(f"Bot: https://t.me/{bot['username']}")
                print(f"Recipient: @{notifier.username}; chat ID: {notifier.chat_id or 'not linked'}")
                return 0
            if args.command == "link":
                chat_id = notifier.link(wait=args.wait, config_path=args.config)
                print(f"Linked @{notifier.username} (chat ID {chat_id}).")
                return 0
            if args.command == "text":
                result = notifier.send_text(args.text)
            elif args.command == "image":
                result = notifier.send_image(args.path, args.caption)
            else:
                result = notifier.send_document(args.path, args.caption)
            print(f"Sent {args.command}; message ID {result['message_id']}.")
            return 0
    except (NotifierError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
