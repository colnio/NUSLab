from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def make_icon(path: Path, label: str, fg, bg):
    size = 256
    img = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(img)

    pad = 20
    draw.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=42,
        outline=(255, 255, 255, 180),
        width=6,
    )

    try:
        font = ImageFont.truetype("arial.ttf", 110)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((size - tw) / 2, (size - th) / 2 - 8), label, font=font, fill=fg)

    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(str(path), format="ICO", sizes=sizes)


def main():
    out = Path(__file__).resolve().parent / "icons"
    out.mkdir(parents=True, exist_ok=True)

    palette = {
        "VAC": ((255, 255, 255, 255), (22, 101, 52, 255)),
        "AVC": ((255, 255, 255, 255), (30, 64, 175, 255)),
        "PUL": ((255, 255, 255, 255), (107, 33, 168, 255)),
        "FET": ((255, 255, 255, 255), (180, 83, 9, 255)),
        "MAP": ((255, 255, 255, 255), (190, 18, 60, 255)),
        "4PR": ((255, 255, 255, 255), (13, 148, 136, 255)),
        "4PF": ((255, 255, 255, 255), (71, 85, 105, 255)),
    }

    for label, (fg, bg) in palette.items():
        make_icon(out / f"{label}.ico", label, fg=fg, bg=bg)

    # Default Keithley icon.
    make_icon(out / "Keithley.ico", "K", fg=(255, 255, 255, 255), bg=(15, 23, 42, 255))
    print(f"Icons generated in {out}")


if __name__ == "__main__":
    main()
