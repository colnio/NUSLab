from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def rounded_box(draw, xy, fill, outline=(190, 198, 210), radius=18, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped_text(draw, text, box, font, fill=(15, 23, 42), line_spacing=8):
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        tw = draw.textbbox((0, 0), test, font=font)[2]
        if tw <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = y0
    for ln in lines:
        draw.text((x0, y), ln, font=font, fill=fill)
        h = draw.textbbox((0, 0), ln, font=font)[3]
        y += h + line_spacing
        if y > y1:
            break


def paste_icon(canvas, icon_path: Path, xy, size=64):
    try:
        ico = Image.open(icon_path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        canvas.paste(ico, xy, ico)
    except Exception:
        pass


def main():
    base = Path(__file__).resolve().parent
    icons = base / "icons"
    out_path = base / "Keithley_QuickStart.png"

    w, h = 2200, 1400
    img = Image.new("RGB", (w, h), (243, 246, 251))
    draw = ImageDraw.Draw(img)

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(28)
    section_font = load_font(34, bold=True)
    card_title_font = load_font(30, bold=True)
    body_font = load_font(24)
    small_font = load_font(20)

    # Header
    rounded_box(draw, (30, 30, w - 30, 190), fill=(15, 23, 42), outline=(15, 23, 42), radius=24, width=1)
    draw.text((60, 52), "Keithley Measurement Quick Start", font=title_font, fill=(255, 255, 255))
    draw.text(
        (62, 130),
        "Use this guide before every run. Refresh GPIB first, then verify wiring and limits.",
        font=subtitle_font,
        fill=(203, 213, 225),
    )

    # Left checklist
    left_x0, left_x1 = 30, 760
    rounded_box(draw, (left_x0, 220, left_x1, h - 30), fill=(255, 255, 255), radius=20)
    draw.text((left_x0 + 26, 248), "Universal Setup", font=section_font, fill=(15, 23, 42))
    checklist = [
        "1) Launch the target app EXE.",
        "2) Click Refresh GPIB and wait for \"Found N device(s)\".",
        "3) Select output folder: app auto-creates DATE and SAMPLE folders.",
        "4) Enter sample name and check instrument roles.",
        "5) Wire HI / LO to the correct contacts before Start.",
        "6) Start with Auto-range, NPLC = 1, safe compliance for your sample.",
        "7) Click Start. Watch progress bar, ETA, and live plots.",
        "8) Use Stop for controlled finish and data save.",
    ]
    y = 308
    for line in checklist:
        draw_wrapped_text(
            draw,
            line,
            (left_x0 + 30, y, left_x1 - 26, y + 90),
            body_font,
            fill=(30, 41, 59),
            line_spacing=4,
        )
        y += 82

    warn_box = (left_x0 + 24, h - 250, left_x1 - 24, h - 58)
    rounded_box(draw, warn_box, fill=(255, 247, 237), outline=(251, 146, 60), radius=16)
    draw.text((warn_box[0] + 18, warn_box[1] + 18), "Safety Notes", font=card_title_font, fill=(154, 52, 18))
    warn_text = (
        "- Compliance protects samples. Keep it conservative.\n"
        "- Verify contact map before high-voltage sweeps.\n"
        "- If readings look unstable, stop and re-check wiring."
    )
    draw.multiline_text((warn_box[0] + 20, warn_box[1] + 66), warn_text, font=small_font, fill=(124, 45, 18), spacing=8)

    # Right cards
    cards = [
        ("VAC", "IV curves", "1 SourceMeter\nConnect HI/LO to DUT pair.\nDefault: Auto-range, NPLC 1."),
        ("AVC", "V(I) curves", "1 SourceMeter (current source mode)\nSweep current and read voltage."),
        ("PUL", "Set/Reset pulses", "1 SourceMeter\nFor endurance-style pulse testing."),
        ("FET", "Id-Vg transfer", "2 instruments\nGate source + SD source/measure."),
        ("MAP", "Id(Vsd, Vg) map", "2 instruments\nNested SD sweep per gate step."),
        ("4PR", "4-probe resistance", "2 instruments\nCurrent source + voltmeter."),
        ("4PF", "4-probe FET", "3 instruments\nSD source, probe meter, gate source."),
    ]

    right_x0, right_x1 = 790, w - 30
    rounded_box(draw, (right_x0, 220, right_x1, h - 30), fill=(255, 255, 255), radius=20)
    draw.text((right_x0 + 26, 248), "Apps and What They Do", font=section_font, fill=(15, 23, 42))

    cols = 2
    card_w = (right_x1 - right_x0 - 26 * 3) // cols
    card_h = 220
    x_start = right_x0 + 26
    y_start = 310
    gap_x = 26
    gap_y = 20

    for i, (icon_key, title, desc) in enumerate(cards):
        r = i // cols
        c = i % cols
        x0 = x_start + c * (card_w + gap_x)
        y0 = y_start + r * (card_h + gap_y)
        x1 = x0 + card_w
        y1 = y0 + card_h
        rounded_box(draw, (x0, y0, x1, y1), fill=(248, 250, 252), outline=(203, 213, 225), radius=16)

        icon_path = icons / f"{icon_key}.ico"
        paste_icon(img, icon_path, (x0 + 18, y0 + 18), size=62)
        draw.text((x0 + 94, y0 + 24), icon_key, font=card_title_font, fill=(15, 23, 42))
        draw.text((x0 + 94, y0 + 66), title, font=body_font, fill=(51, 65, 85))
        draw.multiline_text((x0 + 18, y0 + 108), desc, font=small_font, fill=(71, 85, 105), spacing=6)

    footer = "Generated for Measurement PC | Keep this file next to the EXE launchers"
    tw = draw.textbbox((0, 0), footer, font=small_font)[2]
    draw.text(((w - tw) // 2, h - 22), footer, font=small_font, fill=(100, 116, 139))

    img.save(out_path, "PNG")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
