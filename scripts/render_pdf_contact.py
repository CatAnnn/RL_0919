from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    out = ROOT / 'evidence/visual'
    out.mkdir(exist_ok=True)
    canvas = Image.new('RGB', (1500, 1900), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(sorted((ROOT / 'figures/main_seed0').glob('fig*.pdf'))):
        with fitz.open(path) as document:
            pix = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        picture = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        picture.thumbnail((490, 900))
        x, y = i % 3 * 500, i // 3 * 950
        draw.text((x + 10, y + 10), path.stem, fill='black')
        canvas.paste(picture, (x, y + 35))
    canvas.save(out / 'pdf_contact.png')
    print(out / 'pdf_contact.png')
