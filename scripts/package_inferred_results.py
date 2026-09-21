"""打包最新六图、渲染PDF预览并核验链接与文件哈希。"""
import json
import py_compile
import re
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import sha256, json_dump


def main():
    latest=ROOT/'latest_results';evidence=ROOT/'evidence/inferred_20260921'
    figures=ROOT/'figures/inferred_20260921_seed0'
    visual=evidence/'visual';visual.mkdir(exist_ok=True)
    combined=fitz.open();previews=[];fonts={}
    for n in range(5,11):
        with fitz.open(figures/f'fig{n:02d}.pdf') as document:
            combined.insert_pdf(document)
            fonts[str(n)]=sorted({font[3] for font in document[0].get_fonts()})
            pix=document[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6))
            p=visual/f'fig{n:02d}_pdf.png';pix.save(p)
            previews.append(p)
    combined.set_toc([[1,f'Figure {n}',n-4] for n in range(5,11)])
    combined.save(latest/'figures_05_10.pdf');combined.close()
    # 接触表仅用于检查六个实际PDF的排版；不参与任何结果计算。
    sheet=Image.new('RGB',(1560,1830),'white');draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(previews):
        with Image.open(p) as im:
            im.thumbnail((740,550))
            x=(i%2)*780+(780-im.width)//2;y=(i//2)*610+35
            sheet.paste(im,(x,y));draw.text(((i%2)*780+20,(i//2)*610+10),f'Figure {i+5}',fill='black')
    sheet.save(visual/'pdf_contact.png')
    docs=[latest/'index.html',latest/'reproduction_report.md',figures/'index.html',ROOT/'LATEST.md']
    broken=[]
    for p in docs:
        content=p.read_text(encoding='utf-8')
        links=re.findall(r'(?:href|src)="([^"]+)"',content) if p.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',content)
        for link in links:
            if not link.startswith(('https:','http:','#')) and not (p.parent/link.split('#')[0]).exists():
                broken.append(dict(source=str(p.relative_to(ROOT)),target=link))
    assert not broken,broken
    sources=[ROOT/'scripts'/name for name in ['run_inferred_training.py','plot_paper_style.py','deliver_inferred_results.py','package_inferred_results.py']]
    for p in sources:
        py_compile.compile(str(p),doraise=True)
    encoding_files=sources+docs+[ROOT/'configs/inferred_20260921.yaml',evidence/'registration.md',ROOT/'reproduction_state.md']
    for p in encoding_files:
        text=p.read_text(encoding='utf-8')
        garbled_marker=''.join(map(chr,(0x951f,0x65a4,0x62f7)))
        assert '\ufffd' not in text and garbled_marker not in text,str(p)
    manifest=json.loads((evidence/'artifact_sha256.json').read_text(encoding='utf-8'))
    mismatches=[p for p,h in manifest.items() if sha256(ROOT/p)!=h]
    assert not mismatches,mismatches
    # 增补PDF合辑与本轮文档；不重写旧实验清单。
    for p in [latest/'figures_05_10.pdf',ROOT/'LATEST.md',ROOT/'reproduction_state.md',Path(__file__)]+list(visual.glob('*.png')):
        manifest[str(p.relative_to(ROOT))]=sha256(p)
    json_dump(evidence/'artifact_sha256.json',manifest)
    json_dump(evidence/'package_audit.json',dict(checked_artifact_hashes=len(manifest),hash_mismatches=[],broken_links=[],
        combined_pdf_pages=6,fonts_by_figure=fonts,utf8_passed=True,python_compile_passed=True,
        pdf_visual_contact=str((visual/'pdf_contact.png').relative_to(ROOT))))
    print(json.dumps(dict(checked_artifacts=len(manifest),broken_links=broken,combined_pdf_pages=6),indent=2))


if __name__=='__main__':
    main()
