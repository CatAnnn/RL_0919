"""读取论文原图中的曲线中心，仅输出反推参考数据。"""
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference_digitized/inverse'


def read_curve(arr, x0, dx, guesses, color, y0, value0, y1, value1):
    """利用人工定位的小窗口读取有色曲线，避免图例干扰。"""
    result = []
    for i, guess in enumerate(guesses):
        x = round(x0 + i * dx)
        low, high = max(0, round(guess) - 12), min(arr.shape[0], round(guess) + 13)
        crop = arr[low:high, x-2:x+3, :3].astype(float)
        distance = np.linalg.norm(crop - np.array(color), axis=2)
        ys, _ = np.where(distance < 90)
        if len(ys) < 3:
            raise ValueError(f'曲线定位失败: point={i}, x={x}, y={guess}')
        y = float(np.median(ys) + low)
        result.append((y - y0) / (y1 - y0) * (value1 - value0) + value0)
    return np.array(result)


def main():
    OUT.mkdir(exist_ok=True)
    pdf = next(ROOT.glob('*.pdf'))
    doc = fitz.open(pdf)
    sources = {5: 126, 6: 124, 7: 125, 8: 131, 9: 132}
    for fig, xref in sources.items():
        fitz.Pixmap(doc, xref).save(str(OUT / f'figure{fig}_source.png'))
    # 校准坐标均对应嵌入图片的原始像素；纵坐标人工初值只用于定位。
    specs = {
        1: dict(fig=6, x0=234, dx=(1401-234)/20,
                cg=[385,177,216,122,416,389,266,395,335,173,235,235,219,219,281,523,285,91,347,250,202,39,320,382],
                ba=[39,523,216,413,323,519,314,283,493,207,353,372,249,435,333,382,385,263,383,488,213,517,365,202],
                cgcolor=[190,0,225], bacolor=[135,0,245], cgaxis=[521,110,100,150], baaxis=[374,0,60,7.5],
                ux=257, udx=(1385-257)/20, uy0=607, uy1=1122, uv1=-400,
                u=[-292,-153,-98,-85,-100,-78,-115,-147,-159,-179,-121,-141,-153,-107,-132,-167,-122,-235,-388,-394,-389,-340,-272,-285], ucolor=[0,0,230]),
        2: dict(fig=7, x0=220, dx=(1423-220)/20,
                cg=[319,538,271,433,539,246,100,156,131,99,330,228,41,188,211,313,212,210,144,75,110,140,163,134],
                ba=[41,533,533,533,533,533,533,533,533,539,525,533,533,535,531,533,533,533,533,533,533,533,533,533],
                cgcolor=[0,185,214], bacolor=[0,235,174], cgaxis=[557,60,67,140], baaxis=[535,0,41,25],
                ux=244, udx=(1406-244)/20, uy0=958, uy1=721, uv1=40,
                u=[13,-9,36,9,-24,19,31,-9,-10,11,-15,16,33,28,15,-32,-12,-18,-10,-3,-5,1,12,52], ucolor=[0,215,62]),
        3: dict(fig=8, x0=220, dx=(1423-220)/20,
                cg=[399,319,359,434,261,232,399,292,228,238,177,371,387,483,539,376,359,308,41,410,324,363,98,318],
                ba=[40]+[539]*23, cgcolor=[225,148,0], bacolor=[255,110,0],
                cgaxis=[550,110,68,160], baaxis=[539,0,40,25],
                ux=244, udx=(1406-244)/20, uy0=925, uy1=690, uv1=40,
                u=[25,47,45,36,38,36,6,-18,-18,-4,14,9,-6,3,-12,-26,-21,-21,4,-38,-26,-18,31,42], ucolor=[224,42,0]),
    }
    rows = []
    for mg, s in specs.items():
        arr = np.asarray(Image.open(OUT / f"figure{s['fig']}_source.png"))
        cg = read_curve(arr, s['x0'], s['dx'], s['cg'], s['cgcolor'], *s['cgaxis'])
        ba = read_curve(arr, s['x0'], s['dx'], s['ba'], s['bacolor'], *s['baaxis'])
        # 柱图中心与上方折线横轴具有不同边距，单独校准。
        us = []
        for i, guess in enumerate(s['u']):
            x = round(s['ux'] + i * s['udx'])
            expected = s['uy0'] + guess / s['uv1'] * (s['uy1'] - s['uy0'])
            low, high = max(0, round(expected)-15), min(arr.shape[0], round(expected)+16)
            crop = arr[low:high, x-3:x+4, :3].astype(float)
            mask = np.linalg.norm(crop-np.array(s['ucolor']), axis=2) < 90
            ys = np.where(mask.sum(axis=1) >= 4)[0] + low
            if not len(ys):
                raise ValueError(f'柱图定位失败 {mg}, {i}')
            edge = max(ys) if guess < 0 else min(ys)
            us.append((edge-s['uy0'])/(s['uy1']-s['uy0'])*s['uv1'])
        for i in range(24):
            rows.append(dict(figure=s['fig'], mg=mg, hour=i+1, plot_hour=i,
                             p_cg=cg[i], p_ba=ba[i], unbalanced=us[i],
                             cg_error_kw=.5, ba_error_kw=.15, unbalanced_error_kw=1.0))
    pd.DataFrame(rows).to_csv(OUT/'schedule_reference.csv', index=False)
    # 图9存在圆点被三角形遮挡的情况，颜色中位数会偏移，改用原图可见轮廓的中心。
    # 此处均为人工读图的像素中心，不是作者原始数值。
    xs = [323,636,949,1262,1575,1888]
    fig9 = []
    guesses = {
        'abs_deficit': [[53,62,129,217,219,222],[225,319,395,398,394,402],[365,373,391,393,399,409]],
        'cg_cost': [[827,819,759,669,673,674],[553,647,753,751,752,752],[703,709,732,728,737,757]],
        'ba_cost': [[1278,1287,1228,1081,997,997],[1208,1270,1337,1333,985,1337],[1227,1253,1330,1331,1329,1023]],
    }
    axes = {'abs_deficit':(15,6000,332,1500), 'cg_cost':(478,60000,887,10000), 'ba_cost':(985,25000,1305,5000)}
    for metric, gs in guesses.items():
        y0,v0,y1,v1 = axes[metric]
        for j in range(3):
            for k, x in enumerate(xs):
                value = (gs[j][k]-y0)/(y1-y0)*(v1-v0)+v0
                fig9.append(dict(mg=j+1,iteration=[1,50,500,700,900,1400][k],metric=metric,value=value,
                                 reading_error=50 if metric=='abs_deficit' else 500))
    pd.DataFrame(fig9).to_csv(OUT/'fig09_reference.csv',index=False)
    (OUT/'digitization.json').write_text(json.dumps(dict(
        paper_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), xrefs=sources,
        schedule_calibration=specs, fig9_x_pixels=xs, fig9_y_pixels=guesses, fig9_axes=axes,
        warning='图像读数，不是原始作者数据或训练日志；误差为保守读图容差。'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('数字化完成：72条小时数据、54条图9参考指标。')


if __name__ == '__main__':
    main()
