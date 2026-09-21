"""只读取已保存结果，按论文图5—10的版式输出最新图形。"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import ALGORITHMS, sha256
from plot_figures import display_curve

MG_COLORS=['#b64a59','#57538b','#167184']
PHASE_COLORS=['#fcf8dd','#d8fff2','#edd7fa']
ALGORITHM_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#ff3333']


def apply_style():
    """以原论文Times字体和盒式坐标轴覆盖通用绘图默认样式。"""
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':10,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.linewidth':.9,'axes.spines.top':True,'axes.spines.right':True,'axes.grid':True,
        'grid.color':'#aaaaaa','grid.alpha':.65,'grid.linewidth':.45,'xtick.direction':'in','ytick.direction':'in',
        'xtick.major.size':3.5,'ytick.major.size':3.5,'legend.fontsize':9,'legend.framealpha':.94,
        'legend.edgecolor':'#999999','legend.fancybox':False,'svg.fonttype':'none','pdf.fonttype':42,
        'savefig.dpi':400,'figure.facecolor':'white','axes.facecolor':'white'})


def panel(ax, letter):
    ax.text(.012,.965,f'({letter})',transform=ax.transAxes,ha='left',va='top',fontsize=11,zorder=20)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor',axis='y',linestyle=':',alpha=.35)


def plot_run(run_id):
    apply_style()
    logs=ROOT/'logs'/run_id
    out=ROOT/'figures'/run_id
    out.mkdir(parents=True,exist_ok=True)
    paths={a:logs/a/'history.csv' for a in ALGORITHMS}
    histories={a:pd.read_csv(p) for a,p in paths.items()}
    meta=json.loads((logs/'F-MADRL/metadata.json').read_text(encoding='utf-8'))
    sources=json.loads((logs/'evaluation/sources.json').read_text(encoding='utf-8'))
    schedule_path=logs/'evaluation/schedule.csv';metrics_path=logs/'evaluation/checkpoint_metrics.csv'
    schedule=pd.read_csv(schedule_path);metrics=pd.read_csv(metrics_path)
    window=meta['config']['smoothing_window']
    provenance={}
    annotation_rows=[]

    def save(fig,n,inputs,checkpoint):
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'fig{n:02d}.{ext}',bbox_inches='tight',pad_inches=.06)
        provenance[f'fig{n:02d}']=dict(run_id=run_id,seed=meta['seed'],config_hash=meta['config_hash'],
            inputs={str(p.relative_to(ROOT)):sha256(p) for p in inputs},data=meta['input_hashes'],checkpoint=checkpoint,
            source_script_sha256=sha256(Path(__file__)),style='paper_layout_times_boxed_axes_original_palette',
            numeric_status='actual_new_training_not_reference_digitized',
            display={'training_statistic':'undiscounted_episode_sum','rolling_window':window,
                     'smoothing_resets_at_federation_boundaries':True,'faint_curves':'raw_episode_sum',
                     'hour_axis':'time_index=hour-1','figure9_aggregation':'sum(abs(hourly deficit)); costs summed',
                     'figure9_xaxis':'equally_spaced_checkpoint_labels'},
            axes=[{'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())} for ax in fig.axes])
        plt.close(fig)

    f=histories['F-MADRL']
    shared=f[['run_id','seed','config_hash','iteration','mg','phase','episode_reward']].copy()
    shared['display_reward']=f.groupby('mg',sort=False).apply(lambda g:display_curve(g,window),include_groups=False).reset_index(level=0,drop=True).reindex(f.index)
    shared.to_csv(out/'fig05_fmadrl_source.csv',index=False)
    shared.to_csv(out/'fig10_fmadrl_source.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(7.16,4.9),sharex=True)
    fig.subplots_adjust(left=.12,right=.99,bottom=.10,top=.91,hspace=0)
    positions=[[472,956,1328],[480,942,1437],[482,938,1406]]
    for mg,ax in enumerate(axes,1):
        frame=f[f.mg==mg]
        for k,color in enumerate(PHASE_COLORS):
            ax.axvspan(k*500,(k+1)*500,color=color,zorder=0)
        ax.plot(frame.iteration,frame.episode_reward,color=MG_COLORS[mg-1],alpha=.22,lw=.45)
        ax.plot(frame.iteration,display_curve(frame,window),color=MG_COLORS[mg-1],lw=1.1)
        for b in [500,1000]:
            ax.axvline(b,color='#888888',ls=':',lw=.8)
        panel(ax,chr(96+mg))
        ax.set_xlim(0,1500)
        # 下方留出独立标注区，避免文字框遮住真实训练曲线。
        low=float(frame.episode_reward.min());high=float(frame.episode_reward.max());span=max(high-low,1)
        ax.set_ylim(low-.55*span,high+.12*span)
        for k,it in enumerate(positions[mg-1]):
            value=float(frame.loc[frame.iteration==it,'episode_reward'].iloc[0])
            ax.scatter([it],[value],s=8,color=MG_COLORS[mg-1],zorder=6)
            # 标注保留原文迭代位置，但奖励来自本次日志，箭头指向原始点。
            ax.annotate(f'Iteration: {it}\nReward: {value:.2f}',xy=(it,value),xycoords='data',
                xytext=((k+.49)/3,.20),textcoords='axes fraction',ha='center',va='center',fontsize=7.3,
                bbox=dict(boxstyle='square,pad=.20',fc=['#f5d996','#64eeee','#be78ef'][k],ec='#777777',lw=.4,alpha=.92),
                arrowprops=dict(arrowstyle='-|>',color=['#996012','#174b95','#7919a2'][k],lw=.7),zorder=10)
            annotation_rows.append(dict(mg=mg,iteration=it,raw_episode_reward=value))
    for k,color in enumerate(PHASE_COLORS):
        axes[0].text((k+.5)/3,1.055,f'Phase {k+1}',transform=axes[0].transAxes,ha='center',va='bottom',fontsize=11,fontweight='bold')
    axes[0].legend([Line2D([0],[0],color=c,lw=1.4) for c in MG_COLORS],['MG1','MG2','MG3'],
                   loc='upper center',ncol=3,fontsize=8,borderpad=.2,handlelength=1.6,columnspacing=.8)
    fig.supylabel('Reward',x=.012,fontsize=12)
    axes[-1].set_xlabel('Iteration');axes[-1].set_xticks(np.arange(0,1501,200))
    save(fig,5,[paths['F-MADRL']],'online training; FedAvg after 500/1000/1500')
    pd.DataFrame(annotation_rows).to_csv(out/'fig05_annotations.csv',index=False)

    palettes=[('#c100e8','#8700ed','#1400ee'),('#00b9d7','#00e4b7','#00d646'),('#ec9a00','#ff6900','#e42c00')]
    for mg,(cg_color,ba_color,bar_color) in enumerate(palettes,1):
        g=schedule[schedule.mg==mg]
        fig,axes=plt.subplots(2,1,figsize=(7.16,5.0),sharex=True)
        fig.subplots_adjust(left=.14,right=.86,bottom=.11,top=.98,hspace=.13)
        right=axes[0].twinx();right.grid(False)
        l1=axes[0].plot(g.time_index,g.p_cg,'-o',color=cg_color,ms=3.6,lw=1.25,label=r'$P_{\mathrm{CG}}$')
        l2=right.plot(g.time_index,g.p_ba,'-s',color=ba_color,ms=3.6,lw=1.25,label=r'$P_{\mathrm{BA}}$')
        axes[0].set_ylabel(r'$P_{\mathrm{CG}}$ (kW)');right.set_ylabel(r'$P_{\mathrm{BA}}$ (kW)')
        panel(axes[0],'a');right.yaxis.set_major_locator(MaxNLocator(5))
        # 近乎恒定的出力不使用百分之一kW的自动缩放，避免视觉夸大变化。
        cg_low=float(g.p_cg.min());cg_high=float(g.p_cg.max())
        if cg_high-cg_low<20:
            center=(cg_low+cg_high)/2
            axes[0].set_ylim(center-10,center+10)
        for axis in [axes[0],right]:
            lo,hi=axis.get_ylim();span=hi-lo
            axis.set_ylim(lo,hi+.22*span)
        axes[0].legend(l1+l2,[x.get_label() for x in l1+l2],loc='upper center',ncol=2,
                       borderpad=.3,columnspacing=.8,handlelength=1.8)
        axes[1].bar(g.time_index,g.unbalanced,width=.78,color=bar_color,zorder=2)
        axes[1].plot(g.time_index,g.unbalanced,color=bar_color,lw=.85,zorder=3)
        axes[1].axhline(0,color='#555555',lw=.65)
        axes[1].set_ylabel('Unbalanced demand (kW)');axes[1].set_xlabel('Time (h)')
        panel(axes[1],'b')
        # 柱图保留零基线，纵轴覆盖全部真实余缺功率。
        lo=min(0,float(g.unbalanced.min()));hi=max(0,float(g.unbalanced.max()));span=max(hi-lo,1)
        axes[1].set_ylim(lo-.08*span,hi+.20*span)
        axes[1].set_xlim(-1.15,24.15);axes[1].set_xticks([0,5,10,15,20])
        save(fig,mg+5,[schedule_path],sources['checkpoints'][-1])

    fig,axes=plt.subplots(3,1,figsize=(7.16,5.05),sharex=True)
    fig.subplots_adjust(left=.13,right=.99,bottom=.11,top=.99,hspace=.20)
    colors=['#007c77','#ffd2a4','#a34700'];markers=['s','o','^']
    iterations=[1,50,500,700,900,1400]
    for index,(key,label) in enumerate([('abs_deficit','Unbalanced demand\n(kW)'),('cg_cost','Cost of CG ($)'),('ba_cost','Cost of BA ($)')]):
        for mg in [1,2,3]:
            g=metrics[(metrics.mg==mg)&metrics.iteration.isin(iterations)].set_index('iteration').loc[iterations]
            axes[index].plot(range(6),g[key],'-'+markers[mg-1],color=colors[mg-1],lw=1.25,ms=4,label=f'MG{mg}')
        panel(axes[index],chr(97+index));axes[index].set_ylabel(label)
        axes[index].margins(y=.16);axes[index].set_xlim(-.45,5.45)
        axes[index].tick_params(labelbottom=True)
    lo,hi=axes[0].get_ylim();axes[0].set_ylim(lo,hi+.15*(hi-lo))
    axes[0].legend(loc='upper right',ncol=3,borderpad=.25,handlelength=1.7,columnspacing=.7,fontsize=8.5)
    axes[-1].set_xticks(range(6),[str(x) for x in iterations]);axes[-1].set_xlabel('Iteration')
    save(fig,9,[metrics_path],sources['checkpoints'][:-1])

    fig,axes=plt.subplots(3,1,figsize=(7.16,5.35),sharex=True)
    fig.subplots_adjust(left=.11,right=.99,bottom=.10,top=.95,hspace=0)
    plot_records=[]
    for mg,ax in enumerate(axes,1):
        for a,color in zip(ALGORITHMS,ALGORITHM_COLORS):
            g=histories[a][histories[a].mg==mg]
            plotted=display_curve(g,window)
            ax.plot(g.iteration,g.episode_reward,color=color,lw=.4,alpha=.18)
            ax.plot(g.iteration,plotted,color=color,lw=.95,label=a)
            record=g[['iteration','mg','algorithm','episode_reward']].copy();record['display_reward']=plotted
            plot_records.append(record)
        panel(ax,chr(96+mg));ax.margins(y=.12);ax.set_xlim(0,1500)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v/10000:g}'))
    axes[0].text(0,1.035,r'$\times10^4$',transform=axes[0].transAxes,ha='left',va='bottom',fontsize=9)
    axes[0].legend(loc='lower right',fontsize=8,borderpad=.3,handlelength=1.6,labelspacing=.15)
    fig.supylabel('Reward',x=.015,fontsize=12)
    axes[-1].set_xlabel('Iterations');axes[-1].set_xticks(np.arange(0,1501,200))
    save(fig,10,list(paths.values()),'online training, all algorithms 1500 iterations')
    pd.concat(plot_records).to_csv(out/'fig10_all_algorithms_source.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    captions={5:'三个MG的训练奖励，背景区分三段500次训练；浅线为原始奖励，实线为阶段内25点均值。',
              6:'MG1调度及交易前失衡，横轴0—23对应数据hour=1—24。',
              7:'MG2调度及交易前失衡，电池正功率为放电。',
              8:'MG3调度及交易前失衡，电池正功率为放电。',
              9:'六个实际检查点的失衡与成本；检查点标签等间距，失衡为逐小时绝对功率求和。',
              10:'四算法训练奖励；F-MADRL与图5使用同一份原始日志和显示规则。'}
    cards=''.join(f'<article><h2>Fig. {n}</h2><a href="fig{n:02d}.png"><img src="fig{n:02d}.png" alt="Fig. {n}"></a><p>{captions[n]}</p><p><a href="fig{n:02d}.pdf">PDF</a> · <a href="fig{n:02d}.svg">SVG</a> · <a href="fig{n:02d}.png">PNG</a></p></article>' for n in range(5,11))
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>最新重训图5—10</title><style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#222}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:24px}article{border:1px solid #ddd;padding:18px}img{width:100%}a{color:#165b97}</style><h1>反推候选重训：图5—10</h1><p>运行：'+run_id+'；w_C=0.02，w_de=0.45。图片来自本轮真实训练；样式匹配不代表论文数值复现通过。</p><p><a href="../../latest_results/index.html">最新数据与报告入口</a></p><main>'+cards+'</main></html>',encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='inferred_20260921_seed0')
    plot_run(parser.parse_args().run_id)
