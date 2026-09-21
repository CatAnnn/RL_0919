"""独立逆问题诊断：不调用训练入口，不修改主实验。"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'reference_digitized/inverse'
OUT = ROOT / 'evidence/inverse_parameters'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def cumulative_energy(power, eta):
    """正功率放电，以kWh累计考虑自放电后的电池能量消耗。"""
    h = [0.0]
    for p in power:
        h.append(.998*h[-1] + (p/eta if p >= 0 else p*eta))
    return np.array(h)


def capacity_constraints(groups, sign, eta, bounds, robust):
    """以容量C和初始储能E0为变量构造整个24小时的线性约束。"""
    a, b = [], []
    decay = .998**np.arange(25)
    for _, g in groups:
        power = sign*g.p_ba.to_numpy()
        error = .15 if robust else 0.0
        hlow = cumulative_energy(power-error, eta)
        hhigh = cumulative_energy(power+error, eta)
        for q, low, high in zip(decay, hlow, hhigh):
            a.extend([[bounds[0], -q], [-bounds[1], q]])
            b.extend([-high, low])
    return np.array(a), np.array(b)


def weight_region(fig9, phases, tolerance):
    """只做同阶段条件匹配，使用价格上下界而不是错误的未加权失衡。"""
    rewards = {900: [15059.85,3524.99,6164.79], 1400:[15059.85,3415.94,3743.82]}
    a, b, constraints = [], [], []
    for iteration in phases:
        for mg in [1,2,3]:
            vals = fig9.loc[(iteration, mg)]
            cost = vals.cg_cost + vals.ba_cost
            deficit = vals.abs_deficit
            y = rewards[iteration][mg-1]
            lower = [cost-1000, 8.10*max(0,deficit-50)]
            upper = [cost+1000, 27.35*(deficit+50)]
            a.extend([lower, list(-np.array(upper))])
            b.extend([y*(1+tolerance), -y*(1-tolerance)])
            constraints.append(dict(iteration=iteration,mg=mg,reward_magnitude=y,
                                    total_cost=cost,abs_deficit=deficit,
                                    minimum_price_weighted_deficit=lower[1],maximum_price_weighted_deficit=upper[1]))
    a.extend([[-1,0],[1,0],[0,-1],[0,1]])
    b.extend([0,1,0,1])
    a, b = np.array(a,dtype=float), np.array(b)
    feasible = linprog([0,0], A_ub=a,b_ub=b,bounds=[(0,1)]*2,method='highs')
    result = dict(phases=phases,reward_protocol_tolerance=tolerance,feasible=bool(feasible.success),constraints=constraints)
    vertices=[]
    if feasible.success:
        for i,j in itertools.combinations(range(len(b)),2):
            if abs(np.linalg.det(a[[i,j]])) > 1e-9:
                point = np.linalg.solve(a[[i,j]],b[[i,j]])
                if np.all(a@point <= b+1e-7):
                    vertices.append(point)
        vertices = np.unique(np.round(vertices,10),axis=0)
        center=vertices.mean(axis=0)
        vertices=vertices[np.argsort(np.arctan2(*(vertices-center)[:,::-1].T))]
        result.update(vertices=vertices.tolist(),w_cost_range=[float(vertices[:,0].min()),float(vertices[:,0].max())],
                      w_deviation_range=[float(vertices[:,1].min()),float(vertices[:,1].max())],
                      equal_half_feasible=bool(np.all(a@np.array([.5,.5]) <= b+1e-7)),
                      centroid_candidate_not_author_parameters=center.tolist())
    else:
        result['vertices']=[]
    return result


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    schedule=pd.read_csv(REF/'schedule_reference.csv')
    raw9=pd.read_csv(REF/'fig09_reference.csv')
    fig9=raw9.pivot(index=['iteration','mg'],columns='metric',values='value')
    data=pd.read_csv(ROOT/'data/day_ahead_profiles.csv')
    params=pd.read_csv(ROOT/'data/mg_parameters.csv').set_index(['mg','device'])
    physics, schedule_stats, hourly, kkt = [],[],[],[]
    for mg,g in schedule.groupby('mg'):
        cg,ba,u=g.p_cg.to_numpy(),g.p_ba.to_numpy(),g.unbalanced.to_numpy()
        gen=cg+data.wind_kw.to_numpy()+data.pv_kw.to_numpy()
        load=data[f'load_mg{mg}_kw'].to_numpy()
        for sign,loss in itertools.product([1,-1],[.02,.01,0]):
            residual=u-((1-loss)*(gen+sign*ba)-load)
            physics.append(dict(mg=mg,battery_plot_to_discharge_sign=sign,loss=loss,
                                rmse_kw=float(np.sqrt(np.mean(residual**2))),maximum_error_kw=float(abs(residual).max()),
                                within_reading_error_hours=int((abs(residual)<=1+(1-loss)*.65).sum())))
        # 不强制论文系数，单独估计图中数值等式的两个线性系数。
        coefficients=np.linalg.lstsq(np.column_stack([gen,ba]),u+load,rcond=None)[0]
        residual=u+load-np.column_stack([gen,ba])@coefficients
        c=params.loc[(mg,'CG')]
        costs=c.a*cg**2+c.b*cg+c.c
        stats=dict(mg=mg,cg_daily_cost=float(costs.sum()),abs_unbalanced=float(abs(u).sum()),
                   price_weighted_unbalanced=float((abs(u)*data.price_grid).sum()),
                   fitted_generation_coefficient=float(coefficients[0]),fitted_battery_coefficient=float(coefficients[1]),
                   fitted_balance_rmse=float(np.sqrt(np.mean(residual**2))),
                   fig09_1400_cg_cost=float(fig9.loc[(1400,mg)].cg_cost),fig09_1400_abs_deficit=float(fig9.loc[(1400,mg)].abs_deficit))
        schedule_stats.append(stats)
        for t in range(24):
            hourly.append(dict(mg=mg,hour=t+1,cg=cg[t],ba_plot=ba[t],u_plot=u[t],
                               u_paper=.98*(gen[t]+ba[t])-load[t],u_alternative=.99*(gen[t]-ba[t])-load[t]))
            if mg==1:
                kkt.append(dict(hour=t+1,ratio_wde_over_wc=(2*c.a*cg[t]+c.b)/(.98*data.price_grid.iloc[t]),
                                cg_reading_ratio_error=2*c.a*.5/(.98*data.price_grid.iloc[t])))
    pd.DataFrame(physics).to_csv(OUT/'balance_hypotheses.csv',index=False)
    pd.DataFrame(schedule_stats).to_csv(OUT/'schedule_consistency.csv',index=False)
    pd.DataFrame(hourly).to_csv(OUT/'balance_hourly.csv',index=False)
    pd.DataFrame(kkt).to_csv(OUT/'mg1_conditional_stationarity.csv',index=False)
    capacities, soc_ranges=[] ,[]
    groups=list(schedule.groupby('mg'))
    for sign,eta,bounds,robust in itertools.product([1,-1],[1.,.95],[(0.,1.),(.1,.9)],[False,True]):
        a,b=capacity_constraints(groups,sign,eta,bounds,robust)
        for s0 in [None,.3,.5,.7]:
            kwargs={} if s0 is None else dict(A_eq=[[-s0,1]],b_eq=[0])
            fit=linprog([1,0],A_ub=a,b_ub=b,bounds=[(1e-6,None),(0,None)],method='highs',**kwargs)
            capacities.append(dict(battery_plot_to_discharge_sign=sign,eta=eta,soc_min=bounds[0],soc_max=bounds[1],
                                   robust_to_reading_error=robust,fixed_initial_soc=s0,
                                   feasible=bool(fit.success),minimum_capacity_kwh=float(fit.x[0]) if fit.success else None,
                                   initial_soc_at_minimum=float(fit.x[1]/fit.x[0]) if fit.success else None))
        for capacity in [50,100,200,500]:
            lower=linprog([0,1],A_ub=a,b_ub=b,bounds=[(capacity,capacity),(0,None)],method='highs')
            upper=linprog([0,-1],A_ub=a,b_ub=b,bounds=[(capacity,capacity),(0,None)],method='highs')
            soc_ranges.append(dict(battery_plot_to_discharge_sign=sign,eta=eta,soc_min=bounds[0],soc_max=bounds[1],
                                   robust_to_reading_error=robust,capacity_kwh=capacity,feasible=bool(lower.success and upper.success),
                                   initial_soc_low=float(lower.x[1]/capacity) if lower.success else None,
                                   initial_soc_high=float(upper.x[1]/capacity) if upper.success else None))
    pd.DataFrame(capacities).to_csv(OUT/'capacity_lower_bounds.csv',index=False)
    pd.DataFrame(soc_ranges).to_csv(OUT/'initial_soc_intervals.csv',index=False)
    regions=[weight_region(fig9,phases,tol) for phases in [[900],[1400],[900,1400]] for tol in [0,.05,.20]]
    save_json(OUT/'reward_weight_regions.json',regions)
    # 两个MG的近似相同电池曲线若使用同一容量/初态，成本差不能任意大。
    b2=schedule[schedule.mg==2].p_ba.to_numpy()
    b3=schedule[schedule.mg==3].p_ba.to_numpy()
    battery_cost_bounds=[]
    for sign in [1,-1]:
        # 只使用首步能量及SOC属于[0,1]的必要容量下界，避免以保守充分下界代替必要下界。
        first_min=min(b2[0],b3[0])-.15
        capacity_min=first_min/.95 if sign==1 else .95*first_min
        difference_bound=np.maximum(
            abs(cumulative_energy(sign*b2-.15,.95)-cumulative_energy(sign*b3+.15,.95)),
            abs(cumulative_energy(sign*b2+.15,.95)-cumulative_energy(sign*b3-.15,.95)))[:-1]
        # z范围[-50,200]；系数差项≤66；MG3成本关于z的导数绝对值≤12.66。
        cost_bound=24*66+12.66*np.sum(abs(b2-b3)+.3+150*difference_bound/capacity_min)
        observed_difference=abs(fig9.loc[(1400,3)].ba_cost-fig9.loc[(1400,2)].ba_cost)
        battery_cost_bounds.append(dict(sign=sign,eta=.95,capacity_necessary_lower_bound=capacity_min,
              maximum_cost_difference_with_common_capacity_and_initial_soc=float(cost_bound),
              fig09_1400_cost_difference=float(observed_difference),reading_error_difference=1000,
              common_parameters_same_schedule_hypothesis_excluded=bool(observed_difference-1000>cost_bound)))
    save_json(OUT/'battery_cost_consistency_bound.json',battery_cost_bounds)
    candidates=[]
    for wc,wd in [(0.01,.5),(.02,.45),(.03,.4),(.05,.3),(.5,.5)]:
        for z in regions:
            valid=True
            for c in z['constraints']:
                low=wc*(c['total_cost']-1000)+wd*c['minimum_price_weighted_deficit']
                high=wc*(c['total_cost']+1000)+wd*c['maximum_price_weighted_deficit']
                reward=c['reward_magnitude'];tol=z['reward_protocol_tolerance']
                valid=valid and low<=reward*(1+tol) and high>=reward*(1-tol)
            candidates.append(dict(w_cost=wc,w_deviation=wd,phases=str(z['phases']),
                                    reward_tolerance=z['reward_protocol_tolerance'],passes_necessary_bounds=bool(valid)))
    pd.DataFrame(candidates).to_csv(OUT/'reward_candidate_checks.csv',index=False)
    # 使用固定TD目标演示符号；这不是作者的Critic训练记录。
    y,v,lr=10.,0.,.001
    error0=(v-y)**2
    grad=2*(v-y)
    desc=(v-lr*grad-y)**2
    asc=(v+lr*grad-y)**2
    critic=dict(target=y,initial_value=v,learning_rate=lr,loss_before=error0,loss_after_descent=desc,
                loss_after_ascent=asc,gradient=grad,scope='单参数固定目标平方误差；不推断作者实现')
    assert desc<error0<asc
    # 将一个隐藏单元复制成两个，将下游权重各减半，构造不同宽度的等价Tanh网络。
    rng=np.random.default_rng(20260921)
    x=rng.normal(size=(100,5)); w=rng.normal(size=(5,8)); bias=rng.normal(size=8); v=rng.normal(size=(8,2))
    output=np.tanh(x@w+bias)@v
    wider_w=np.column_stack([w,w[:,0]])
    wider_b=np.r_[bias,bias[0]]
    wider_v=np.vstack([v.copy(),v[0]/2]); wider_v[0]=v[0]/2
    gap=float(abs(output-np.tanh(x@wider_w+wider_b)@wider_v).max())
    assert gap<1e-12
    # 内循环为一次时，新旧策略相同，r=1处的clip半径不影响本次梯度。
    gradients=[]
    for clip in [.1,.2,.3]:
        ratio=torch.tensor([1.,1.,1.],requires_grad=True)
        adv=torch.tensor([-2.,.5,1.5])
        objective=torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        objective.backward()
        gradients.append(dict(clip=clip,gradient=ratio.grad.tolist()))
    assert all(g['gradient']==gradients[0]['gradient'] for g in gradients)
    eta=.95
    # 单位充电输入1kWh后放电，正确分支往返产出η²；对调分支变成1/η²。
    identities=dict(critic=critic,network=dict(original_hidden_width=8,equivalent_hidden_width=9,
                     sample_count=100,maximum_output_difference=gap,meaning='不同网络宽度可产生相同策略输出'),
                    ppo_clip_first_step_gradients=gradients,soc=dict(eta=eta,physical_roundtrip_ratio=eta**2,
                    wrong_reversed_branch_roundtrip_ratio=1/eta**2,unit_efficiency_branches_indistinguishable=True))
    save_json(OUT/'identifiability_checks.json',identities)
    # 诊断图单独命名，禁止覆盖六张训练图。
    fig,axs=plt.subplots(1,3,figsize=(15,4.4))
    for mg,g in pd.DataFrame(hourly).groupby('mg'):
        axs[0].plot(g.hour,g.u_plot-g.u_paper,label=f'MG{mg}: printed')
        axs[0].plot(g.hour,g.u_plot-g.u_alternative,'--',label=f'MG{mg}: alternative')
    axs[0].set(xlabel='Hour',ylabel='Balance residual (kW)',title='Figure 6-8 internal balance')
    axs[0].legend(fontsize=6,ncol=2)
    r=next(x for x in regions if x['phases']==[900,1400] and x['reward_protocol_tolerance']==.05)
    for phase,color in [([900],'tab:blue'),([1400],'tab:orange'),([900,1400],'tab:green')]:
        z=next(x for x in regions if x['phases']==phase and x['reward_protocol_tolerance']==.05)
        if z['feasible']:
            verts=np.array(z['vertices']); axs[1].fill(verts[:,0],verts[:,1],alpha=.3,color=color,label=str(phase))
    axs[1].set(xlabel='Cost weight',ylabel='Imbalance weight',title='Conditional weight sets (5% tolerance)')
    axs[1].legend(fontsize=7)
    soc=pd.DataFrame(soc_ranges)
    for sign,color in [(1,'tab:blue'),(-1,'tab:orange')]:
        capacities_dense=np.linspace(40,500,461)
        a,b=capacity_constraints(groups,sign,.95,(.1,.9),True)
        lows=[];highs=[]
        for cap in capacities_dense:
            # 从线性约束直接求初态区间，避免稀疏容量点线性连接扭曲可行域。
            lo=max((rhs-row[0]*cap)/row[1]/cap for row,rhs in zip(a,b) if row[1]<0)
            hi=min((rhs-row[0]*cap)/row[1]/cap for row,rhs in zip(a,b) if row[1]>0)
            lows.append(lo if lo<=hi else np.nan);highs.append(hi if lo<=hi else np.nan)
        axs[2].fill_between(capacities_dense,lows,highs,alpha=.3,color=color,label=f'plot sign {sign:+d}')
    axs[2].set(xlabel='Capacity (kWh)',ylabel='Initial SOC',title='Robust SOC intervals, eta=.95, [.1,.9]')
    axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'inverse_diagnostics.png',dpi=160); fig.savefig(OUT/'inverse_diagnostics.pdf'); plt.close(fig)
    files=[Path(__file__),ROOT/'scripts/digitize_inverse.py',REF/'protocol.md',REF/'schedule_reference.csv',REF/'fig09_reference.csv',ROOT/'data/mg_parameters.csv',ROOT/'data/day_ahead_profiles.csv',next(ROOT.glob('*.pdf'))]
    save_json(OUT/'analysis_manifest.json',dict(executable=sys.executable,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
              analysis_type='conditional_inverse_diagnostics_not_RL_training',training_executed=False,
              inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              checks=dict(critic_sign_checked=True,network_equivalence_checked=True,ppo_first_step_checked=True),
              reward_joint_5pct_feasible=r['feasible']))
    # 用逐步递推独立复核LP给出的边界容量；同时验证低于边界会违反至少一个约束。
    replay_checks=[]
    for entry in capacities:
        if not entry['feasible']:
            continue
        cap=entry['minimum_capacity_kwh'];s0=entry['initial_soc_at_minimum']
        extremum=[]
        for _,g in groups:
            for direction in [-1,1]:
                ss=s0
                for p in entry['battery_plot_to_discharge_sign']*g.p_ba.to_numpy()+direction*(.15 if entry['robust_to_reading_error'] else 0):
                    ss=.998*ss-(p/entry['eta'] if p>=0 else p*entry['eta'])/cap
                    extremum.append(ss)
        a,b=capacity_constraints(groups,entry['battery_plot_to_discharge_sign'],entry['eta'],
                                 (entry['soc_min'],entry['soc_max']),entry['robust_to_reading_error'])
        smaller=cap*.999
        s0fixed=entry['fixed_initial_soc']
        smaller_bounds=[(smaller,smaller),(0,None)]
        kwargs={} if s0fixed is None else dict(A_eq=[[-s0fixed,1]],b_eq=[0])
        reduced=linprog([0,0],A_ub=a,b_ub=b,bounds=smaller_bounds,method='highs',**kwargs)
        replay_checks.append(min(extremum)>=entry['soc_min']-1e-8 and max(extremum)<=entry['soc_max']+1e-8 and not reduced.success)
    assert all(replay_checks)
    previous=json.loads((ROOT/'evidence/artifact_sha256.json').read_text(encoding='utf-8'))
    protected={k:v for k,v in previous.items() if Path(k).parts[0] in ['checkpoints','logs','figures','data','agents','envs','federated','configs']
               or k in ['train.py','common.py','evaluate.py','plot_figures.py','assumptions.yaml']}
    differences=[k for k,v in protected.items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    assert not differences
    save_json(OUT/'analysis_audit.json',dict(capacity_bound_forward_replay_checks=len(replay_checks),all_capacity_checks_passed=all(replay_checks),
                  protected_existing_artifacts_checked=len(protected),protected_artifact_mismatches=differences,
                  reference_rows=len(schedule),figure9_metric_rows=len(raw9),scientific_parameter_recovery_unique=False))
    print(pd.DataFrame(schedule_stats).round(3).to_string(index=False))
    print('权重集合:',[(x['phases'],x['reward_protocol_tolerance'],x['feasible'],x.get('w_cost_range'),x.get('w_deviation_range')) for x in regions])
    print(pd.DataFrame(capacities).query('eta==.95 and soc_min==.1 and robust_to_reading_error and fixed_initial_soc==.5').to_string(index=False))
    print(soc.query('eta==.95 and soc_min==.1 and robust_to_reading_error and capacity_kwh==100').to_string(index=False))


if __name__=='__main__':
    main()
