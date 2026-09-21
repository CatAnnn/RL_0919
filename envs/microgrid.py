import numpy as np
import pandas as pd

from common import ROOT


def settle_trade(deficit, prices, grid_price):
    """按卖价与编号分配盈余，电网吸收剩余净电量。"""
    deficit = np.asarray(deficit, dtype=float)
    need = np.maximum(deficit, 0).copy()
    surplus = np.maximum(-deficit, 0).copy()
    flows = np.zeros((3, 3))
    for seller in sorted(range(3), key=lambda j: (prices[j], j)):
        for buyer in range(3):
            volume = min(surplus[seller], need[buyer])
            flows[seller, buyer] = volume
            surplus[seller] -= volume
            need[buyer] -= volume
    imports = flows.sum(axis=0)
    exports = flows.sum(axis=1)
    grid = need - surplus
    expenses = (flows * np.asarray(prices)[:, None]).sum(axis=0)
    expenses -= exports * prices
    expenses += need * grid_price
    residual = -deficit + imports - exports + grid
    return imports, exports, grid, expenses, residual, flows


class MicrogridEnv:
    """三份独立局部物理状态，同步一步仅用于结算交易。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.table = pd.read_csv(ROOT / 'data/day_ahead_profiles.csv')
        params = pd.read_csv(ROOT / 'data/mg_parameters.csv')
        self.cg = params[params.device == 'CG'].set_index('mg')
        self.ba = params[params.device == 'BA'].set_index('mg')
        self.cg_max = self.cg.p_max_kw.to_numpy(float)
        self.cg_coef = self.cg[['a', 'b', 'c']].to_numpy(float)
        self.ba_coef = self.ba[['a', 'b', 'c']].to_numpy(float)
        self.scenario = cfg['scenario']

    def reset(self, seed=0):
        self.t = 0
        self.soc = np.full(3, self.cfg['initial_soc'], dtype=float)
        self.load = self.table[['load_mg1_kw', 'load_mg2_kw', 'load_mg3_kw']].to_numpy(float).copy()
        self.wind = np.repeat(self.table.wind_kw.to_numpy(float)[:, None], 3, axis=1)
        self.pv = np.repeat(self.table.pv_kw.to_numpy(float)[:, None], 3, axis=1)
        if self.scenario == 'gaussian_forecast_errors':
            rng = np.random.default_rng(seed)
            for arr, sd in [(self.wind, .15), (self.pv, .15), (self.load, .03)]:
                arr[:] = np.maximum(arr * (1 + rng.normal(0, sd, arr.shape)), 0)
        elif self.scenario != 'deterministic':
            raise ValueError(self.scenario)
        return self.observation()

    def observation(self):
        k = (self.t - 1) % 24
        return np.column_stack([self.load[k], self.wind[k], self.pv[k], self.soc,
                                np.full(3, self.table.price_grid.iloc[k])])

    @staticmethod
    def normalize(obs):
        return np.asarray(obs, dtype=np.float32) / np.array([600, 100, 100, 1, 30], dtype=np.float32)

    def step(self, latent):
        if self.t >= 24:
            raise RuntimeError('终止后必须先reset')
        cfg = self.cfg
        obs = self.observation()
        latent = np.asarray(latent, dtype=float)
        mapped = np.tanh(latent)
        cg = np.clip((mapped[:, 0] + 1) * self.cg_max / 2, 0, self.cg_max)
        ba_raw = np.clip(mapped[:, 1] * 50, -50, 50)
        before = self.soc.copy()
        decay = (1 - cfg['delta']) * before
        capacity = cfg['battery_capacity_kwh']
        dt = cfg['dt_hours']
        # 自放电后若低于下界，最高允许功率也是充电，必须采用充电效率反解。
        inverse = lambda delta: np.where(delta >= 0, delta * cfg['eta_discharge'], delta / cfg['eta_charge']) * capacity / dt
        low = inverse(decay - cfg['soc_max'])
        high = inverse(decay - cfg['soc_min'])
        ba = np.clip(ba_raw, np.maximum(-50, low), np.minimum(50, high))
        self.soc = decay - np.where(ba >= 0, ba / cfg['eta_discharge'], ba * cfg['eta_charge']) * dt / capacity
        generation = cg + self.wind[self.t] + self.pv[self.t] + ba
        loss = cfg['loss_coefficient'] * generation
        deficit = self.load[self.t] - (generation - loss)
        z = ba + 150 * (1 - before)
        cg_cost = self.cg_coef[:, 0] * cg ** 2 + self.cg_coef[:, 1] * cg + self.cg_coef[:, 2]
        ba_cost = self.ba_coef[:, 0] * z ** 2 + self.ba_coef[:, 1] * z + self.ba_coef[:, 2]
        price = float(self.table.price_grid.iloc[self.t])
        imbalance_cost = price * np.abs(deficit)
        reward = -cfg['w_cost'] * (cg_cost + ba_cost) - cfg['w_deviation'] * imbalance_cost
        prices = np.full(3, self.table.price_mg.iloc[self.t])
        imports, exports, grid, expenses, residual, flows = settle_trade(deficit, prices, price)
        rows = []
        for i in range(3):
            row = dict(mg=i + 1, hour=self.t + 1, time_index=self.t,
                       load=self.load[self.t, i], wind=self.wind[self.t, i], pv=self.pv[self.t, i],
                       price_grid=price, price_mg=prices[i], latent_cg=latent[i, 0], latent_ba=latent[i, 1],
                       cg_raw=cg[i], ba_raw=ba_raw[i], p_cg=cg[i], p_ba=ba[i],
                       soc_before=before[i], soc_after=self.soc[i], p_loss=loss[i],
                       p_de=deficit[i], unbalanced=-deficit[i], cg_cost=cg_cost[i], ba_cost=ba_cost[i],
                       imbalance_cost=imbalance_cost[i], reward=reward[i], mg_import=imports[i],
                       mg_export=exports[i], grid_power=grid[i], transaction_expense=expenses[i],
                       balance_residual=residual[i])
            for j, key in enumerate(['load', 'wind', 'pv', 'soc', 'price']):
                row['obs_' + key] = obs[i, j]
            for j in range(3):
                row[f'sold_to_mg{j + 1}'] = flows[i, j]
            rows.append(row)
        self.t += 1
        return self.observation(), reward, self.t == 24, rows
