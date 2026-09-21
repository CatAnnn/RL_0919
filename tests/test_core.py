import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agents.on_policy import Agent, conjugate_gradient, gae
from common import ALGORITHMS, ROOT, load_config
from envs.microgrid import MicrogridEnv, settle_trade
from federated.average import fedavg
from train import collect_episode, train_one


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.env = MicrogridEnv(self.cfg)
        self.env.reset()

    def test_tables(self):
        table = self.env.table
        self.assertEqual(table.hour.tolist(), list(range(1, 25)))
        self.assertEqual(table.load_mg1_kw.iloc[0], 457.7)
        self.assertEqual(table.price_grid.iloc[13], 27.35)
        self.assertEqual(table.pv_kw.iloc[11], 42.68)
        self.assertEqual(table.load_mg3_kw.iloc[22], 161.39)
        np.testing.assert_equal(self.env.cg_max, [200, 280, 200])
        self.assertEqual(self.env.ba_coef[2, 0], .0173)

    def test_strict_unknowns_rejected(self):
        with self.assertRaisesRegex(ValueError, '缺失参数'):
            load_config('configs/strict.yaml')

    def test_observation_is_lagged(self):
        obs = self.env.observation()
        self.assertEqual(obs[0, 0], 447.3)
        nxt, _, _, _ = self.env.step(np.zeros((3, 2)))
        self.assertEqual(nxt[0, 0], 457.7)
        self.assertEqual(nxt.shape, (3, 5))

    def test_random_physical_invariants(self):
        rng = np.random.default_rng(42)
        previous = np.full(3, self.cfg['initial_soc'])
        for _ in range(24):
            _, reward, _, rows = self.env.step(rng.normal(0, 4, (3, 2)))
            for i, r in enumerate(rows):
                self.assertAlmostEqual(r['soc_before'], previous[i])
                self.assertTrue(-50 <= r['p_ba'] <= 50)
                self.assertTrue(0 <= r['p_cg'] <= self.env.cg_max[i])
                self.assertTrue(self.cfg['soc_min'] - 1e-12 <= r['soc_after'] <= self.cfg['soc_max'] + 1e-12)
                self.assertAlmostEqual(r['p_de'], -r['unbalanced'])
                self.assertAlmostEqual(r['p_loss'], .02 * (r['p_cg'] + r['wind'] + r['pv'] + r['p_ba']))
                self.assertAlmostEqual(reward[i], -.5 * (r['cg_cost'] + r['ba_cost'] + r['imbalance_cost']))
                self.assertAlmostEqual(r['balance_residual'], 0)
                expected = .998 * r['soc_before'] - (r['p_ba'] / .95 if r['p_ba'] >= 0 else r['p_ba'] * .95) / 100
                self.assertAlmostEqual(r['soc_after'], expected)
                previous[i] = r['soc_after']

    def test_charge_discharge_sign(self):
        self.env.step(np.array([[0, 1], [0, -1], [0, 0]]))
        self.assertLess(self.env.soc[0], .5 * .998)
        self.assertGreater(self.env.soc[1], .5 * .998)

    def test_self_discharge_at_lower_bound_requires_charge(self):
        self.env.soc[:] = self.cfg['soc_min']
        _, _, _, rows = self.env.step(np.ones((3, 2)) * 5)
        for row in rows:
            self.assertLess(row['p_ba'], 0)
            self.assertAlmostEqual(row['soc_after'], self.cfg['soc_min'])

    def test_seller_priority_and_limits(self):
        imp, exp, grid, expense, residual, flows = settle_trade(np.array([-30, -50, 60]), np.array([5, 2, 4]), 10)
        np.testing.assert_allclose(flows[:, 2], [10, 50, 0])
        np.testing.assert_allclose(residual, 0)
        np.testing.assert_allclose(grid, [-20, 0, 0])
        self.assertAlmostEqual(expense.sum(), 0)

    def test_gaussian_reproducible(self):
        cfg = dict(self.cfg, scenario='gaussian_forecast_errors')
        a, b = MicrogridEnv(cfg), MicrogridEnv(cfg)
        a.reset(3)
        b.reset(3)
        np.testing.assert_array_equal(a.load, b.load)
        self.assertTrue((a.pv >= 0).all())
        self.assertFalse(np.array_equal(a.wind[:, 0], a.wind[:, 1]))


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cfg = load_config()
        torch.manual_seed(7)

    def test_gae_terminal_mask(self):
        rewards = torch.tensor([1., 2., 3.])
        values = torch.tensor([.1, .2, .3])
        next_values = torch.tensor([.2, .3, 999.])
        done = torch.tensor([0., 0., 1.])
        result = gae(rewards, values, next_values, done, 1., 1.)
        torch.testing.assert_close(result, torch.tensor([5.9, 4.8, 2.7]))

    def test_old_policy_and_joint_log_prob(self):
        a = Agent(self.cfg)
        a.sync_old()
        obs, action = torch.ones(2, 5), torch.ones(2, 2)
        old = a.old_actor.log_prob(obs, action).clone()
        with torch.no_grad():
            a.actor.log_std.add_(.5)
        torch.testing.assert_close(old, a.old_actor.log_prob(obs, action))
        self.assertTrue(all(not p.requires_grad for p in a.old_actor.parameters()))
        torch.testing.assert_close(a.actor.log_prob(obs, action), a.actor.distribution(obs).log_prob(action).sum(-1))

    def test_fedavg_exact_and_optimizer_retention(self):
        agents = [Agent(self.cfg) for _ in range(3)]
        for i, a in enumerate(agents):
            with torch.no_grad():
                for module in [a.actor, a.critic]:
                    for p in module.parameters():
                        p.fill_(i)
        fedavg(agents)
        for a in agents:
            for module in [a.actor, a.critic, a.old_actor]:
                for p in module.parameters():
                    torch.testing.assert_close(p, torch.ones_like(p))

    def test_cg_linear_system(self):
        matrix = torch.tensor([[4., 1.], [1., 3.]])
        b = torch.tensor([1., 2.])
        torch.testing.assert_close(conjugate_gradient(lambda v: matrix @ v, b, 10), torch.linalg.solve(matrix, b))

    def test_all_algorithms_real_updates_and_trpo_kl(self):
        for name in ALGORITHMS:
            agents = [Agent(self.cfg) for _ in range(3)]
            batch, _ = collect_episode(MicrogridEnv(self.cfg), agents, 0)
            stats = agents[0].update([x[:, 0] for x in batch], name)
            self.assertGreater(stats['actor_delta'], 0, name)
            self.assertGreater(stats['critic_delta'], 0, name)
            if name == 'TRPO-MADRL':
                self.assertLessEqual(stats['kl'], self.cfg['trpo_max_kl'] + 1e-6)
                self.assertEqual(stats['trpo_accepted'], 1)


if __name__ == '__main__':
    unittest.main()
