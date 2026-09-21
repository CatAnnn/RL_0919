import copy
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


def mlp(widths, output, final_gain):
    layers = []
    dim = 5
    for width in widths:
        layer = nn.Linear(dim, width)
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.Tanh()])
        dim = width
    layer = nn.Linear(dim, output)
    nn.init.orthogonal_(layer.weight, final_gain)
    nn.init.zeros_(layer.bias)
    layers.append(layer)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = mlp(cfg['hidden_sizes'], 2, .01)
        self.log_std = nn.Parameter(torch.full((2,), cfg['initial_log_std']))

    def distribution(self, obs):
        return Normal(self.net(obs), self.log_std.exp())

    def log_prob(self, obs, latent):
        # 潜变量联合密度：相同tanh变换的雅可比在新旧策略比中抵消。
        return self.distribution(obs).log_prob(latent).sum(-1)


def gae(rewards, values, next_values, done, gamma, lam):
    delta = rewards + gamma * next_values * (1 - done) - values
    advantage = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        running = delta[t] + gamma * lam * (1 - done[t]) * running
        advantage[t] = running
    return advantage


def flat_grad(scalar, params, create_graph=False):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(scalar, params, create_graph=create_graph)])


def conjugate_gradient(product, b, iterations):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = r.dot(r)
    for _ in range(iterations):
        ap = product(p)
        denom = p.dot(ap)
        if denom <= 0 or rr < 1e-12:
            break
        alpha = rr / denom
        x += alpha * p
        r -= alpha * ap
        new_rr = r.dot(r)
        p = r + (new_rr / rr) * p
        rr = new_rr
    return x


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actor = Actor(cfg)
        self.critic = mlp(cfg['hidden_sizes'], 1, 1)
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_learning_rate'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])

    def value(self, obs):
        return self.critic(obs).squeeze(-1) * self.cfg['value_output_scale']

    def sync_old(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def trpo_step(self, obs, action, old_logp, advantage):
        cfg = self.cfg
        params = list(self.actor.parameters())
        with torch.no_grad():
            old_dist = self.old_actor.distribution(obs)
            old_dist = Normal(old_dist.loc.detach(), old_dist.scale.detach())
        objective = lambda: ((self.actor.log_prob(obs, action) - old_logp).exp() * advantage).mean()
        divergence = lambda: kl_divergence(old_dist, self.actor.distribution(obs)).sum(-1).mean()
        before = torch.nn.utils.parameters_to_vector(params).detach().clone()
        base = objective()
        gradient = flat_grad(base, params).detach()

        def product(v):
            first = flat_grad(divergence(), params, create_graph=True)
            return flat_grad((first * v).sum(), params).detach() + cfg['trpo_damping'] * v

        direction = conjugate_gradient(product, gradient, cfg['trpo_cg_iterations'])
        curvature = direction.dot(product(direction))
        accepted = False
        if torch.isfinite(curvature) and curvature > 0:
            step = direction * torch.sqrt(2 * cfg['trpo_max_kl'] / curvature)
            expected = gradient.dot(step).item()
            for k in range(cfg['trpo_line_search_steps']):
                fraction = .5 ** k
                torch.nn.utils.vector_to_parameters(before + fraction * step, params)
                with torch.no_grad():
                    improvement = (objective() - base).item()
                    kl = divergence().item()
                if math.isfinite(kl) and kl <= cfg['trpo_max_kl'] and improvement > max(0, .1 * fraction * expected):
                    accepted = True
                    break
        if not accepted:
            torch.nn.utils.vector_to_parameters(before, params)
        return {'actor_loss': -objective().item(), 'kl': divergence().item(),
                'trpo_accepted': int(accepted), 'actor_updates': int(accepted),
                'actor_grad_norm': gradient.norm().item()}

    def update(self, batch, algorithm):
        cfg = self.cfg
        obs, actions, rewards, next_obs, done, old_logp = [torch.as_tensor(x, dtype=torch.float32) for x in batch]
        with torch.no_grad():
            values = self.value(obs)
            next_values = self.value(next_obs)
            advantages = gae(rewards, values, next_values, done, cfg['gamma'], cfg['gae_lambda'])
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            # 式27采用单步TD目标，终止处不bootstrap，内循环不反向传播目标。
            target = rewards + cfg['gamma'] * next_values * (1 - done)
        before_actor = torch.nn.utils.parameters_to_vector(self.actor.parameters()).detach().clone()
        before_critic = torch.nn.utils.parameters_to_vector(self.critic.parameters()).detach().clone()
        if algorithm == 'TRPO-MADRL':
            stats = self.trpo_step(obs, actions, old_logp, advantages)
        else:
            epochs = 1 if algorithm == 'A2C-MADRL' else cfg['ppo_epochs']
            for _ in range(epochs):
                logp = self.actor.log_prob(obs, actions)
                if algorithm == 'A2C-MADRL':
                    loss = -(logp * advantages).mean()
                else:
                    ratio = (logp - old_logp).exp()
                    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - cfg['ppo_clip'], 1 + cfg['ppo_clip']) * advantages).mean()
                self.actor_opt.zero_grad()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg['max_grad_norm'])
                self.actor_opt.step()
            with torch.no_grad():
                kl = kl_divergence(self.old_actor.distribution(obs), self.actor.distribution(obs)).sum(-1).mean().item()
            stats = {'actor_loss': loss.item(), 'kl': kl, 'actor_updates': epochs,
                     'trpo_accepted': 0, 'actor_grad_norm': float(norm)}
        for _ in range(cfg['critic_epochs']):
            loss_v = ((self.value(obs) - target) ** 2).mean()
            self.critic_opt.zero_grad()
            loss_v.backward()
            vnorm = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg['max_grad_norm'])
            self.critic_opt.step()
        stats.update(critic_loss=loss_v.item(), critic_grad_norm=float(vnorm), critic_updates=cfg['critic_epochs'],
                     actor_delta=float((torch.nn.utils.parameters_to_vector(self.actor.parameters()) - before_actor).norm()),
                     critic_delta=float((torch.nn.utils.parameters_to_vector(self.critic.parameters()) - before_critic).norm()))
        if not all(np.isfinite(v) for v in stats.values()):
            raise FloatingPointError(stats)
        return stats

    def state(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'old_actor': self.old_actor.state_dict(), 'actor_opt': self.actor_opt.state_dict(),
                'critic_opt': self.critic_opt.state_dict()}

    def restore(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.old_actor.load_state_dict(state['old_actor'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
