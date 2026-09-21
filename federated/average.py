import torch


def fedavg(agents):
    """只交换模型参数，保留各本地优化器状态。"""
    for name in ['actor', 'critic']:
        states = [getattr(agent, name).state_dict() for agent in agents]
        mean = {key: torch.stack([state[key] for state in states]).mean(0) for key in states[0]}
        for agent in agents:
            getattr(agent, name).load_state_dict(mean)
    for agent in agents:
        agent.sync_old()
