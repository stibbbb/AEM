import torch
from torch.optim.optimizer import Optimizer

class OurProposedSGD(Optimizer):
    def __init__(self, params, lr=0.01, s=1.2, d=0.95, iter_decay=10, k_init=2.0, k_decay=0.95):
        defaults = dict(lr=lr, s=s, d=d, iter_decay=iter_decay, step=0, k_init=k_init, k_decay=k_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _angle_between(v1, v2):
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        if v1_norm.item() == 0 or v2_norm.item() == 0:
            return torch.tensor(0.0)
        cosine = torch.clamp(torch.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        angle_rad = torch.acos(cosine)
        return angle_rad / torch.pi  # Normalize to [0, 1]

    @staticmethod
    def F(angle, s, k):
        # Nonlinear coefficient function using sigmoid shape
        return s * (1 + (-2 / (1 + torch.exp(-k * (2 * angle - 1)))))

    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            # Increment step count first for correct decay timing
            group['step'] += 1
            s, d, step_num = group['s'], group['d'], group['step']
            k = group['k_init']

            # Decay parameters every iter_decay steps
            if step_num > 0 and step_num % group['iter_decay'] == 0:
                group['s'] *= d
                group['k_init'] *= group['k_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'prev_grad' not in state:
                    # Initialize prev_grad with small noise to avoid zero angle
                    state['prev_grad'] = grad.clone().detach() + 1e-8 * torch.randn_like(grad)
                    p.data -= group['lr'] * grad
                    continue

                prev_grad = state['prev_grad']
                angle = self._angle_between(prev_grad.view(-1), grad.view(-1))
                coefpg = self.F(angle, s, k)  # Coefficient based on angle and k
                coefcg = s - coefpg

                new_grad = coefpg * prev_grad + coefcg * grad
                p.data -= group['lr'] * new_grad

                state['prev_grad'] = grad.clone().detach()

        return loss


class AGSGD(Optimizer):
    def __init__(self, params, lr=0.01, s=1.2, d=0.95, iter_decay=10):
        defaults = dict(lr=lr, s=s, d=d, iter_decay=iter_decay, step=0)
        super().__init__(params, defaults)

    def _angle_between(self, v1, v2):
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        if v1_norm.item() == 0 or v2_norm.item() == 0:
            return torch.tensor(0.0)
        cosine = torch.clamp(torch.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        angle_rad = torch.acos(cosine)
        return angle_rad / torch.pi  # Normalize to [0, 1]

    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            group['step'] += 1
            s = group['s']
            d = group['d']
            step_num = group['step']

            if step_num > 0 and step_num % group['iter_decay'] == 0:
                group['s'] *= d

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'prev_grad' not in state:
                    state['prev_grad'] = grad.clone().detach() + 1e-8 * torch.randn_like(grad)
                    p.data -= group['lr'] * grad
                    continue

                prev_grad = state['prev_grad']
                angle = self._angle_between(prev_grad.view(-1), grad.view(-1))

                # AG-SGD coefficient calculation as per paper
                coefpg = -s * (2 * angle - 1)
                coefcg = s - coefpg

                new_grad = coefpg * prev_grad + coefcg * grad
                p.data -= group['lr'] * new_grad

                state['prev_grad'] = grad.clone().detach()

        return loss
