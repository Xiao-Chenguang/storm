import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional

__all__ = ["STORM", "storm"]


class STORM(Optimizer):
    def __init__(
        self,
        params,
        k=0.1,
        w=0.1,
        c=required,
        weight_decay=0,
        *,
        maximize=False,
        foreach: Optional[bool] = None,
        differentiable=False,
    ):
        if k < 0.0:
            raise ValueError("Invalid k value: {}".format(k))
        if w < 0.0:
            raise ValueError("Invalid w value: {}".format(w))
        if c is not required and c < 0.0:
            raise ValueError("Invalid c value: {}".format(c))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            k=k,
            w=w,
            c=c,
            lr=1,
            momentum=1,
            max_norm=1e-2,
            g2=1e-6,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
        super(STORM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    def store_next_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if "next_grad" in self.state[p]:
                        self.state[p]["next_grad"][0] = self.state[p]["next_grad"][1]
                        self.state[p]["next_grad"][1] = p.grad.clone()
                    else:
                        self.state[p]["next_grad"] = [None, p.grad.clone()]
        self.zero_grad()

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            prev_d_p_list = []
            has_sparse_grad = False

            for p in group["params"]:
                if p.grad is not None:
                    # 1. update G^2: g2 = g2 + ||g||^2
                    group["max_norm"] = max(group["max_norm"], p.grad.norm(2).item())
                    group["g2"] += p.grad.norm(2)

                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])
                    prev_d_p_list.append(state["next_grad"][0])

            # 2. update eta: eta = k / (g2 + w)^(1/3)
            group["lr"] = group["k"] * (group["g2"] + group["w"]) ** (-1 / 3)

            # 3. update p and d_p
            storm(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                prev_d_p_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                maximize=group["maximize"],
                max_norm=group["max_norm"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

            # 4. update momentum a: a = 1 - c * eta^2
            group["momentum"] = 1 - min(1, group["c"] * group["lr"] ** 2)
        return loss


def storm(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    prev_d_p_list: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: bool = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    maximize: bool,
    max_norm: Optional[float] = None,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_storm
    else:
        func = _single_tensor_storm

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        prev_d_p_list=prev_d_p_list,
        lr=lr,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        max_norm=max_norm,
    )


def _single_tensor_storm(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    prev_d_p_list: List[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
    max_norm: Optional[float] = None,
):
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
        prev_d_p = prev_d_p_list[i] if not maximize else -prev_d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.sub_(prev_d_p).mul_(momentum).add_(d_p)

            d_p = buf

        if max_norm is not None:
            d_p = d_p.clamp(-max_norm, max_norm)

        param.add_(d_p, alpha=-lr)


def _multi_tensor_storm(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    prev_d_p_list: List[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
    max_norm: Optional[float] = None,
):
    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]
        prev_d_p_list = torch._foreach_neg(tuple(prev_d_p_list))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_sub_(bufs, prev_d_p_list)
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.sub_(prev_d_p_list[i]).mul_(momentum).add_(grads[i])

                bufs.append(buf)

        grads = bufs

    if max_norm is not None:
        grads = [grad.clamp(-max_norm, max_norm) for grad in grads]

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)
