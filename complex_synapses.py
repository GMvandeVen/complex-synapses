import math
import torch
from torch.optim.optimizer import Optimizer


def print_info(n_beakers, alpha, beta, adam=False):
    '''Print information on the complex synapse algorithm with chosen settings to the screen.'''
    print("Complex Synapse Optimizer...")
    if n_beakers == 0:
        print(" --> # of beakers: 0 (i.e., standard {} optimizer)".format("Adam" if adam else "SGD"))
    else:
        print(" --> # of beakers:{}{:6}".format("        " if n_beakers > 1 else "", n_beakers))
        if n_beakers > 1:
            print(" --> shortest time-scale: {:6}".format(int(alpha)))
        print(" --> {}time-scale: {:7}".format("longest " if n_beakers > 1 else "", int(beta)))



class ComplexSynapse(Optimizer):
    '''Implements the complex synapse algorithm (Benna & Fusi, 2016) as an SGD-based PyTorch-optimizer.

    Args:
        params (iterable): iterable of parameters (`synapses`) to optimize or iterable of dicts defining param groups
        lr (float, optional): learning rate (default: 0.1)
        n_beakers (int, optional): number of beakers (0 = standard; 1 = only decay at timescale ``beta``; default: 6)
        alpha (float, optional): shortest synaptic timescale (timescale of 1st beaker = ``C_1/g_{1,2}``; default: 1)
        beta (float, optional): longest synaptic timescale (timescale of last beaker = ``C_k/g_{k,k+1}``; default: 1024)
        init (str, optional, `same`|`zero`): how should the beakers be initialized (default: `same`)
        verbose (bool, optional): if ``True``, information about chosen settings is printed to screen

    NOTES:
        - the arguments `n_beakers`, `alpha` and `beta` cannot be set per parameter-group!
        - if `n_beakers` is set to 0, this corresponds to standard SGD
        - if `n_beakers` is set to 1, this corresponds to standard SGD with decay of ``1/beta``
    '''

    def __init__(self, params, lr=0.1, n_beakers=6, alpha=1., beta=1024.,
                 init='same', verbose=False):

        # Check for invalid arguments
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not init in ('same', 'zero'):
            raise ValueError("Invalid initialization code: {}".format(init))

        # Deal with arguments set per parameter group
        defaults = dict(lr=lr, init=init)
        super(ComplexSynapse, self).__init__(params, defaults)

        # Set the parameters of the beakers
        self.n_beakers = n_beakers
        self.alpha = alpha
        self.beta = beta
        self.x = (beta/alpha)**(1/(2*n_beakers-2)) if n_beakers>1 else beta
        for id in range(1, n_beakers+1):
            setattr(self, 'C{}'.format(id), self.x**(id-1))
            setattr(self, 'g{}_{}'.format(id,id+1), (1/alpha) * (self.x**(1-id)) if n_beakers>1 else (1/beta))
            # -> if only 1 beaker (i.e., only decay), the longest time scale is used for that beaker

        # If requested, print information to the screen
        if verbose:
            print_info(n_beakers, alpha, beta, adam=False)


    def step(self, closure=None):
        '''Performs a single optimization step.

        Args:
            closure (callable, optional): a closure that re-evaluates the model and returns the loss
        '''
        loss = None
        if closure is not None:
            loss = closure()

        # Loop over all parameter-groups
        for group in self.param_groups:

            # Loop over all parameters within this parameter-group
            for p in group['params']:

                # Get the gradient for this parameter `p`
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ComplexSynapse does not support sparse gradients.')

                # Get the state for this parameter (initially this is an empty dict)
                state = self.state[p]

                # If not yet done, initialize the state
                if len(state)==0 and self.n_beakers>0:
                    init = group['init']
                    # -"level of the beakers"
                    for beaker_id in range(2, self.n_beakers+1):
                        if init=="same":
                            state['u{}'.format(beaker_id)] = p.data.clone().detach()
                        elif init=="zero":
                            state['u{}'.format(beaker_id)] = torch.zeros_like(p.data)
                    # -last 'extra' beaker is not really a beaker, but a leak-term
                    state['u{}'.format(self.n_beakers+1)] = torch.zeros_like(p.data)

                # Update the synaptic strength (i.e., the first beaker)
                step_size = group['lr']/self.C1 if self.n_beakers>0 else group['lr']
                change = (-grad + self.g1_2 * (state['u2']-p.data)) if (self.n_beakers>0) else -grad
                p.data.add_(step_size, change)

                # Update all other beakers one-by-one
                if self.n_beakers>1:
                    state['u1'] = p.data
                    for id in range(2, self.n_beakers+1):
                        step_size = group['lr'] / getattr(self, 'C{}'.format(id))
                        inflow = getattr(self, 'g{}_{}'.format(id-1, id)) * (
                            state['u{}'.format(id-1)]-state['u{}'.format(id)]
                        )
                        backflow = getattr(self, 'g{}_{}'.format(id,id+1)) * (
                            state['u{}'.format(id+1)]-state['u{}'.format(id)]
                        )
                        state['u{}'.format(id)].add_(step_size, inflow+backflow)

        # If provided, execute and return the closure-object
        return loss




class AdamComplexSynapse(Optimizer):
    '''Implements the complex synapse algorithm (Benna & Fusi, 2016) as a PyTorch-optimizer, combined with Adam.

    Args:
        params (iterable): iterable of parameters (`synapses`) to optimize or iterable of dicts defining param groups
        lr (float, optional): learning rate (default: 0.001)
        betas (tuple, optional): coefs for computing running mean of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        n_beakers (int, optional): number of beakers (0 = standard; 1 = only decay at timescale ``beta``; default: 6)
        alpha (float, optional): shortest synaptic timescale (timescale of 1st beaker = ``C_1/g_{1,2}``; default: 1)
        beta (float, optional): longest synaptic timescale (timescale of last beaker = ``C_k/g_{k,k+1}``; default: 1024)
        init (str, optional, `same`|`zero`): how should the beakers be initialized (default: `same`)
        verbose (bool, optional): if ``True``, information about chosen settings is printed to screen

    NOTES:
        - the arguments `n_beakers`, `alpha` and `beta` cannot be set per parameter-group!
        - if `n_beakers` is set to 0, this corresponds to standard SGD
        - if `n_beakers` is set to 1, this corresponds to standard SGD with decay of ``1/beta``
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, n_beakers=6, alpha=1., beta=1024.,
                 init='same', verbose=False):

        # Check for invalid arguments
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not init in ('same', 'zero', 'random'):
            raise ValueError("Invalid initialization code: {}".format(init))

        # Deal with arguments set per parameter group
        defaults = dict(lr=lr, betas=betas, eps=eps, init=init)
        super(AdamComplexSynapse, self).__init__(params, defaults)

        # Set the parameters of the beakers
        self.n_beakers = n_beakers
        self.alpha = alpha
        self.beta = beta
        self.x = (beta/alpha)**(1/(2*n_beakers-2)) if n_beakers>1 else beta
        for id in range(1, n_beakers+1):
            setattr(self, 'C{}'.format(id), self.x**(id-1))
            setattr(self, 'g{}_{}'.format(id,id+1), (1/alpha) * (self.x**(1-id)) if n_beakers>1 else (1/beta))
            # -> if only 1 beaker, the longest time scale is used for that beaker

        # If requested, print information to the screen
        if verbose:
            print_info(n_beakers, alpha, beta, adam=True)


    def step(self, closure=None):
        '''Performs a single optimization step.

        Args:
            closure (callable, optional): a closure that re-evaluates the model and returns the loss
        '''
        loss = None
        if closure is not None:
            loss = closure()

        # Loop over all parameter-groups
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            # Loop over all parameters within this parameter-group
            for p in group['params']:

                # Get the gradient for this parameter `p`
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamComplexSynapse does not support sparse gradients.')

                # Get the state for this parameter (initially this is an empty dict)
                state = self.state[p]

                # If not yet done, initialize the state
                if len(state)==0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # The beakers of the Complex-Synapse part
                    if self.n_beakers>0:
                        init = group['init']
                        # -"level of the beakers"
                        for beaker_id in range(2, self.n_beakers+1):
                            if init=="same":
                                state['u{}'.format(beaker_id)] = p.data.clone().detach()
                            elif init=="zero":
                                state['u{}'.format(beaker_id)] = torch.zeros_like(p.data)
                            elif init=="random":
                                raise NotImplementedError()
                        # -last 'extra' beaker is not really a beaker, but a leak-term
                        state['u{}'.format(self.n_beakers+1)] = torch.zeros_like(p.data)

                # Keep track of number of updates so far
                state['step'] += 1

                # Read out the relevant state variables for the Adam-part
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Get the by Adam "proposed_change" (i.e., the input to the Complex Synapse algorithm)
                proposed_change = exp_avg / denom

                # Calculate bias-correction and step-size
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                adjusted_lr = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Update the synaptic strength (i.e., the first beaker)
                step_size = adjusted_lr/self.C1 if self.n_beakers>0 else adjusted_lr
                change = (-proposed_change + self.g1_2*(state['u2']-p.data)) if (self.n_beakers>0) else -proposed_change
                p.data.add_(step_size, change)

                # Update all other beakers one-by-one
                if self.n_beakers>1:
                    state['u1'] = p.data
                    for id in range(2, self.n_beakers+1):
                        step_size = adjusted_lr / getattr(self, 'C{}'.format(id))  ## QUESTION: adjusted or normal lr?
                        inflow = getattr(self, 'g{}_{}'.format(id-1, id)) * (
                            state['u{}'.format(id-1)]-state['u{}'.format(id)]
                        )
                        backflow = getattr(self, 'g{}_{}'.format(id,id+1)) * (
                            state['u{}'.format(id+1)]-state['u{}'.format(id)]
                        )
                        state['u{}'.format(id)].add_(step_size, inflow+backflow)

        # If provided, execute and return the closure-object
        return loss