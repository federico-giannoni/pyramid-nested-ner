import torch


def get_default_sgd_optim(params, lr=1e-2, momentum=0.9, inverse_time_lr_decay=True):
    """
    Returns the default SGD optimizer used in the paper and the LR scheduler.

    :param params:
    :param lr:
    :param momentum:
    :param inverse_time_lr_decay:
    :return:
    """

    def inverse_time_decay(last_epoch, steps_per_epoch=235, decay_rate=0.05, decay_steps=1000):
        if last_epoch and not last_epoch % 4:
            return 1 / (1 + steps_per_epoch * last_epoch * decay_rate / decay_steps)
        return 1

    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=1e-6)
    if inverse_time_lr_decay:
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=inverse_time_decay)
    else:
        scheduler = None

    return optimizer, scheduler
