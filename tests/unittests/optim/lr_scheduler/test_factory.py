from collections import OrderedDict

import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from clinicadl.optim.lr_scheduler import (
    ImplementedLRScheduler,
    LRSchedulerConfig,
    get_lr_scheduler,
)


def test_get_lr_scheduler():
    network = nn.Sequential(
        OrderedDict(
            [
                ("linear1", nn.Linear(4, 3)),
                ("linear2", nn.Linear(3, 2)),
            ]
        )
    )
    optimizer = SGD(
        [
            {
                "params": network.linear1.parameters(),
                "lr": 1.0,
            },
            {
                "params": network.linear2.parameters(),
            },
        ],
        lr=10.0,
    )

    for scheduler in [e.value for e in ImplementedLRScheduler]:
        if scheduler == "StepLR":
            config = LRSchedulerConfig(scheduler=scheduler, step_size=1)
        elif scheduler == "MultiStepLR":
            config = LRSchedulerConfig(scheduler=scheduler, milestones=[1, 2, 3])
        else:
            config = LRSchedulerConfig(scheduler=scheduler)
        get_lr_scheduler(optimizer, config)

    config = LRSchedulerConfig(
        scheduler="ReduceLROnPlateau",
        mode="max",
        factor=0.123,
        threshold=1e-1,
        cooldown=3,
        min_lr={"linear2": 0.01, "linear1": 0.1},
    )
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler.factor == 0.123
    assert scheduler.patience == 10
    assert scheduler.threshold == 1e-1
    assert scheduler.threshold_mode == "rel"
    assert scheduler.cooldown == 3
    assert scheduler.min_lrs == [0.1, 0.01]
    assert scheduler.eps == 1e-8

    assert updated_config.scheduler == "ReduceLROnPlateau"
    assert updated_config.mode == "max"
    assert updated_config.factor == 0.123
    assert updated_config.patience == 10
    assert updated_config.threshold == 1e-1
    assert updated_config.threshold_mode == "rel"
    assert updated_config.cooldown == 3
    assert updated_config.min_lr == {"linear2": 0.01, "linear1": 0.1}
    assert updated_config.eps == 1e-8

    network.add_module("linear3", nn.Linear(3, 2))
    optimizer.add_param_group({"params": network.linear3.parameters()})
    config.min_lr = {"ELSE": 1, "linear2": 0.01, "linear1": 0.1}
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert scheduler.min_lrs == [0.1, 0.01, 1]

    config = LRSchedulerConfig()
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, LambdaLR)
    assert updated_config.scheduler is None
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [1.0, 10.0, 10.0]