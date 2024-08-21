from typing import Optional, Union, List, Tuple

from timm.scheduler import CosineLRScheduler

from torch.optim import SGD, AdamW
from torch.nn import Conv2d, Linear
import torch.nn as nn

from bort import BortS, BortA

_include_module_ = (Conv2d, Linear) # ConvLayer

def get_param(
        module: nn.Module, 
        include_module: Optional[tuple] = None,
        exclude_module: Optional[tuple] = None,
        include_name: Optional[tuple] = None,
        exclude_name: Optional[tuple] = None,
        return_all: bool = False
    ) -> Union[List, Tuple[List, List]]:
    if isinstance(exclude_name, str):
        exclude_name = tuple([exclude_name])
    
    params_id_list = []
    params_name_list = []
    params_list = []
    for name, sub_module in module.named_modules():
        if (exclude_module is None) or (not isinstance(sub_module, exclude_module)):
            if (include_module is None) or (isinstance(sub_module, include_module)):
                params_id_list += list([id(p) for p in sub_module.parameters(recurse=False)])
    
    for name, param in module.named_parameters():
        if id(param) in params_id_list:
            if (exclude_name is None) or (not any([e_n in name for e_n in exclude_name])):
                if (include_name is None) or (any([i_n in name for i_n in include_name])):
                    params_name_list.append(name)
                    params_list.append(param)

    if return_all:
        return params_name_list, params_list
    else:
        return params_list 

def get_all_other_params(module: nn.Module, params_groups: List, return_name: bool = False) -> Union[List, Tuple[List, List]]:
    exclude_id_list = []
    remain_params_name = []
    remain_params = []
    for params_group in params_groups:
        if isinstance(params_group, list):
            exclude_id_list += [id(param) for param in params_group]
        elif isinstance(params_group, dict):
            exclude_id_list += [id(param) for param in params_group["params"]]
        else:
            raise TypeError(f"Wrong params_group type: {type(params_group)}")
        
    for name, param in module.named_parameters():
        if id(param) not in exclude_id_list:
            remain_params_name.append(name)
            remain_params.append(param)
        
    if return_name:
        return remain_params_name, remain_params
    else:
        return remain_params

def give_optim(config, model: nn.Module):
    optim_name = config.optim
    optim_config = {}
    optimizer = None
    
    if "bogd" not in optim_name:
        params = []
        # weight regularize group
        params.append({
            "params": get_param(model, 
                    include_module=_include_module_,
                    exclude_name=("bias")),
            "weight_decay": config.wd
        })
        # not weight regularize group
        params.append({
            "params": get_all_other_params(model, 
                    params_groups=params),
            "weight_decay": 0.0
        })

        optim_config["params"] = params
        optim_config["lr"] = config.lr
        optim_config["weight_decay"] = config.wd
        if optim_name == "sgd":
            optim_config["momentum"] = config.momen
            optim_config["dampening"] = 0.0
            optim_config["nesterov"] = False
            optimizer = SGD(**optim_config)
        elif optim_name == "adamw":
            optim_config["betas"] = config.betas
            optim_config["eps"] = config.eps
            optim_config["amsgrad"] = False
            optimizer = AdamW(**optim_config)
        else:
            raise ValueError("Invalid optimizer name!")
    else:
        params = []
        # weight regularize group
        params.append({
            "params": get_param(model, 
                    include_module=_include_module_,
                    exclude_name=("bias")),
            "weight_decay": config.wd,
            "weight_constraint": config.wc
        })
        # not weight regularize group
        params.append({
            "params": get_all_other_params(model, 
                    params_groups=params),
            "weight_decay": 0.0,
            "weight_constraint": 0.0
        })

        optim_config["params"] = params
        optim_config["lr"] = config.lr
        optim_config["weight_constraint"] = config.wc
        optim_config["weight_decay"] = config.wd
        if optim_name == "bogd":
            optim_config["momentum"] = config.momen
            optim_config["dampening"] = 0.0
            optim_config["nesterov"] = False
            optimizer = BortS(**optim_config)
        elif optim_name == "abogd":
            optim_config["betas"] = config.betas
            optim_config["eps"] = config.eps
            optim_config["amsgrad"] = False
            optimizer = BortA(**optim_config)
        else:
            raise ValueError("Invalid optimizer name!")
    
    return optimizer

def give_scheduler(config, optimizer):
    name = config.scheduler
    if name == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config.epochs,
            lr_min=config.min_lr,
            warmup_lr_init=config.warmup_lr,
            warmup_t=config.warmup_epochs,
        )
        config.epochs = scheduler.get_cycle_length()
    elif name == "none" or name is None:
        scheduler = None
    else:
        raise KeyError(f"Invalid scheduler name: {name}")
    
    return scheduler