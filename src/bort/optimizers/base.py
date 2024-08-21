import torch
from torch import Tensor

def bort_correction(param: Tensor, amptitude: float = 1.0, mode: str = "default"):
    """
    Args:
        param (Tensor): size [N, ...]
        amptitude (float): correction amptitude
        mode (str): correction mode. "default": bounded and orthogonal constraints; "row": bounded row constraint only; "col": bounded column constraint only.
    """
    param_flatten = param.view(param.size(0), -1)
    if mode == "default":
        param_delta = (
            param_flatten @ param_flatten.t() @ param_flatten - 
            amptitude * param_flatten
        )
            
    elif mode == "row":
        param_eyes = torch.eye(param_flatten.size(0), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            ((param_flatten @ param_flatten.t()) * param_eyes) @ param_flatten - 
            amptitude * param_flatten
        )
    elif mode == "col":
        param_eyes = torch.eye(param_flatten.size(1), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            param_flatten @ ((param_flatten.t() @ param_flatten) * param_eyes) -
            amptitude * param_flatten
        )
    param_delta = param_delta.view(param.size())
    return param_delta
