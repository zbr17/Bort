from typing import Union
import torch
from torch import Tensor

def bort_correction(param: Tensor, amptitude: Union[float, str] = 1.0, mode: str = "l2-full"):
    """
    Args:
        param (Tensor): size [N, ...]
        amptitude (float): correction amptitude
        mode (str): correction mode. "row": bounded row constraint only; "col": bounded column constraint only.
            l2-full: full l2 correction.
            l2-row: row l2 correction.
            l2-col: column l2 correction.
            l1-full: full l1 correction.
            l1-row: row l1 correction.
            l1-col: column l1 correction.
    """
    param_flatten = param.view(param.size(0), -1)
    # compute the amptitude
    if amptitude == "ada":
        amptitude = param_flatten.norm(dim=1).mean().pow(2).item()

    # l2 penalty
    if mode == "l2-full":
        param_delta = (
            param_flatten @ param_flatten.t() @ param_flatten - 
            amptitude * param_flatten
        )
            
    elif mode == "l2-row":
        param_eyes = torch.eye(param_flatten.size(0), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            ((param_flatten @ param_flatten.t()) * param_eyes) @ param_flatten - 
            amptitude * param_flatten
        )
    elif mode == "l2-col":
        param_eyes = torch.eye(param_flatten.size(1), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            param_flatten @ ((param_flatten.t() @ param_flatten) * param_eyes) -
            amptitude * param_flatten
        )
    # l1 penalty
    elif mode == "l1-fullrow":
        param_eyes = torch.eye(param_flatten.size(0), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            torch.sign(param_flatten @ param_flatten.t() - amptitude * param_eyes) @ param_flatten
        )
    elif mode == "l1-fullcol":
        param_eyes = torch.eye(param_flatten.size(1), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            param_flatten @ torch.sign(param_flatten.t() @ param_flatten - amptitude * param_eyes)
        )
    elif mode == "l1-row":
        param_eyes = torch.eye(param_flatten.size(0), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            (torch.sign(param_flatten @ param_flatten.t() - amptitude * param_eyes) * param_eyes) @ param_flatten
        )
    elif mode == "l1-col":
        param_eyes = torch.eye(param_flatten.size(1), device=param_flatten.device, dtype=param_flatten.dtype)
        param_delta = (
            param_flatten @ (torch.sign(param_flatten.t() @ param_flatten - amptitude * param_eyes) * param_eyes)
        )
    else:
        raise ValueError(f"mode {mode} not supported")
    param_delta = param_delta.view(param.size())
    return param_delta
