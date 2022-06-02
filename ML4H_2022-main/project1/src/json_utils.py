import json
import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch


def save_file(path_to_file: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, 'w') as f:
        json.dump(data, f)


def read_file(path_to_file: str) -> Dict[str, Any]:
    with open(path_to_file) as json_file:
        data = json.load(json_file)
    return data


def serialize_tensors(params: Dict[str, Any]):
    """
    needed to add class weights as a searchable parameter and to be able to save the params config
    works for grid_params and just params
    """
    params = deepcopy(params)
    v = params["criterion__weight"]
    if type(v) == torch.Tensor:
        params["criterion__weight"] = list(v.cpu().numpy().astype(np.float64))
    elif type(v) == list:
        for i, el in enumerate(v):
            if type(el) != torch.Tensor:
                raise ValueError("Something is not right in the dict structure")

            v[i] = list(el.cpu().numpy().astype(np.float64))
    else:
        raise ValueError("Something is not right in the dict structure")

    return params


def deserialize_tensors(params: Dict[str, Any]):
    """
    gets tensors ready so that they can be passed
    works only for params!
    """
    params = deepcopy(params)
    if type(params["criterion__weight"][0]) == list:
        raise ValueError("This should be a simple list of floats")
    params["criterion__weight"] = torch.Tensor(params["criterion__weight"]).float()
    return params
