from typing import Any, Dict, List

import numpy as np
import sklearn
import skorch
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

from src.constants import DEVICE


def get_neural_net_classifier(module: nn.Module, n_classes: int,
                              callbacks: List[skorch.callbacks.Callback] = [], train_split: int = 5,
                              params: Dict[str, Any] = {}) -> skorch.NeuralNetClassifier:
    """
    train_split: int
    - number of folds to split the data in (keeping the same name as in skorch)
    """
    return NeuralNetClassifier(
        module=module,
        module__n_classes=n_classes,

        criterion=torch.nn.NLLLoss,
        train_split=skorch.dataset.ValidSplit(train_split),

        callbacks=callbacks,

        iterator_train__shuffle=True,
        device=DEVICE,
        verbose=10,

        # use adam, more stable than SGD with momentum
        optimizer=torch.optim.Adam,

        # set to a high value since we use early stopping anyway
        max_epochs=200,

        **params
    )


def get_class_weights(y: np.ndarray, unbalanced: bool = False) -> torch.Tensor:
    if unbalanced:
        weights = torch.tensor([1] * len(np.unique(y)))
    else:
        weights = torch.tensor(sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y))
    return weights.float()
