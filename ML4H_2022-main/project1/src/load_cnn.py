from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler

from src.cnn_models.cnn import CNN
from src.json_utils import (deserialize_tensors, read_file, save_file,
                            serialize_tensors)
from src.skorch_utils import get_neural_net_classifier


def load_cnn_model(net_type: str, dataset: str, n_classes):
    dir_name = net_type + '_' + dataset
    cp = Checkpoint(dirname=dir_name)
    params = deserialize_tensors(read_file(dir_name + '/params.json'))

    loaded_net = get_neural_net_classifier(module=CNN, n_classes=n_classes, params=params)
    loaded_net.initialize()
    loaded_net.load_params(checkpoint=cp)

    return loaded_net
