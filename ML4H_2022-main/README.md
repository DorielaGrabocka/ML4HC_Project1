# ML4H_2022_Project1

# Where to place the data
The files with the data should be unzipped inside the *project1* folder (paths to the data are configured in the *constants.py* file)

# Creating the conda environment and adding it as a jupyter notebook kernel
To reproduce our conda environment run (it will create an environment called *ml4h_p1*):

`conda env create -f conda_env.yml`

To make it discoverable by jupyter notebooks run:

`conda activate ml4h_p1`

and

`python -m ipykernel install --user --name ml4h_p1 --display-name "ml4h_p1"`

Don't forget to also select the kernel once the notebook is opened!

# How to run the code
Each task has a designated notebook to solve it (eg. *Task1*). The only thing that needs to be chosen is the dataset name at the beginning of it. Afterwards, one can run each cell of the notebook.

**Important!** The tasks (notebooks) are expected to be run one after the next, because the notebooks for Task1 and Task2 will persist the trained models, and then afterwards the notebook for Task3 (ensembling) will just load them.

Optionally, you may also use our pretrained models:
- For the CNN, this can be done by modifying the relevant path under the "CNN Loading" Section in the Task3 notebook.
- For the advanced RNNs, those models can be found under project1/pretrained_models and can be loaded with the load_weights() method
- For the ensembles of tree algorithms in Task3 notebook you can still just load them by running the first cell under the *Loading Tree Models* section. You will find the saved models under *src/tree_models/mitbih/* or *src/tree_models/ptbdb/* for each dataset respectively. In order to change the path, you can find the path in the src/constants.py file. The responsible variables will be FOLDER_NAME_MITBIH, FOLDER_NAME_PTBDB, if any problem should arise from loading.

# Additional resources
The notebooks for the tasks themselves already have the best parameters set. Those were computed using different notebooks, for the CNNs, this was done using the *CNN_Hyperparameter_Search* notebook, which has further instructions within it. For the tree algorithms this was done in the notebooks *Task 2&3 - FinalModels.ipynb* and *Task 2&3 GriDSearch - Final.ipynb* inside folder *trees_hyperparamter_grid_search*.

