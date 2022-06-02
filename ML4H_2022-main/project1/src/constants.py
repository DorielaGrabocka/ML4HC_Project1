try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

except ImportError:
    print("PyTorch not available")


PATH_MITBIH_DATA = "data/archive"
PATH_PTBDB_DATA = "data/archive"
FOLDER_NAME_MITBIH = "src/tree_models/mitbih/"
FOLDER_NAME_PTBDB = "src/tree_models/ptbdb/"

#this is only for colab
# PATH_MITBIH_DATA = "/content/drive/MyDrive/ML4HC/project1/data/archive"
# PATH_PTBDB_DATA = "/content/drive/MyDrive/ML4HC/project1/data/archive"
# FOLDER_NAME_MITBIH = "/content/drive/MyDrive/ML4HC/project1/src/tree_models/mitbih/"
# FOLDER_NAME_PTBDB = "/content/drive/MyDrive/ML4HC/project1/src/tree_models/ptbdb/"