import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
#random: Python's random maker
#numpy: NumPy's random maker
#torch: PyTorch's random maker
    
    #Inputs: model (the trained model) and path (where to save it, like config.BEST_MODEL_PATH).

def save_model(model,path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # path.parent is the folder, models/.]
    # mkdir makes the folder.
    # exist_ok=True means "if the folder is already there, that's okay, no error".

    torch.save(model.state_dict(), path)
    # model.state_dict() is what the model learned, all its numbers.
    # torch.save writes those numbers into the file.
    
    
# inputs: model (an empty model with the same design), path (the saved file) and device ("mps" or "cpu")
def load_model(model, path, device):
    #Line 1: torch.load(path, map_location=device) reads the numbers from the file. Then model.load_state_dict(...) puts those numbers into the empty model.
    model.load_state_dict(torch.load(path, map_location = device))
    # Line 2: model.to(device) moves the model to the GPU (mps).

    model.to(device)
    model.eval()
    #Line 3: model.eval() switches the model to exam mode.    
    #return model
    #During training, some parts of the model act a little randomly, to help it learn.
    #In exam mode, they stop, so the answers are stable.
    return model 

