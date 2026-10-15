import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    array = np.arange(N)
    if shuffle:
        array = np.random.default_rng(seed).permutation(array)
        
    Output = []
    q=N//k
    r=N%k

    start = 0
    
    for i in range(k) :
        if i < r :
            size = q + 1
        else :
            size = q
            
        end = start + size
        val = array[start:end]
        train = np.concatenate((array[:start], array[end:]))
        Output.append({"train_idx":train, "val_idx":val})
        start = end
        
    return Output

# import numpy as np

# def kfold_split(N, k, shuffle=True, seed=0):
#     indices = np.arange(N)
#     if shuffle:
#         indices = np.random.default_rng(seed).permutation(indices)
#     folds = np.array_split(indices, k)
#     result = []
#     for index, validation in enumerate(folds):
#         training = np.concatenate(folds[:index] + folds[index + 1:])
#         result.append({"train_idx": training.astype(int), "val_idx": validation.astype(int)})
#     return result   