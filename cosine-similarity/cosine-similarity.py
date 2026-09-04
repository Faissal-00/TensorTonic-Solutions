import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    A=np.array(a,dtype=float)
    B=np.array(b,dtype=float)
    Scalar=float(A@B) 
    ScalarA=float(np.sqrt(A@A))
    ScalarB=float(np.sqrt(B@B)) 
    if ScalarA == 0 or ScalarB ==0 :
        Output=0.0
    else :
        Output=Scalar / (ScalarA*ScalarB)
    return Output