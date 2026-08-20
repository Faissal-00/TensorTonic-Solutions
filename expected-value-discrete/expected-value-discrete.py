import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Return the expected value of the discrete distribution.
    """
    xi=np.array(x) #[1,2,3]
    pi=np.array(p) #[0.2,0.5,0.3]
    # xipi=xi*pi #[0.2, 1, 0.9]
    # e=sum(xipi) # 2.1
    e = np.dot(xi, pi) 
    return float(e)
    pass


# instead of doing this in two steps multiply then sum
# just do e = np.dot(xi, pi) 