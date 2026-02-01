############################################################
#                       Sympy
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

def PrintLatexStyleSymPyEquation(spe):
    """
    Function that displays a SymPy expression (spe) in a jupyter notebbok after its conversion into a LaTeX / Math output

    Input:
    spe: SymPy expression

    Output:
    Pretty printing of spe

    """
    from IPython.display import display,Math
    import sympy as sym
    display(Math(sym.latex(spe)))
    return

def e2Lists(eigenvectors, sort=False):
    '''
    returns two separate lists from the list of tuples returned by the eigenvects() function of SymPy
    input
        - the list of tuples returned by eigenvects
        - sort (default: False): returns sorted eigenvalues and corresponding eigenvectors if True
    output
        - list of eigenvalues, sorted or not
        - list of corresponding eigenvectors
    '''
    import numpy as np
    eps = list()
    MOs = list()
    for mo in eigenvectors:
        eps.extend(mo[0] for i in range(mo[1]))
        for eigvc in mo[2]:
            MOs.append(eigvc.normalized())
    if (sort):
        sortindex=[]
        for i,j in sorted(enumerate(eps), key=lambda j: j[1]):
            sortindex.append(i)
        eps = sorted(eps)
    
        MOs_sorted=[]
        for i, mo in enumerate(MOs):
            MOs_sorted.append(MOs[sortindex[i]])
        return eps,MOs_sorted
    else:
        return eps,MOs
