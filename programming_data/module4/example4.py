# Some Simple Algorithms and Data Structures (pg.253)
def search(L, e):
    """ Asssumes L is a list.
    Returns True if e is in L and False otherwise"""
    for i in range(len(L)):
        if L[i] == e:
            return True
    return False

