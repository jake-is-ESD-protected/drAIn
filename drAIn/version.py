
def get_version(reduced: bool=True):
    v = "0.0.1"
    if not reduced:
        v = "drAIn version " + v
    return v