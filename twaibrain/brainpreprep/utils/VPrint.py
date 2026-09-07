from typing import Union

class VPrint():
    def __init__(self, verbose : Union[bool, int] = False):
        if (not isinstance(verbose, bool)) or (not isinstance(verbose, int)):
            raise ValueError("verbose flag must be a bool or an int")
        
        self.verbose = verbose
    
    def __call__(self, *args, verbose_index=0, **kwargs):
        if isinstance(self.verbose, bool) and self.verbose == False:
            return
        
        elif isinstance(self.verbose, bool) and self.verbose == True:
            print(*args, **kwargs)
        
        if verbose_index > self.verbose:
            print(*args, **kwargs)
