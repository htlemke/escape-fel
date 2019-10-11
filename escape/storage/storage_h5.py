import h5py



class 

class Store:
    def __init__(self,parent_h5_group,name):
        self.parent_h5_group = parent_h5_group
        self.group = self.parent_h5_group.require_group(name)
        self.name = name



        
        
