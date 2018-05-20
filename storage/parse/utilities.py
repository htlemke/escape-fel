import h5py
from pathlib import Path

def findItemnamesGroups(toplevel,item_names=[]):
    itemGroups = {}
    def find_datasets(item,item_h):
        if not isinstance(item_h,h5py.Group):
            return
        else:
            if set(item_names).issubset(item_h.keys()):
                itemGroups[Path(item).name] = [item_h[name] for name in item_names]
    toplevel.visititems(find_datasets)
    return itemGroups

def filterTypes(listOfItemLists,types=[h5py.Dataset]*2):
    n_good = []
    for n,itemlist in enumerate(listOfItemLists):
        if all([type(ti) is tt \
                for ti,tt in zip(itemlist,types)]):
            n_good.append(itemlist)
    n_good.sort()
    n_good.reverse()
    for n in n_good:
        listOfItemLists.pop(n)


