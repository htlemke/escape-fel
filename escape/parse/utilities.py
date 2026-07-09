import h5py
from pathlib import Path


def findItemnamesGroups(
    toplevel, item_names=[], get_full_name=False, group_name_contains=None
):
    itemGroups = {}

    def find_datasets(item, item_h):
        if not isinstance(item_h, h5py.Group):
            return
        elif (group_name_contains is not None) and (
            group_name_contains not in item_h.name
        ):
            return
        else:
            if set(item_names).issubset(item_h.keys()):
                if get_full_name:
                    itemGroups[item_h.name] = [item_h[name] for name in item_names]
                else:
                    itemGroups[Path(item).name] = [item_h[name] for name in item_names]

    toplevel.visititems(find_datasets)
    return itemGroups


def findItemnamesGroups_v03(
    toplevel, item_names, known_dead_ends=None, group_name_contains=None
):
    """Delta-aware variant of findItemnamesGroups.

    ``visititems`` (used by findItemnamesGroups) cannot be pruned: the HDF5
    library commits to visiting every descendant of every group before our
    callback ever runs, so a group we don't care about still costs exactly
    as much as one we do. This version walks the tree manually instead, so
    it can skip descending into a group entirely when *known_dead_ends*
    (full HDF5 paths of groups previously found, on an earlier file of the
    same run, to contain no matching channel anywhere below them) says
    there is nothing there to find.

    Groups that *do* match are still detected with a single, cheap
    ``.keys()`` call each time (never blindly trusted) — verifying a leaf
    channel group costs essentially nothing beyond opening it, so there is
    no reason to skip that check. Only the expensive part — recursing
    into subtrees that structurally never contain a channel — is elided,
    and only when the caller opts in by supplying *known_dead_ends*.

    Parameters
    ----------
    known_dead_ends : set[str], optional
        Full group paths (``Group.name``) to skip without re-verification.
        Pass ``None``/empty to fall back to a full, unpruned walk.

    Returns
    -------
    tuple[dict, set, set]
        ``(itemGroups, new_dead_ends, seen_channel_paths)`` — the matches
        found in this file, the group paths found to be dead ends *in this
        file* (for the caller to merge into a shared registry so later
        files can skip them too), and the full group paths of channels
        seen this call.
    """
    known_dead_ends = known_dead_ends or set()
    itemGroups = {}
    new_dead_ends = set()
    seen_channel_paths = set()

    def visit(group):
        name = group.name
        if (group_name_contains is not None) and (group_name_contains not in name):
            return False
        if name in known_dead_ends:
            return False  # trusted: nothing below here on the last pass
        keys = list(group.keys())
        if set(item_names).issubset(keys):
            itemGroups[Path(name).name] = [group[n] for n in item_names]
            seen_channel_paths.add(name)
            return True  # leaf channel group: nothing more to find below
        found_below = False
        for key in keys:
            try:
                child = group[key]
            except Exception:
                continue
            if isinstance(child, h5py.Group) and visit(child):
                found_below = True
        if not found_below:
            new_dead_ends.add(name)
        return found_below

    visit(toplevel)
    return itemGroups, new_dead_ends, seen_channel_paths


def filterTypes(listOfItemLists, types=[h5py.Dataset] * 2):
    n_good = []
    for n, itemlist in enumerate(listOfItemLists):
        if all([type(ti) is tt for ti, tt in zip(itemlist, types)]):
            n_good.append(itemlist)
    n_good.sort()
    n_good.reverse()
    for n in n_good:
        listOfItemLists.pop(n)
