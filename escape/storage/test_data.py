from ..stream.testStream import TestData
from .storage import Array
import numpy as np
from dask import array as da


def get_test_data(N_pulses=1e4, as_array=True, as_da=True):
    td = TestData()
    d = {
        key: np.asarray(tl)
        for key, tl in zip(
            td.generateData(0).keys(),
            zip(*[list(td.generateData(n).values()) for n in range(int(N_pulses))]),
        )
    }
    if as_array:
        pulse_id = d.pop("pulse_id")
        if as_da:
            d = {
                key: Array(
                    data=da.from_array(td), index=pulse_id, step_lengths=[len(td)]
                )
                for key, td in d.items()
            }
        else:
            d = {
                key: Array(data=td, index=pulse_id, step_lengths=[len(td)])
                for key, td in d.items()
            }
    return d
