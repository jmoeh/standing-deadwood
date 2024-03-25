from joblib import Parallel, delayed

from typing import Tuple, Iterable, Set, List, Callable
import contextlib
import joblib
from tqdm import tqdm
import numpy as np


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    # credits:
    # https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def paral(
    function: Callable,
    iters: List[Iterable],
    num_cores=-1,
    progress_bar=True,
    backend="loky",
):
    """compute function parallel with arguments in iters.
    function(iters[0][0],iters[0][1],...)"""

    with tqdm_joblib(
        tqdm(
            desc=function.__name__,
            unit="jobs",
            dynamic_ncols=True,
            total=len(iters[0]),
            disable=not progress_bar,
        ),
    ) as progress_bar:
        # backend can be loky or threading (or maybe something else)
        return Parallel(n_jobs=num_cores, batch_size=1, backend=backend)(
            delayed(function)(*its) for its in zip(*iters)
        )
