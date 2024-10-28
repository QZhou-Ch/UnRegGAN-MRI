import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def ptqdm(function, iterable, processes, zipped=False, chunksize=1, desc=None, disable=False, **kwargs):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of iterables and arguments.
    Multiple iterables can be packed into a tuple and passed to the 'iterable argument'. The iterables must be the first arguments in the function that is run in parallel.
    Results are always ordered and the performance is the same as of Pool.map.
    同时运行一个函数和一个tqdm进度条以及任意数量的可迭代变量和参数。
    可以将多个迭代器打包到一个元组中，并将其传递给“可迭代参数”。迭代量必须是并行运行的函数中的第一个参数。
    结果始终是有序的，并且性能与Pool.map相同。
    :param function: The function that should be parallelized.需迭代的函数
    :param iterable: The iterable passed to the function.可迭代对象
    :param processes: The number of processes used for the parallelization.并行运算使用的核心数
    :param zipped: If multiple iterables are packed into a tuple. The iterables will be unpacked and passed as separate arguments to the function.
    :param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
    :param desc: The description displayed by tqdm in the progress bar. Tqdm在进度栏中显示的说明
    :param disable: Disables the tqdm progress bar.
    :param kwargs: Any additional arguments that should be passed to the function. 应传递给函数的任何其他参数
    """
    if kwargs:
        function_wrapper = partial(wrapper, function=function, zipped=zipped, **kwargs)
    else:
        function_wrapper = partial(wrapper, function=function, zipped=zipped)

    if zipped:
        length = len(iterable[0])
        iterable = zip(*iterable)
    else:
        length = len(iterable)

    results = [None] * length
    
    with Pool(processes=processes) as p:
        with tqdm(desc=desc, total=length, disable=disable) as pbar:
            for i, result in p.imap_unordered(function_wrapper, enumerate(iterable), chunksize=chunksize):
                results[i] = result
                pbar.update()
    return results


def wrapper(enum_iterable, function, zipped, **kwargs):
    i = enum_iterable[0]
    if zipped:
        result = function(*enum_iterable[1], **kwargs)
    else:
        result = function(enum_iterable[1], **kwargs)
    return i, result

