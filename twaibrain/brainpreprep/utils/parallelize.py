import multiprocessing

def parallelize_func(func, input_sets):
    with multiprocessing.Pool() as pool:
        results = poolmap(async_func, input_sets)

    return results
