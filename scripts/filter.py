import pandas as pd
import time

def filter(path):
    start_time = time.time()
    file = open(path, 'r')
    data = pd.read_csv(file, sep=';', names=['time', 'part', 'x', 'y', 'z'])

    parts = set(data['part'])
    p_col = len(parts)
    # result = [[0.0, 0.0, 0.0, 0.0, 0.0,] for i in range(20)]
    result = []

    for t in set(data['time']):
        for p in parts:
            if not data[(data['time'] == t) & (data['part'] == p)].empty:
                values = data[(data['time'] == t) & (data['part'] == p)].values[0]
                result.append([t, p, values[2], values[3], values[4]])
            elif len(result) > p_col:
                result.append([t, p, result[-(p_col - 1)][2], result[-(p_col - 1)][3], result[-(p_col - 1)][4]])
            else:
                result.append([t, p, 0.0, 0.0, 0.0])
    print(time.time() - start_time)
    return pd.DataFrame(result, columns=['time', 'part', 'x', 'y', 'z'])
