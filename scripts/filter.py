import pandas as pd
import time

# К1_20181207_110935.csv

def filter(path):
    start_time = time.time()
    file = open(path, 'r')
    data = pd.read_csv(file, sep=';', names=['time', 'part', 'x', 'y', 'z'])

    parts = list(set(data['part']))
    times = sorted(set(data['time']))
    p_col = len(parts)
    result = [[0.0, 0.0, 0.0, 0.0, 0.0,] for i in range(20)]
    # result = []

    for t in times:
        for p in parts:
            row = data[(data['time'] == t) & (data['part'] == p)]
            if not row.empty:
                # print(f"{data[(data['time'] == t) & (data['part'] == p)]}\n-----------------------")
                values = row.values[0]
                result.append([values[0], values[1], values[2], values[3], values[4]])
            else:
                # print(f"{t}|{p}|{result[-p_col]}", file=open('logi.log', 'a'))
                result.append([t, p, result[-p_col][2], result[-p_col][3], result[-p_col][4]])
    result = result[p_col:]
    print(time.time() - start_time)
    return pd.DataFrame(result, columns=['time', 'part', 'x', 'y', 'z'])
