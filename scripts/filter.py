import pandas as pd
import time

# К1_20181207_110935.csv

def filter(path):
    start_time = time.time()
    file = open(path, 'r')
    data = pd.read_csv(file, sep=';', names=['time', 'part', 'x', 'y', 'z'])

    parts = set(data['part'])
    times = sorted(set(data['time']))
    p_col = len(parts)
    # result = [[0.0, 0.0, 0.0, 0.0, 0.0,] for i in range(20)]
    result = []

    for t in times:
        for p in parts:
            if not data[(data['time'] == t) & (data['part'] == p)].empty:
                # print(f"{data[(data['time'] == t) & (data['part'] == p)]}\n-----------------------", file=open('logi.log', 'a'))
                values = data[(data['time'] == t) & (data['part'] == p)].values[0]
                result.append([t, p, values[2], values[3], values[4]])
            elif len(result) > p_col:
                # print(f"{t}|{p}|{result[-p_col]}", file=open('logi.log', 'a'))
                result.append([t, p, result[-p_col][2], result[-p_col][3], result[-p_col][4]])
            else:
                result.append([t, p, 0.0, 0.0, 0.0])
    print(time.time() - start_time)
    return pd.DataFrame(result, columns=['time', 'part', 'x', 'y', 'z'])
