import numpy as np
import json
with open('test.json', 'r') as f:
    reader = f.read()
    content = json.loads(reader)
    y = np.asarray(content['y'])
    x = np.asarray(content['x'])
    for j in x:
        print(len(j))
    print(x.shape)
    print(y.shape)