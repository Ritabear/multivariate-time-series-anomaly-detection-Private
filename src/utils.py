# ref. https://cs230.stanford.edu/blog/hyperparameters/

import json

def get_hyperparameter():
    f = open('params.json', 'r')
    data = json.load(f)
    #print(data['model_1'][1]['type'])
    f.close()
    return data