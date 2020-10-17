import pickle
from itertools import chain


def save_obj(obj, d, name):
    with open(d + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(d, name):
    with open(d + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def append_obj(obj, f):
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


def flatten(items):
    return chain.from_iterable(items)
