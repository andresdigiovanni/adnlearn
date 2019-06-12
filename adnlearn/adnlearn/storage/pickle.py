import pickle

def save_pickle(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def load_pickle(filename):
    file = open(filename,'rb')
    obj = pickle.load(file)
    file.close()

    return obj
