import pickle


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file
