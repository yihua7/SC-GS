import pickle


def save_obj(path, obj):
    file = open(path, 'wb')
    obj_str = pickle.dumps(obj)
    file.write(obj_str)
    file.close()


def load_obj(path):
    file = open(path, 'rb')
    obj = pickle.loads(file.read())
    file.close()
    return obj
