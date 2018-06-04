import pickle
import os
import errno
from tqdm import tqdm


def write_all_objects(tgt_dir, objs):
    print("Saving to pickle...")
    try:
        os.makedirs(tgt_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(tgt_dir):
            pass
        else:
            raise

    for index, obj in enumerate(tqdm(objs)):
        filepath = os.path.join(tgt_dir, "pickle-{}.pkl".format(index))
        pickle.dump(obj, open(filepath, 'wb'))


def read_all_objects(tgt_dir):
    objs = []

    print("Loading from pickle...")
    if os.path.isdir(tgt_dir):
        for f in tqdm(get_all_files(tgt_dir)):
            if os.path.isfile(f):
                objs.append(pickle.load(open(f, 'rb')))
    else:
        raise Exception('Directory not found.')

    return tuple(objs)


def get_all_files(tgt_dir):
    all_files = [f for f in os.listdir(tgt_dir) if os.path.isfile(os.path.join(tgt_dir, f))]
    return all_files

# Example:
# write_all_objects('./pt1', (a, b))
# a, b = read_all_objects('./pt1')
