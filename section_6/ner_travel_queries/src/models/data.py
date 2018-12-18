try:
    import cPickle as pickle
except:
    import pickle
import os


def _load_data(input_path, fold_num):
    f = 'atis.fold%s.pkl' % fold_num
    with open(os.path.join(input_path, f), 'rb') as in_file:
        train_set, valid_set, test_set, dicts = pickle.load(in_file, encoding='latin1')

    return train_set, valid_set, test_set, dicts


def get_raw_data(input_path):
    all_train_set = [[], [], []]
    all_valid_set = [[], [], []]
    all_test_set = [[], [], []]
    all_dicts = {'labels2idx': {}, 'tables2idx': {}, 'words2idx': {}}
    for i in range(5):
        train_set, valid_set, test_set, dicts = _load_data(input_path, i)
        all_train_set[0].extend(train_set[0])
        all_train_set[1].extend(train_set[1])
        all_train_set[2].extend(train_set[2])

        all_valid_set[0].extend(valid_set[0])
        all_valid_set[1].extend(valid_set[1])
        all_valid_set[2].extend(valid_set[2])

        all_test_set[0].extend(test_set[0])
        all_test_set[1].extend(test_set[1])
        all_test_set[2].extend(test_set[2])

        all_dicts['labels2idx'].update(dicts['labels2idx'])
        all_dicts['tables2idx'].update(dicts['tables2idx'])
        all_dicts['words2idx'].update(dicts['words2idx'])

    return all_train_set, all_valid_set, all_test_set, all_dicts
