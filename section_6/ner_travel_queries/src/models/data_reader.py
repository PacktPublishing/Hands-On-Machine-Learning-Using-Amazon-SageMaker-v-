import numpy
from keras.preprocessing.sequence import pad_sequences

from src.models.data import get_raw_data


def _encode(input_number, n_classes):
    new_array = numpy.zeros((n_classes,), dtype=numpy.int)
    new_array[input_number] = 1

    return new_array


def _encode_labels(
        labels,
        n_classes,
        max_train_sentence_length,
        nil_label_numerical_value
):
    new_labels = []
    for _seq_labels in labels:
        encoded_seq_labels = []
        for _label in _seq_labels:
            encoded_seq_labels.append(_encode(_label, n_classes))
        new_labels.append(encoded_seq_labels)

    return pad_sequences(
        new_labels,
        maxlen=max_train_sentence_length,
        dtype=numpy.float64,
        value=nil_label_numerical_value
    )


class DataReader(object):
    def __init__(self, input_path):
        train_set, valid_set, test_set, dicts = get_raw_data(input_path)
        train_x, _, train_label = train_set
        valid_x, _, valid_label = valid_set
        test_x, _, test_label = test_set
        self.w2idx, _, self._labels2idx = dicts['words2idx'], \
            dicts['tables2idx'], dicts['labels2idx']

        self.n_classes = len(self._labels2idx)

        self._idx2word = dict((v, k) for k, v in self.w2idx.items())
        word_at_index_0 = self._idx2word[0]
        max_idx2word = max(self._idx2word.keys())
        self._idx2word[max_idx2word + 1] = word_at_index_0
        self.w2idx[word_at_index_0] = max_idx2word + 1

        self.idx2label = dict((v, k) for k, v in self._labels2idx.items())

        self.n_vocabulary = len(self.w2idx) + 1  # because of the padding
        self.max_train_sentence_length = 50

        self.pad_value = 0

        self.labels = sorted(
            self._labels2idx.keys(),
            key=lambda name: (name[1:], name[0])
        )

        self.train_X = pad_sequences(
            train_x,
            maxlen=self.max_train_sentence_length,
            dtype=numpy.float64,
            value=self.pad_value
        )

        self.validation_X = pad_sequences(
            valid_x,
            maxlen=self.max_train_sentence_length,
            dtype=numpy.float64,
            value=self.pad_value
        )

        self.test_X = pad_sequences(
            test_x,
            maxlen=self.max_train_sentence_length,
            dtype=numpy.float64,
            value=self.pad_value
        )

        self.train_y = _encode_labels(
            train_label,
            self.n_classes,
            self.max_train_sentence_length,
            self.pad_value
        )
        self.validation_y = _encode_labels(
            valid_label,
            self.n_classes,
            self.max_train_sentence_length,
            self.pad_value
        )
        self.test_y = _encode_labels(
            test_label,
            self.n_classes,
            self.max_train_sentence_length,
            self.pad_value
        )

    def encode_sentence(self, sentence_tokens, skip_unknown_words=False):
        X = []
        outer_X = []
        for word in sentence_tokens:
            if word in self.w2idx:
                X.append(self.w2idx[word])

            elif skip_unknown_words:
                continue

            else:
                X.append(0)

        outer_X.append(X)
        return pad_sequences(
            outer_X,
            maxlen=self.max_train_sentence_length,
            dtype=numpy.float64
        )

    def decode_single_label(self, encoded_label):
        return self.idx2label[numpy.argmax(encoded_label)]

    def decode_labels(self, encoded_labels):
        return [
            self.decode_single_label(class_prs)
            for class_prs in encoded_labels
        ]
