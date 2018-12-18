import os
import pickle

import numpy
from keras.layers import Embedding, LSTM, TimeDistributed, Dense
from keras.models import Sequential, load_model
from sklearn_crfsuite import metrics


def _save_metrics(
        output_path,
        classification_report,
        sequence_accuracy,
        precision,
        recall
):
    with open(os.path.join(output_path, 'metrics.txt'), 'w') as out:
        out.write(
            "### Classification Report ###\n{classification_report}\n\n"
            "### Sequence Accuracy Score ###\n{sequence_accuracy}\n\n"
            "### Weighted Precision Score ###\n{precision}\n\n"
            "### Weighted Recall Score ###\n{recall}\n".format(
                classification_report=classification_report,
                sequence_accuracy=str(sequence_accuracy),
                precision=str(precision),
                recall=str(recall)
            )
        )


class NERModelTrainer(object):
    def __init__(self, data_reader):
        self._data_reader = data_reader
        self._model = None

    def train(
            self,
            epochs=10,
            lstm_units=100,
            batch=20,
            dropout=0.0,
            word_embedding_size=200
    ):
        self._model = Sequential()
        self._model.add(
            Embedding(
                self._data_reader.n_vocabulary,
                word_embedding_size,
                input_length=self._data_reader.max_train_sentence_length,
                mask_zero=True
            )
        )
        self._model.add(
            LSTM(
                units=lstm_units,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=True,
                input_shape=(
                    self._data_reader.max_train_sentence_length,
                    word_embedding_size
                )
            )
        )
        self._model.add(
            TimeDistributed(
                Dense(self._data_reader.n_classes, activation='softmax'),
                input_shape=(
                    self._data_reader.max_train_sentence_length,
                    word_embedding_size
                )
            )
        )
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer='adam'
        )

        print(self._model.summary())

        train_history = self._model.fit(
            x=self._data_reader.train_X,
            y=self._data_reader.train_y,
            epochs=epochs,
            batch_size=batch,
            validation_data=(
                self._data_reader.validation_X,
                self._data_reader.validation_y
            )
        )

        loss = train_history.history['loss']
        print("History Loss")
        print(loss)
        val_loss = train_history.history['val_loss']
        print("History Validation Loss")
        print(val_loss)

    def persist(self, output_path):
        def _save_dict(input_dict, output_path, file_name):
            with open(os.path.join(output_path, file_name), 'wb') as _out:
                pickle.dump(input_dict, _out)

        self._model.save(os.path.join(output_path, 'model.h5'))

        _save_dict(self._data_reader.w2idx, output_path, 'w2idx.pkl')
        _save_dict(
            self._data_reader.idx2label, output_path, 'idx2label.pkl'
        )

    def evaluate(self, output_path):
        loss = self._model.evaluate(
            self._data_reader.test_X, self._data_reader.test_y
        )
        print('Loss is: %f' % loss)

        all_predicted_labels = []
        all_true_labels = []
        for i, _test_instance in enumerate(self._data_reader.test_X):
            test_prediction = self._model.predict(_test_instance.reshape(
                1, self._data_reader.max_train_sentence_length)
            )[0]

            predicted_labels, true_labels = [], []
            for encoded_true_label_array, encoded_test_label_array in zip(
                    self._data_reader.test_y[i], test_prediction
            ):
                contains_all_zeros = not numpy.any(encoded_true_label_array)
                if not contains_all_zeros:
                    predicted_labels.append(
                        self._data_reader.decode_single_label(
                            encoded_test_label_array
                        )
                    )
                    true_labels.append(
                        self._data_reader.decode_single_label(
                            encoded_true_label_array
                        )
                    )

            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)

        classification_report = metrics.flat_classification_report(
            all_true_labels,
            all_predicted_labels,
            labels=self._data_reader.labels
        )

        sequence_accuracy = metrics.sequence_accuracy_score(
            all_true_labels,
            all_predicted_labels
        )

        precision = metrics.flat_precision_score(
            all_true_labels, all_predicted_labels, average='weighted'
        )

        recall = metrics.flat_recall_score(
            all_true_labels,
            all_predicted_labels,
            average='weighted'
        )

        _save_metrics(
            output_path=output_path,
            classification_report=classification_report,
            sequence_accuracy=sequence_accuracy,
            precision=precision,
            recall=recall
        )

        return classification_report, sequence_accuracy, precision, recall
