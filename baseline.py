from preprocessor import Preprocessor

import pycrfsuite
import nltk
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"


class CNFModel:

    def __init__(self, train_directory=TRAIN_DIRECTORY, test_directory=TEST_DIRECTORY):
        self.preprocessed = Preprocessor(train_directory, test_directory)
        self.train_data = self.preprocessed.train_data
        self.test_data =  self.preprocessed.test_data

    def word2features(self, sentence, i):
        current_word = sentence[i][0]
        current_pos = sentence[i][1]

        # Features relevant to the CURRENT token in sentence
        features = [
            'bias',
            'word.lower=' + current_word.lower(),
            'word[-3:]=' + current_word[-3:],
            'word[-2:]=' + current_word[-2:],
            'word.istitle=%s' % current_word.istitle(),
            'word.isdigit=%s' % current_word.isdigit(),
            'word.isupper=%s' % current_word.isupper(),
            'postag=' + current_pos,
            'postag[:2]=' + current_pos[:2],
        ]

        # Features for words that are not at the beginning of a sentence
        if i > 0:
            prev_word = sentence[i - 1][0]
            previous_pos = sentence[i - 1][1]
            features.extend([
                '-1:word.lower=' + prev_word.lower(),
                '-1:word.istitle=%s' % prev_word.istitle(),
                '-1:word.isdigit=%s' % prev_word.isdigit(),
                '-1:word.isupper=%s' % prev_word.isupper(),
                '-1:postag=' + previous_pos,
                '-1:postag[:2]=' + previous_pos[:2],
            ])
        else:
            features.append('BOS')

        # Features for words that are not at the end of a sentence
        if i < len(sentence) - 1:
            next_word = sentence[i + 1][0]
            next_pos = sentence[i + 1][1]
            features.extend([
                '+1:word.lower=' + next_word.lower(),
                '+1:word.istitle=%s' % next_word.istitle(),
                '+1:word.isdigit=%s' % next_word.isdigit(),
                '+1:word.isupper=%s' % next_word.isupper(),
                '+1:postag=' + next_pos,
                '+1:postag[:2]=' + next_pos[:2],
            ])
        else:
            features.append('EOS')

        return features

    def extract_features(self, sentence):
        return [self.word2features(sentence, i) for i in range(len(sentence))]

    def get_label(self, sentence):
        return [label for (token, pos, label) in sentence]

    def train_model(self):
        X_train = [self.extract_features(sentence) for sentence in self.train_data]
        y_train = [self.get_label(sentence) for sentence in self.train_data]

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            # coefficient for L1 penalty
            'c1': 0.1,

            # coefficient for L2 penalty
            'c2': 0.01,

            # maximum number of iterations
            'max_iterations': 200,

            # whether to include transitions that
            # are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train('crf.model')

    def predict(self):
        X_test = [self.extract_features(sentence) for sentence in self.test_data]
        y_test = [self.get_label(sentence) for sentence in self.test_data]

        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        y_pred = [tagger.tag(xseq) for xseq in X_test]
        labels = {"B": 0, 'I': 1, 'O': 2}  # row indexes for position of labels in the classification matrix

        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in y_test for tag in row])
        print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))


if __name__ == "__main__":
    model = CNFModel()
    model.train_model()
    model.predict()