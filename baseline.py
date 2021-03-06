from preprocessor import Preprocessor
from preprocessor2 import Preprocessor2

import pycrfsuite
import nltk
import numpy as np
import sklearn_crfsuite

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


RESTAURANT_TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
RESTAURANT_TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"

LAPTOP_TRAIN_DIRECTORY = "data/train_data/Laptop_Train_v2.xml"
LAPTOP_TEST_DIRECTORY = "data/test_data/Laptops_Test_Truth.xml"

class CNFBaselineModel:

    '''
        Change desired directory here to test on restaurant/laptop
    '''
    def __init__(self, train_directory=LAPTOP_TRAIN_DIRECTORY, test_directory=LAPTOP_TEST_DIRECTORY):
        self.preprocessed = Preprocessor2(train_directory, test_directory)
        self.train_data = self.preprocessed.train_data
        self.test_data =  self.preprocessed.test_data
        self.predicted = []
        self.truth = []

    def word2features(self, sentence, i):
        current_word = sentence[i][0]
        current_pos = sentence[i][1]

        # Features relevant to the CURRENT token in sentence
        features = {
            'bias': 1.0,
            'word.lower()': current_word.lower(),
            'word[-3:]': current_word[-3:],
            'word[-2:]': current_word[-2:],
            'word.istitle': current_word.istitle(),
            'word.isdigit': current_word.isdigit(),
            'word.isupper': current_word.isupper(),
            'postag': current_pos,
            'postag[:2]': current_pos[:2]
        }

        return features

    def extract_features(self, sentence):
        return [self.word2features(sentence, i) for i in range(len(sentence))]

    def get_label(self, sentence):
        return [label for (token, pos, dep, ner, label) in sentence]

    def train_model(self):
        print("Training Model...")
        X = [self.extract_features(sentence) for sentence in self.train_data]
        y = [self.get_label(sentence) for sentence in self.train_data]

        print('Generated Training Features + Labels...')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150, random_state=2)

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            # coefficient for L1 penalty
            'c1': 0.03655,

            # coefficient for L2 penalty
            'c2': 0.09558,

            # maximum number of iterations
            'max_iterations': 200,

            # whether to include transitions that
            # are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train('crf.model')
        print("Finished training model")

        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        y_pred = [tagger.tag(xseq) for xseq in X_test]
        labels = {"B": 0, 'I': 1, 'O': 2}  # row indexes for position of labels in the classification matrix

        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in y_test for tag in row])

        print("------------------------------------------------ VALIDATION RESULTS ------------------------------------------------")
        # print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))
        new_y_test = list(map(lambda x: list(map(self.change_BIO, x)), y_test))
        new_y_pred = list(map(lambda x: list(map(self.change_BIO, x)), y_pred))

        print(self.get_metrics(new_y_test, new_y_pred, b=1))  ## printing new metric to calculate F1

    def predict(self):
        print("Predicting Model...")
        X_test = [self.extract_features(sentence) for sentence in self.test_data]
        y_test = [self.get_label(sentence) for sentence in self.test_data]

        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        y_pred = [tagger.tag(xseq) for xseq in X_test]

        print('Generated Predicted Features + Labels...')
        labels = {"B": 0, 'I': 1, 'O': 2}  # row indexes for position of labels in the classification matrix

        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in y_test for tag in row])

        print("------------------------------------------------ SEMEVAL RESULTS ------------------------------------------------")
        # print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))

        new_y_test = list(map(lambda x: list(map(self.change_BIO, x)), y_test))
        new_y_pred = list(map(lambda x: list(map(self.change_BIO, x)), y_pred))

        print(self.get_metrics(new_y_test, new_y_pred, b=1))  ## printing new metric to calculate F1

    def evaluate_SemEVAL(self):
        X = [self.extract_features(sentence) for sentence in self.train_data]
        y = [self.get_label(sentence) for sentence in self.train_data]

        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=150, random_state=2)

        print('Generated Training Features + Labels...')

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.03655,
            c2=0.09558,
            max_iterations=200,
            all_possible_transitions=True,
        )

        crf.fit(X_train, y_train)

        print('Fitted model...')

        X_test = [self.extract_features(sentence) for sentence in self.test_data]
        y_test = [self.get_label(sentence) for sentence in self.test_data]

        print('Generated Test Features + Labels...')

        y_pred = crf.predict(X_test)

        self.predicted = y_pred
        self.truth = y_test

        new_y_test = list(map(lambda x: list(map(self.change_BIO, x)), y_test))
        new_y_pred = list(map(lambda x: list(map(self.change_BIO, x)), y_pred))

        print("F1 score = %s" % self.get_metrics(new_y_test, new_y_pred, b=1))  ## printing new metric to calculate F1

    '''
        Helper Function to Change Bio label to numerical values. "O" = 0, "B" = 1, "I" = 2
    '''
    def change_BIO(self, label):
        if label == 'O':
            return 0
        elif label == 'B':
            return 1
        else:
            return 2

    '''
        Helper Function for get_metrics() to calculate new F1 metric measure
    '''
    def get_term_pos(self, labels):
        start, end = 0, 0
        tag_on = False
        terms = []
        labels = np.append(labels, [0])
        for i, label in enumerate(labels):
            if label == 1 and not tag_on:
                tag_on = True
                start = i
            if tag_on and labels[i + 1] != 2:
                tag_on = False
                end = i
                terms.append((start, end))
        return terms

    '''
        Function to calculate new metric to evaluate our model instead of classification report.
    '''
    def get_metrics(self, test_y, pred_y, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(len(test_y)):
            cor = self.get_term_pos(test_y[i])
            pre = self.get_term_pos(pred_y[i])
            common += len([a for a in pre if a in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.

        text = "Precision = {pre}, Recall = {rec}, F1 = {f1_score}," \
               " Common = {com}, Retrieved = {ret}, Relevant = {rel}".format(pre=p, rec=r, f1_score=f1, com=common,
                                                                             ret=retrieved, rel=relevant)
        return text

    def find_different_predictions(self):

        token_wrong = 0

        wrong_sentences = []

        printed = []

        for i in range(len(self.predicted)):
            predicted_sentence_labels = self.predicted[i]
            truth_sentence_labels = self.truth[i]
            original_sentence = [text for (text, pos, dep, ner, bio) in self.test_data[i]]

            sentence_wrong = False
            for j in range(len(predicted_sentence_labels)):
                predicted_token = predicted_sentence_labels[j]
                truth_token = truth_sentence_labels[j]

                if ((truth_token == 'B') or truth_token == 'I') and (predicted_token == 'O'):
                    token_wrong += 1

                    full_sentence = ' '.join(original_sentence)
                    if full_sentence not in printed:
                        print("Tokens = %s" % original_sentence)
                        print("Truth Labels     = %s" % truth_sentence_labels)
                        print("Predicted Labels = %s" % predicted_sentence_labels)
                        print("-------------------------------------------------------------------------------------------")
                        printed.append(full_sentence)


        print("Without BG Num Tokens Wrong = %s" % token_wrong)

        return wrong_sentences


if __name__ == "__main__":
    model = CNFBaselineModel()
    model.evaluate_SemEVAL()
