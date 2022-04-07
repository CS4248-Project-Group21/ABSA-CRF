from preprocessor2 import Preprocessor2

import pycrfsuite
import numpy as np
from collections import Counter


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


RESTAURANT_TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
RESTAURANT_TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"

LAPTOP_TRAIN_DIRECTORY = "data/train_data/Laptop_Train_v2.xml"
LAPTOP_TEST_DIRECTORY = "data/test_data/Laptops_Test_Truth.xml"

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('vader_lexicon')
class CNFModel:

    '''
        Change desired directory here to test on restaurant/laptop
    '''
    def __init__(self, train_directory=LAPTOP_TRAIN_DIRECTORY, test_directory=LAPTOP_TEST_DIRECTORY):
        self.preprocessed = Preprocessor2(train_directory, test_directory)
        self.train_data = self.preprocessed.train_data
        self.test_data = self.preprocessed.test_data
        self.corpus_freq = self.build_corpus_freq()


    '''
        Customise the type of model by commenting out features below
        E.g if Using Word Shape + POS + Bigram Model, comment out all other features
    '''
    # sentence = [(w1, pos, dep, NER, bio_label), (w2, pos, dep, NER, bio_label),...,(wn, pos, dep, NER, bio_label)]
    def extract_features(self, sentence):

        sentiment_analyzer = SentimentIntensityAnalyzer()

        all_features = []

        for i in range(len(sentence)):
            current_word = sentence[i][0]
            current_pos = sentence[i][1]
            current_dep = sentence[i][2]
            current_ner = sentence[i][3]
            polarity_score = sentiment_analyzer.polarity_scores(current_word)
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()

            features = {
                'bias': 1.0,
                'word.lower()': current_word.lower(),
                'word[-3:]': current_word[-3:],
                'word[-2:]': current_word[-2:],
                'word.istitle': current_word.istitle(),
                'word.isdigit': current_word.isdigit(),
                'word.isupper': current_word.isupper(),
                'postag': current_pos,
                'postag[:2]': current_pos[:2],
                'word.lemmatized': lemmatizer.lemmatize(current_word),
                'word.stemmed': stemmer.stem(current_word),
                'word.positivityscore': polarity_score['pos'],
                'word.negativityscore': polarity_score['neg'],
                'word.isStopWord': self.isStopword(current_word),
                'word.isFrequent': self.isTokenFrequent(current_word),
                'word.is_dobj': current_dep == 'dobj',
                'word.is_iobj': current_dep == 'iobj',
                'word.is_nsubj': current_dep == 'nsubj',
                'word.NER': current_ner
            }

            # Features for words that are not at the beginning of a sentence
            if i > 0:
                prev_word = sentence[i - 1][0]
                previous_pos = sentence[i - 1][1]
                features.update({
                    '-1:word.lower=': prev_word.lower(),
                    '-1:word.istitle=%s': prev_word.istitle(),
                    '-1:word.isdigit=%s': prev_word.isdigit(),
                    '-1:word.isupper=%s': prev_word.isupper(),
                    '-1:postag=': previous_pos,
                    '-1:postag[:2]=': previous_pos[:2],
                })
            else:
                features['BOS'] = True

            # Features for words that are not at the end of a sentence
            if i < len(sentence) - 1:
                next_word = sentence[i + 1][0]
                next_pos = sentence[i + 1][1]
                features.update({
                    '+1:word.lower=': next_word.lower(),
                    '+1:word.istitle=%s': next_word.istitle(),
                    '+1:word.isdigit=%s': next_word.isdigit(),
                    '+1:word.isupper=%s': next_word.isupper(),
                    '+1:postag=': next_pos,
                    '+1:postag[:2]=': next_pos[:2],
                })
            else:
                features['EOS'] = True

            all_features.append(features)

        return all_features

    '''
        Helper function to build a freq count of all the words in training/test corpus
    '''
    def build_corpus_freq(self):
        freq_lst = {}
        for sentence in self.train_data:
            for word, pos, dep, ner, label in sentence:
                if word not in freq_lst:
                    freq_lst[word] = 0
                freq_lst[word] += 1

        return freq_lst

    def isTokenFrequent(self, token):
        if token in self.corpus_freq:
            return self.corpus_freq[token] > 4
        return False


    def get_label(self, sentence):
        return [label for (token, pos, dep, ner, label) in sentence]

    def isStopword(self, token):
        if token in set(stopwords.words('english')):
            return True
        return False

    def train_model(self):
        print("Training Model...")
        X = [self.extract_features(sentence) for sentence in self.train_data]
        y = [self.get_label(sentence) for sentence in self.train_data]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150, random_state=2)

        print('Generated Training Features + Labels...')
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
        print("Finished training model...")

        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        y_pred = [tagger.tag(xseq) for xseq in X_test]
        labels = {"B": 0, 'I': 1, 'O': 2}  # row indexes for position of labels in the classification matrix

        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in y_test for tag in row])

        print("------------------------------------------------ VALIDATION RESULTS ------------------------------------------------")
        print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))
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
        print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))

        new_y_test = list(map(lambda x: list(map(self.change_BIO, x)), y_test))
        new_y_pred = list(map(lambda x: list(map(self.change_BIO, x)), y_pred))

        print(self.get_metrics(new_y_test, new_y_pred, b=1))  ## printing new metric to calculate F1

    def check_learnt(self):
        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')

        info = tagger.info()

        print("\nTop Positive Features")
        counter = 1
        for (attr, label), weight in Counter(info.state_features).most_common(100):
            if counter < 10:
                print("  Rank %s ---> %0.6f %-6s %s" % (counter, weight, label, attr))
            elif counter >= 10 and counter <= 99:
                print(" Rank %s ---> %0.6f %-6s %s" % (counter, weight, label, attr))
            else:
                print("Rank %s ---> %0.6f %-6s %s" % (counter, weight, label, attr))
            counter += 1


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


if __name__ == "__main__":
    model = CNFModel()
    model.train_model()
    model.predict()
    #model.check_learnt()