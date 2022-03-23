import string

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import re
import nltk
import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report


from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# nltk.download('punkt')

# Read data file and parse the XML
with codecs.open("data/train_data/Restaurants_Train_v2.xml", "r", "utf-8") as infile:
    soup = bs(infile, "html5lib")


def findall_occurences(aspectTerms, text):
    lst = []
    for term in aspectTerms:
        currLst = [(m.start(), term) for m in re.finditer(term, text)]
        lst = currLst + lst
        text = re.sub(term, "ASPECT", text)
    lst.sort(key=lambda a: a[0])
    return [item[1:][0] for item in lst], text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

docs = []
for elem in soup.find_all("sentence"):
    texts = []
    aspectTerms = []
    updated_tokens = []

    for c in elem.find_all("aspectterms"):
        for aspectTerm in c.find_all("aspectterm"):
            # add aspect terms into the list
            aspectTerms.append(aspectTerm['term'])

    for text in elem.find("text"):
        aspectTermsLst, modified_text = findall_occurences(aspectTerms, text)

        tokens = nltk.word_tokenize(expand_contractions(modified_text))

        for token in tokens:
            word = token.strip(string.punctuation)
            if len(word) != 0:
                updated_tokens.append(word)

        for i in range(len(updated_tokens)):
            if updated_tokens[i] == "ASPECT":
                updated_tokens[i] = aspectTermsLst[0]
                aspectTermsLst.pop(0)

        for token in updated_tokens:
            # Remove punctuations except for special cases such as 17-inch, "sales" team
            label = ""
            # if in aspectTerm then label = A
            # else label = N

            if token in aspectTerms:
                token_list = token.split()
                label = "B"
                texts.append((token_list[0], label))
                if len(token_list) > 1:
                    label = "I"
                    token_list.pop(0)
                    for t in token_list:
                        texts.append((t, label))
            else:
                label = "O"
                texts.append((token, label))
    docs.append(texts)

data = []
for i, doc in enumerate(docs):

    # Obtain the list of tokens in the document
    tokens = [t for t, label in doc]

    # Perform POS tagging
    tagged = nltk.pos_tag(tokens)

    # Take the word, POS tag, and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])


def word2features(sentence, i):
    current_word = sentence[i][0]
    current_pos = sentence[i][1]

    lemmatizer = WordNetLemmatizer()
    superlatives = ['JJS', 'RBS']
    comparatives = ['JJR', 'RBR']

    features = [
        'bias',
        'word.lower=' + current_word.lower(),
        'word[-3:]=' + current_word[-3:],
        'word[-2:]=' + current_word[-2:],
        'postag=' + current_pos,
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        prev_word = sentence[i-1][0]
        previous_pos = sentence[i-1][1]
        features.extend([
            '-1:word.lower=' + prev_word.lower(),
            '-1:postag=' + previous_pos,
        ])
    else:
        features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]
        next_pos = sentence[i + 1][1]
        features.extend([
            '+1:word.lower=' + next_word.lower(),
            '+1:postag=' + next_pos,
        ])
    else:
        features.append('EOS')

    return features

def extract_features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

def get_label(sentence):
    return [label for (token, pos, label) in sentence]

def train_and_validate_model():

    X = [extract_features(sentence) for sentence in data]
    y = [get_label(sentence) for sentence in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')

    y_pred = [tagger.tag(xseq) for xseq in X_test]

    labels = {"B": 0, 'I': 1, 'O': 2}

    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    print(classification_report(truths, predictions, target_names=['B', 'I', 'O']))

    pass



if __name__ == "__main__":
    train_and_validate_model()