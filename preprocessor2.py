import codecs
import re
import nltk
import string
import spacy

from bs4 import BeautifulSoup as bs
from bs4.element import Tag

from contractions import CONTRACTION_MAP

RESTAURANT_TRAIN_DIRECTORY = "data/train_data/Restaurants_Train_v2.xml"
RESTAURANT_TEST_DIRECTORY = "data/test_data/Restaurants_Test_Truth.xml"

LAPTOP_TRAIN_DIRECTORY = "data/train_data/Laptop_Train_v2.xml"
LAPTOP_TEST_DIRECTORY = "data/test_data/Laptops_Test_Truth.xml"


class Preprocessor2:

    def __init__(self, train_file_directory, test_file_directory):

        with codecs.open(train_file_directory, "r", "utf-8") as train_file:
            self.train_soup = bs(train_file, "html5lib")

        with codecs.open(test_file_directory, "r", "utf-8") as test_file:
            self.test_soup = bs(test_file, "html5lib")


        # contains data of all training sentences, each sentence broken down into individual (word, pos, bio_tag)
        self.train_data = self.build_corpus(self.train_soup)

        print("Preprocessed Train data...")

        # contains data of all test sentences
        self.test_data = self.build_corpus(self.test_soup)

        print("Preprocessed Test data...")

    def build_corpus(self, soup_used):
        docs = []
        nlp = spacy.load("en_core_web_sm")

        for elem in soup_used.find_all("sentence"):

            aspectTerms = []
            text = elem.text.strip()

            for c in elem.find_all("aspectterms"):
                for aspectTerm in c.find_all("aspectterm"):
                    # add aspect terms into the list
                    aspectTerms.append(aspectTerm['term'])

            labelled_sentence = []


            # Individual contents are not string tokens. It is spaCy's wrapper token
            tokenized_sentence = list(filter(lambda x: x.text not in string.punctuation, [token for token in nlp(self.expand_contractions(text))]))

            # split ['laptop charger', 'dim sum'] into [['laptop', 'charger'], ['dim', 'sum']]
            aspectTerms = list(map(lambda x: list(filter(lambda x: x not in string.punctuation, [word.text for word in nlp(self.expand_contractions(x))])), aspectTerms))

            for token in tokenized_sentence:
                if len(aspectTerms) == 0: # sentence has no aspect terms
                    labelled_sentence.append((token.text, token.pos_, 'O'))
                else:
                    # sentence has at least 1 aspect term
                    match = False
                    match_idx = -1

                    for aspect_term_tokenized in aspectTerms:
                        idx = 0

                        for aspect_word in aspect_term_tokenized:
                            if token.text == aspect_word:
                                match = True
                                match_idx = idx
                                break
                            else:
                                idx += 1

                        if match:
                            break

                    if match:
                        if match_idx == 0:
                            labelled_sentence.append((token.text, token.pos_, 'B'))
                        else:
                            labelled_sentence.append((token.text, token.pos_, 'I'))
                    else:
                        labelled_sentence.append((token.text, token.pos_, 'O'))



            docs.append(labelled_sentence)

        return docs

    def expand_contractions(self, text, contraction_mapping=CONTRACTION_MAP):
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


if __name__ == "__main__":
    pp = Preprocessor2(RESTAURANT_TRAIN_DIRECTORY, RESTAURANT_TEST_DIRECTORY)
    print(pp.train_data)