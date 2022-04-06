import re
from contractions import CONTRACTION_MAP
import spacy
import string

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


def test():
    nlp = spacy.load("en_core_web_sm")
    text = "Food is always fresh and hot- ready to eat!"

    tokenized_sentence = list(
        filter(lambda x: x.text not in string.punctuation, [token for token in nlp(expand_contractions(text))]))

    print(tokenized_sentence)


if __name__ == "__main__":
    test()