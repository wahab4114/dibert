import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.stem import SnowballStemmer

spacy_model = spacy.load("en_core_web_lg")
stemmer = SnowballStemmer("english")
default = ["rmPunct"]
#default = ["rmDigit", "rmPunct", "lowCase"]

# simple pre-processor
def apply_preprocessing(sentence, listOfPreprocessingSteps=default):
    preprocessed_text = strip_quotes(sentence)
    if "rmDigit" in listOfPreprocessingSteps:
        preprocessed_text = re.sub(r'[0-9]+', '', preprocessed_text)
    if "rmPunct" in listOfPreprocessingSteps:
        preprocessed_text = re.sub(r'[^\w\d.\s]', '', preprocessed_text)
    if "lowCase" in listOfPreprocessingSteps:
        preprocessed_text = preprocessed_text.lower()
    if "rmStop" in listOfPreprocessingSteps:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(preprocessed_text)
        preprocessed_text = (" ").join([w for w in word_tokens if not w in stop_words])
    if "lemma" in listOfPreprocessingSteps:
        spacyText = spacy_model(preprocessed_text)
        preprocessed_text = " ".join([token.lemma_ for token in spacyText])
    if "stem" in listOfPreprocessingSteps:
        spacyText = spacy_model(preprocessed_text)
        text_new = []
        for token in spacyText:
            stemmed_token = stemmer.stem(token.text)
            if token.pos_ not in ["VERB", "AUX"]:
                if token.text[0].isupper():
                    text_new.append(stemmed_token[0].upper() + stemmed_token[1:])
                else:
                    text_new.append(stemmed_token)
            else:
                text_new.append(token.text)
        preprocessed_text = " ".join(text_new)
    # remove multiple whitespaces
    preprocessed_text = " ".join(preprocessed_text.split())
    return preprocessed_text

def preprocess(text):
        regex = re.compile('(\"\s)(.*?)(\s\")')  # remove whitespace around quotes
        text = re.sub(regex, '\"\\2\"', text)
        regex = re.compile('(\(\s)(.*?)(\s\))')  # remove whitespace inside brackets
        text = re.sub(regex, '\(\\2\)', text)
        regex = re.compile("(\s*)([\.\,\'\!\?\)])")  # remove whitespace before .,'!?)
        text = re.sub(regex, '\\2', text)
        regex = re.compile("\(\s*")  # remove whitespace after (
        text = re.sub(regex, '(', text)
        text = text.replace('@-@', '')  # remove @-@
        regex = re.compile("\s+")  # remove multiple whitespace
        text = re.sub(regex, ' ', text)
        return text

def strip_quotes(text: str) -> str:
    if isinstance(text, str):
        text = text.strip()
        text = text.strip('\"')
    return text

def main():
   pass
main()