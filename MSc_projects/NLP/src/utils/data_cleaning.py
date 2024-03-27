import re
import string
from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords

class DataCleaningPipeline(object):
    def __init__(self, *cleaners) -> None:
        self.cleaners = cleaners

    def __call__(self, sample):
        sample_cleaned = sample.copy()
        for cleaner in self.cleaners:
            sample_cleaned['texts'] = cleaner(sample_cleaned['texts'])
        return sample_cleaned
    

class TokenizeLinks(object):
    def __init__(self) -> None:
        pass

    def __call__(self, text: str):    
        return re.sub(r'https? : \S+', '[WEBSITE]', text)


class RemoveReferencing(object):
    def __init__(self) -> None:
        pass

    def __call__(self, text: str):    
        return re.sub(r'@\S+', '', text)


class RemoveSpecialCharaters(object):
    def __init__(self) -> None:
        self.exclude = set(string.punctuation).union(set(string.digits))

    def __call__(self, text: str):
        # return ''.join(ch for ch in text if ch not in self.exclude)
        text = ''.join(ch for ch in text if ch not in self.exclude)
        # if '5' in text: print("Oooops 5")
        return text


class ToLowercase(object):
    def __init__(self) -> None:
        pass

    def __call__(self, text: str):
        return text.lower()
    

class RemoveStopwords(object):
    def __init__(self) -> None:
        nltk.download('stopwords')
        self.stopwords = set(stopwords.words('english'))

    def __call__(self, text: str):
        return " ".join([word for word in text.split() if word.lower() not in self.stopwords])
    

class RemoveShortWords(object):
    def __init__(self, min_len=3) -> None:
        self.min_len = min_len

    def __call__(self, text: str):
        return " ".join([word for word in text.split() if len(word) >= self.min_len])

class StemWords(object):
    def __init__(self) -> None:
        self.porter = PorterStemmer()

    def __call__(self, text: str):
        return " ".join([self.porter.stem(word) for word in text.split()])


class RemoveSingleQuotes(object):
    def __init__(self) -> None:
        pass

    def __call__(self, text: str):
        # return text.upper()
        stop_words = ['this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'don', "don't", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'shan', "shan't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

        text.lower()
        # removing " at start of sentences
        # text = text.strip("\"")
        # replacing repetitions of punctations
        # text = re.sub(r'\"+', '', text)

        # Tokenize links
        # text = re.sub(r'https? : \S+', '[WEBSITE]', text)
        # removing referencing on usernames with @
        # text = re.sub(r'@\S+', '', text)
        # removing smileys with : (like :),:D,:( etc)
        # text = re.sub(r':\S+', '', text)
        # Remove punctation
        # text = re.sub(r"[!.,;:?\'\"\´]", "", text)
        # text = re.sub('(?<![\w])20[0-5][0-9]-?[0-9]*',
        #             '[YEAR]', text)              # Year token
        # text = re.sub('(?<![\w])1[0-9]{3}-?[0-9]*',
        #             '[YEAR]', text)                 # Year token
        # replacing numbers with [NUM] tag  eg 1,000, 1.32, 5-7. Assert these numbers are not inside words (i.e. H1, )
        # text = re.sub('(?<![\w])[0-9]+[.,]?[0-9]*(?![\w])', '[NUM]', text)
        # text = re.sub('\[NUM\]-\[NUM\]', '[NUM]', text)
        # Again to delete account numbers lol 12-5223-231
        # text = re.sub('\[NUM\]-\[NUM\]', '[NUM]', text)
        # text = re.sub('(?<=\[NUM\])-(?=[a-zA-Z])', ' ', text)
        # text = re.sub('[ ]*', ' ', text)
        # text = re.sub('<h>', '.', text)

        porter = PorterStemmer()
        words = text.split()
        for i, word in enumerate(words):
            # if word in stop_words:
            #     words.pop(i)
            # else:
            words[i] = porter.stem(word)
        # return words
        text = " ".join(words)
        
        text.lower()

        for i, word in enumerate(words):
            if word in stop_words:
                words.pop(i)



        return text

    # def __call__(self, text: str):
    #     # return text.upper()
    #     text = text.apply(lambda x: re.sub("'", '', x))
    #     text = text.apply(lambda x: replace_contractions(x)) 
    #     text = text.apply(lambda x: re.sub(r'https?:/\/\S+', ' ', x)) # remove urls
    #     text = text.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    #     text = text.apply(lambda x: ''.join(ch for ch in x if ch in include))
    #     text = text.apply(lambda x: x.strip())
    #     text = text.apply(lambda x: re.sub(" +", " ", x))
    #     return text