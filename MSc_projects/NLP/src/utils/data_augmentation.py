"""
for class imbalances 
e.g. nlplaug (Ma, 2019)
- word level techniques: synonym, antonyms, random word deleting (Del), random word substitution (Sub), contextual word embeddings (CWE): chained or Sub?
    · chained CWE best?
e.g. substitue or insert words with p probability?
- character level techniques: keyboard augmentation (adding characters depending on keyboard distance)
- sentence level: CWE to add sentence at the end of the text?, translate to another similar language - german?

- up and down sample min and maj class respectively?
"""
import nltk
import spacy
from nltk.corpus import stopwords, wordnet
import translators as ts
import random
from math import ceil

# Download NLTK resources
if not wordnet.fileids():
    nltk.download('wordnet')
if not stopwords.fileids():
    nltk.download('stopwords')

# Download SpaCy resources
try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
          "(don't worry, this will only happen once)")
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')


class DataAugmentationPipeline(object):
    def __init__(self, *augmentations) -> None:
        self.augmentations = augmentations

    def __call__(self, sample):
        sample_augmented = sample.copy()
        for augmentation in self.augmentations:
            sample_augmented['texts'] = augmentation(sample_augmented['texts'])
        return sample_augmented

class SynonymReplacement(object):
    def __init__(self, p: float = 0.1, t: float=0.8) -> None:
        # load spaCy model
        nlp = spacy.load('en_core_web_md') #English()
        self.spacy_tokenizer = nlp.tokenizer   
        
        self.p = p # probability of replacement
        self.t = t # similarity threshold

    def get_synonyms(self, word):
        synonyms = set()

        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = ''.join([char for char in synonym if char.isalpha() or char.isspace()])
                synonyms.add(synonym)

        if word in synonyms:
            synonyms.remove(word)

        return list(synonyms)

    def replace_with_synonym(self, token):
        synonyms = self.get_synonyms(token.text)
        synonyms = [self.spacy_tokenizer(syn) for syn in synonyms if syn != ""]

        # filter words and synonyms that don't have a vector representation 
        synonyms = list(
            filter(lambda syn_token: syn_token 
                    and syn_token.vector_norm 
                    and token 
                    and token.vector_norm
                    and token.similarity(syn_token) > self.t, synonyms
            )
        )

        if not synonyms or random.random() > self.p:
            return token.text
        else:
            return random.choice(synonyms).text

    def __call__(self, text: str):
        tokens = self.spacy_tokenizer(text)
        stop_words = set(stopwords.words('english'))

        # Replace tokens with synonyms
        new_tokens = [self.replace_with_synonym(token) if token.text.lower() not in stop_words else token.text for token in tokens]

        new_text = ' '.join([token for token in new_tokens])
        return new_text

class RandomDeleteWords(object):
    def __init__(self, p: float = 0.1) -> None:
        self.p = p
    
    def __call__(self, text: str):
        # Ensure sentence is not empty
        while True:
            sentence = ' '.join([word for word in text.split() if random.random() > self.p])
            if len(sentence) > 0:
                break
        return sentence

class RandomSubstitution(object):
    def __init__(self) -> None:
        # TODO
        pass

    def __call__(self, text: str):
        # TODO
        return text

class RandomSwapWords(object):
    def __init__(self, p=0.1) -> None:
        self.p = p

    def __call__(self, text: str):
        text_split = text.split()
        text_split = self.shuffle(text_split, 
                                  count=ceil(len(text_split)*self.p))
        return ' '.join(text_split)
    
    @staticmethod
    def shuffle(input_list, count):
        """Shuffles any n number of values in a list"""
        indices_to_shuffle = random.sample(range(len(input_list)), k=count)
        to_shuffle = [input_list[i] for i in indices_to_shuffle]
        random.shuffle(to_shuffle)
        for index, value in enumerate(to_shuffle):
            old_index = indices_to_shuffle[index]
            input_list[old_index] = value
        return input_list

class TranslateWords(object):
    def __init__(self, target_lang: str = "de", source_lang: str = "en", p: float = 0.4) -> None:
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.p = p 

    def __call__(self, text: str) -> str:
        translated_text = text

        if random.random() < self.p:
            try:
                translated_text = ts.translate_text(ts.translate_text(text, to_language=self.target_lang), to_language=self.source_lang)
            except:
                print(f'ERRROR parsing: \n{text}')
                
        # print(f'TRANSLATION: \n{text}\n-> {translated_text}\n')
        return translated_text
