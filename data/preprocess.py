import pandas as pd
import numpy as np
import re
import json
import string
from spellchecker import SpellChecker
import spacy
from bs4 import BeautifulSoup
import unicodedata
import nltk
from nltk.corpus import stopwords
import configs.config as config

DATA_DIR = config.path["data"]

DATA_PATH = f"{DATA_DIR}amazon_reviews.csv"

# Based on the work of:
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
# https://towardsdatascience.com/text-normalization-7ecc8e084e31

with open(f"{DATA_DIR}english_contractions.json", "r") as fd:
    CONTRACTION_MAP = json.loads(fd.read())

with open(f"{DATA_DIR}eng_abbrv.json", "r") as fd:
    ABBRV_MAP = json.loads(fd.read())

class TextProcessing():
    def __init__(self, remove_stopwords=False, remove_punctuations=False, correct_spelling=True):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuations = remove_punctuations
        self.correct_spelling = correct_spelling
        self.contraction_map = CONTRACTION_MAP
        self.abbreviation_map = ABBRV_MAP
        self.nlp = spacy.load("en_core_web_sm")
        self.spell = SpellChecker()
        
    def tokenize(self, text):
        return [str(word) for word in self.nlp(str(text))]
    
    def preprocess(self, text):
        text = self.remove_markup_tags(text)
        
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.normalize_punctuations(text)
        text = self.substitute_value_to_types(text)
        text = self.remove_special_chars(text)
        text = self.expand_contractions(text)
        text = self.expand_abbreviations(text)
        if self.correct_spelling:
            text = self.spelling_correction(text)
        text = self.to_lowercase(text)
        text = self.normalize_whitespaces(text)
        if self.remove_punctuations:
            text = ''.join([c for c in text if c not in string.punctuation])
        if self.remove_stopwords:
            text = [i for i in text.split() if i not in stopwords.words('english')]
            text = " ".join(text)
        # Remove extra whitespaces
        return re.sub(r'^\s*|\s\s*|( )+', ' ', text).strip()
    
    def to_lowercase(self, text):
        return text.lower()
    
    def remove_urls(self, text):
        text = str(text)
        url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+' +\
            r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]' +\
            r'\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}' +\
            r'|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
        text = re.sub(url_regex, " ", text)
        return text
    
    def remove_markup_tags(self, text):
        """
        This function removes any markup tag.
        """
        new_text = BeautifulSoup(text, 'html.parser').get_text()
        return new_text
    
    def normalize_punctuations(self, text):
        """
        This function simplifies doubled or more complex punctuation. The exception is '...'.
        """
        text = str(text)
        text = re.sub(r'([!?,;])\1+', r'\1', text)
        text = re.sub(r'\.{2,}', r'...', text)
        text = re.sub(r'\.{3}( )+', r'. ', text)
        text = re.sub(r'\.{3}', '. ', text)
        return text
    
    def remove_emojis(self, text):
        emojis_pattern = r"(:|;|\^\^)\-?[\(\)@\$\%oDp]"
        text = re.sub(emojis_pattern, " ", text)
        #text = re.sub("(^|( )+)[0-9A-Fa-f]+($|( )+)", " ", text)
        return text
    
    def normalize_whitespaces(self, text):
        """
        This function normalizes whitespaces, removing duplicates.
        """
        text = str(text)
        text = re.sub(r"//t", r" ", text)
        text = re.sub(r"( )\1+", r"\1", text)
        text = re.sub(r"(\n)\1+", r" ", text)
        text = re.sub(r"(\r)\1+", r" ", text)
        text = re.sub(r"(\t)\1+", r" ", text)
        return text.strip(" ")
    
    def remove_accented_chars(self, text):
        text = str(text)
        new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return new_text
    
    def remove_special_chars(self, text):
        text = str(text)
        
        # Replace parenthesis with spaces
        new_text = re.sub(r"\(|\)", r" ", text)
        
        # Turn "--" and underscores (_) to spaces
        new_text = re.sub(r"\-\-+|_+", r" ", new_text)
        
        # handle forward slash
        slash_pattern = re.compile(r"([a-zA-Z\'])(\/)([a-zA-Z\'])", flags=re.IGNORECASE|re.DOTALL)
        new_text = slash_pattern.sub(
            lambda contraction: contraction.group(1).strip() + " or " +  contraction.group(3).strip(), new_text)
        
        # define the pattern to keep
        pattern = r'[^a-zA-z0-9.,!?/:;\'\s\-]'
        new_text = re.sub(pattern, '', new_text)
        
        # remove hanging dash
        new_text = re.sub(r"( )+\-( )+", "", new_text)
        
        return new_text
    
    def expand_contractions(self, text):
        text = str(text)
        return self.__expand_word_form(text, map_=self.contraction_map)
    
    def expand_abbreviations(self, text):
        text = str(text)
        return self.__expand_word_form(text, map_=self.abbreviation_map)
    
    def __expand_word_form(self, text, map_):

        map_keys = [re.escape(k) for k in map_.keys()]
        pattern = re.compile(r'(^|\W+)({})(\W+|$)'.format('|'.join(map_keys)), flags=re.IGNORECASE|re.DOTALL)
        
        def get_match(contraction):
            match = contraction.group(2)
            expanded = map_.get(match) if map_.get(match) else map_.get(match.lower())
            if expanded is None:
                #Return same text if it match but no expand
                expanded = match
            
            return contraction.group(1) + expanded + contraction.group(3) 
        
        new_text = pattern.sub(get_match, text)
        new_text = re.sub("'", "", new_text)

        return new_text
    
    def substitute_value_to_types(self, text):
        """
        This function substitute numeric values to their types
        e.g. $50 -> MONEY
        """
        text = str(text)
        money_pattern = r'((about|around|nearly|close to|exactly|almost)( )*)?' +\
                        r'((\$|USD|CAD|CA\$|GBP|EUR|AUD|HKD|INR|NZD)( )*[0-9\.,]+' +\
                        r'|[0-9\.,]+( )*(\$|USD|CAD|CA\$|GBP|EUR|AUD|HKD|INR|NZD|bucks?)' +\
                        r'|[0-9\.,]+( )*(dollar|euro|ruppee)(s)?)'
        quantity_pattern = r'((about|around|nearly|close to|exactly|almost)( )*)?' +\
                           r'[0-9\.,]+( )*(millilitters?|litters?|kilograms?|grams?|ounces?|' +\
                           r'ml|l|kg|g|oz)'
        distance_pattern = r'((about|around|nearly|close to|exactly|almost)( )*)?' +\
                           r'[0-9\.,]+( )*((kilometters?|millimitters?|centimetters?|yards?|inches?|feet|foot|' +\
                           r'km|mm|cm|yd|in|ft)|(\"))'
        percentage_pattern = r'((about|around|nearly|close to|exactly|almost)( )*)?' +\
                             r'(\%( )*[0-9\.,]+|[0-9\.,]+( )*\%|[0-9\.,]+( )*percent)'
        
        new_text = re.sub(money_pattern, "some amount of money", text)
        new_text = re.sub(quantity_pattern, "some amount", new_text)
        new_text = re.sub(percentage_pattern, "a percentage", new_text)
        
        # TODO: implement year conversion e.g. in 2005 -> some time ago | july, 25th -> some time ago
        # TODO: implement number conversion e.g. 5000 people -> a number of people
        #new_text = re.sub(r'[0-9\.,]+', "a number of", new_text)
        
        # Remove duplicated "of"
        new_text = re.sub("of( )?of", "of", new_text)
        return new_text
    
    def spelling_correction(self, text):
        """
        This function does very simple spell correction normalization using pyspellchecker module. 
        It works over a tokenized sentence and only the token representations are changed.
        """
        text = str(text)
        
        if len(text) < 1:
            return ""
        
        token_list = self.tokenize(text)
        
        for word_pos in range(len(token_list)):
            word = token_list[word_pos]
            
            if word is None:
                token_list[word_pos] = ""
                continue
            
            if not '\n' in word and word not in string.punctuation and not self.is_numeric(word):
                #Checks first uppercase to conserve the case.
                upperfirst = word[0].isupper()
                #Checks for correction suggestions. #We call our __reduce_exaggerations function if no suggestion is found. Maybe there are repeated chars.
                replacement = self.__reduce_exaggerations(word)
                #Takes the case back to the word.
                if upperfirst:
                    replacement = replacement[0].upper()+replacement[1:]
                word = replacement
                
                if word != " " and self.spell.correction(word) != word:
                    word = self.spell.correction(word)
                
                token_list[word_pos] = word
                
        return " ".join(token_list).strip()

    def __reduce_exaggerations(self, text):
        """
        Auxiliary function to help with exxagerated words.
        Examples:
            woooooords -> words
            yaaaaaaaaaaaaaaay -> yay
        """
        correction = str(text)
        return re.sub(r'([\w])\1{3,}', r'\1', correction)

    def is_numeric(self, text):
        if not re.search("[0-9,\%\.\$]", text):
            return False
        return True


