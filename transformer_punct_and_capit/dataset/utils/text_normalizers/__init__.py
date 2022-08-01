import re, os
from num2words import num2words
    
class NemoNormalizer:
    def __init__(self, lang):
        from nemo_text_processing.text_normalization.normalize import Normalizer
        import logging
        logger = logging.getLogger("nemo_logger")
        logger.setLevel(logging.ERROR)
        
        self.normalizer = Normalizer(lang=lang, input_case='cased')
        
    def __call__(self, txt):
        norm_txt = []
        for sent in txt.split(". "):
            norm_sent = []
            for sub_sent in sent.split(", "):
                norm_sub_sent = self.normalizer.normalize(sub_sent, verbose=False)
                if len(norm_sub_sent):
                    norm_sent.append(norm_sub_sent.strip())
                    
            norm_txt.append(", ".join(norm_sent))
            
        return ". ".join(norm_txt)
    
class SnakersRussianNormalizer:    
    def __init__(self):
        from .ru_text_normalization.normalizer import Normalizer
        jit_model = f"{str(os.path.dirname(__file__))}/ru_text_normalization/jit_s2s.pt"
        self.normalizer = Normalizer(jit_model=jit_model, device='cpu')
        
    def __call__(self, txt):
        norm_txt = []
        for sent in txt.split(". "):
            norm_sent = []
            for sub_sent in sent.split(", "):
                if len(sub_sent.strip()):
                    norm_sub_sent = self.normalizer.normalize(sub_sent.strip(), verbose=False)
                else:
                    norm_sub_sent = ""
                    
                if len(norm_sub_sent):
                    norm_sent.append(norm_sub_sent.strip())
                    
            norm_txt.append(", ".join(norm_sent))
            
        return ". ".join(norm_txt)
    
class SpeechioChineseNormalizer:    
    def __init__(self, remove_punctuation=False):
        from .zh_text_normalization import Normalizer
        self.normalizer = Normalizer(remove_punctuation=remove_punctuation)
        
    def __call__(self, txt):
        norm_txt = []
        for sent in txt.split(". "):
            norm_sent = []
            for sub_sent in sent.split(", "):
                norm_sub_sent = self.normalizer.normalize(sub_sent, verbose=False)
                if len(norm_sub_sent):
                    norm_sent.append(norm_sub_sent.strip())
                    
            norm_txt.append(", ".join(norm_sent))
            
        return ". ".join(norm_txt)

_comma_re = re.compile(r'([0-9])(\,)([0-9])')
_decimal_re = re.compile(r'([0-9])(\.)([0-9])')
_ordinal_re = re.compile(r'([0-9])(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

class GenericNumberNormalizer:    
    def __init__(self, lang):
        self.lang = lang
        
    def _remove_comma(self, m):
        return m.group(0).replace(',', '')

    def _expand_decimal_point(self, m):
        return m.group(0).replace('.', ' point ')

    def _num2words_cardinal(self, m):
        if self.lang == 'it':
            if float(m.group(0)) == int(m.group(0)):
                number = int(m.group(0))
            else:
                number = float(m.group(0))
        else:
            number = m.group(0)
            
        return num2words(number, lang=self.lang, to='cardinal')

    def _num2words_ordinal(self, m):
        if self.lang == 'it':
            if float(m.group(1)) == int(m.group(1)):
                number = int(m.group(1))
            else:
                number = float(m.group(1))
        else:
            number = m.group(1)
            
        return num2words(number, lang=self.lang, to='ordinal')
    
    def normalize_numbers(self, text, verbose=None):
#         text = re.sub(_comma_re, self._remove_comma, text)
        # text = re.sub(_decimal_re, self._expand_decimal_point, text)
        text = re.sub(_ordinal_re, self._num2words_ordinal, text)
        text = re.sub(_number_re, self._num2words_cardinal, text)
        return text
        
    def __call__(self, txt):
        norm_txt = []
        for sent in txt.split(". "):
            norm_sent = []
            for sub_sent in sent.split(", "):
                norm_sub_sent = self.normalize_numbers(sub_sent, verbose=False)
                if len(norm_sub_sent):
                    norm_sent.append(norm_sub_sent.strip())
                    
            norm_txt.append(", ".join(norm_sent))
            
        return ". ".join(norm_txt)
    
class SkipNormalizer:
    def __call__(self, txt):
        return txt
    
def load_normalizer(lang):
    if lang in ['en', 'es', 'de']:
        return NemoNormalizer(lang)
    elif lang == 'ru':
        return SnakersRussianNormalizer()
    elif lang == 'zh':
        return SpeechioChineseNormalizer()
    elif lang == 'none':
        return SkipNormalizer()
    else:
        return GenericNumberNormalizer(lang)