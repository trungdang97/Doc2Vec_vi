#from VieTokenizer.tokenization.crf_tokenizer import CrfTokenizer
import importlib, importlib.util

def module_from_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

CrfTokenizer = module_from_file('CrfTokenizer', 'VieTokenizer/tokenization/crf_tokenizer.py')
tokenizer = CrfTokenizer('VieTokenizer/tokenization/pretrained_tokenizer.crfsuite')

print(tokenizer.get_tokenized("Hôm nay trời rất đẹp"))