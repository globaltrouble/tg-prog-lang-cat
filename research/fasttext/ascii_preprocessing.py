from tqdm import tqdm
from typing import List, Dict
import re2 as re

import pandas as pd

import string


SPECIAL_SYMBOLS_REGEX = r"([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\)])"
SPECIAL_SYMBOLS_REGEX_2 = r"(\b\w+\b|[.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)])"

def add_spaces(text: str) -> str:
    return re.sub(SPECIAL_SYMBOLS_REGEX, r' \1 ', text)


def tokenize(text: str) -> List[str]:
    results = re.findall(SPECIAL_SYMBOLS_REGEX_2, text)
    return results

tqdm.pandas()

INT_REGEX = "-?\d+"
FLOAT_REGEX = "-?\d*[.,]\d+"
HEX_REGEX = "0[xX]([0-9a-fA-F])+"
OCTAL_REGEX = "0[oO]([0-7])+"
BINARY_REGEX = "0[bB]([01])+"
EXP_REGEX = "-?\d+[eE]-?\d+"



config = [
    {'regex': BINARY_REGEX, 'change_to': '<num_binary>'},
    {'regex': OCTAL_REGEX, 'change_to': '<num_octal>'},
    {'regex': HEX_REGEX, 'change_to': '<num_hex>'},
    {'regex': EXP_REGEX, 'change_to': '<num_exp>'},
    {'regex': FLOAT_REGEX, 'change_to': '<num_float>'},
    {'regex': INT_REGEX, 'change_to': '<num_int>'},
]


special_mapping = {
    '! !': '<SPECIAL_EXCLAMATIONEXCLAMATION>',
    '! .': '<SPECIAL_EXCLAMATIONDOT>',
    '! =': '<SPECIAL_EXCLAMATIONEQUAL>',
    '! = =': '<SPECIAL_EXCLAMATIONEQUALEQUAL>',
    '$ {': '<SPECIAL_DOLLAR>',
    '+ +': '<SPECIAL_PLUSPLUS>',
    '- -': '<SPECIAL_MINUSMINUS>',
    '- >': '<SPECIAL_MINUSGREATER>',
    '. !': '<SPECIAL_DOTEXCLAMATION>',
    ': :': '<SPECIAL_COLON>',
    ': =': '<SPECIAL_QUAL>',
    '< <': '<SPECIAL_LESSLESS>',
    '< =': '<SPECIAL_LESSEQUAL>',
    '= = =': '<SPECIAL_EQUALEQUALEQUAL>',
    '= >': '<SPECIAL_EQUALGREATER>',
    '> >': '<SPECIAL_GREATERGREATER>',
    '? .': '<SPECIAL_QUESTIONDOT>',
    '@ (': '<SPECIAL_AT>',
    '@ @': '<SPECIAL_ATAT>',
    '` ? ;': '<SPECIAL_BACKTICKQUESTION>'
}


def replace_special_chars(text: str) -> str:
    for key, value in special_mapping.items():
        text = text.replace(key, value)

    return text


leave_only_ascii = lambda text: "".join([symbol for symbol in text if symbol in string.printable]).strip()


def change_nums_to_tokens(config: List, text: str):

    for config_record in config:
        text = re.sub(config_record['regex'], config_record['change_to'], text)

    return text

def preprocess_text_to_ascii(text: str, is_fasttext=False) -> List[str]:

    text = leave_only_ascii(text)
    text = add_spaces(text)
    text = re.sub('\n+', ' <newline> ', text)
    text = change_nums_to_tokens(config, text)
    
    text = re.sub('\s+', ' ', text)
    text = replace_special_chars(text)
    
    return text

# def preprocess_set_to_ascii(df: pd.DataFrame):

#     assert all([column in df.columns for column in ["text"]])

#     df = df.copy()
#     df['text'] = df['text'].apply(add_spaces)
#     df['text_cleaned'] = df['text'].progress_apply(lambda text: change_nums_to_tokens(config, text))
#     df['text_cleaned_ascii'] = df['text_cleaned'].progress_apply(lambda text: leave_only_ascii(text))
#     df['text_cleaned_ascii'] = df['text_cleaned_ascii'].str.replace('\n', ' <newline> ')
    
#     return df

def prepare_fasttext_input(dataframe: pd.DataFrame, path: str):

    dataframe['class'] = "__label__" + dataframe['class']
    with open(path, 'w') as f_path:
        for _, line in dataframe.iterrows():
            f_path.write(f"{line['class']} {line['text_cleaned_ascii']}\n")