from typing import Dict, List, Tuple
import numpy as np
import json
import re

import onnxruntime as ort


SPECIAL_SYMBOLS_REGEX = r"([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)])"
WORDS_AND_TOKENS_REGEX = r"\b([\w]+)\b|(([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)]))"


def add_spaces(text: str) -> str:
    return re.sub(SPECIAL_SYMBOLS_REGEX, r' \1 ', text)

def tokenize(text: str) -> List[str]:
    results = re.findall(WORDS_AND_TOKENS_REGEX, text.lower())
    entries = [entry for tpl in results for entry in tpl if entry]

    return entries


def generate_vector(words: List[str], tfidf_mapping: Dict[str, Tuple[float, int]]):

    n_len_vector = len(tfidf_mapping)
    vector = np.zeros(n_len_vector) # = [0 for _ in range(n_len_vector)]
    for word in words:
        tuple_to_unpack = tfidf_mapping.get(word)
        if tuple_to_unpack:
            tf_idf_val, idx = tuple_to_unpack
            if vector[idx] == 0:
                vector[idx] = tf_idf_val
            else:
                vector[idx] += tf_idf_val

    return [vector]


# PATH_TFIDF = "drive/MyDrive/tg_challenge/tf_idf_mapping_old_data.json"
# MODEL_PATH = "drive/MyDrive/tg_challenge/"

TFIDF_PATH = "tf_idf_mapping_old_data.json"
MODEL_PATH = "svm_model_best.onnx"



with open(TFIDF_PATH) as f:
    tfidf_mapping = json.load(f)


sess = ort.InferenceSession(MODEL_PATH)

test_string = """
public static void main(String[] args) {
    System.out.print("Hello World");
    LinkedList<String> list;
    for (int i = 0; i < list.size(); i++) {
        System.out.print("Hey there");
    }
}
"""

input_string_list = tokenize(add_spaces(test_string))

float_input = generate_vector(input_string_list, tfidf_mapping)


input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: float_input})

print(pred_onnx[0])