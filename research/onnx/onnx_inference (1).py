import numpy as np
import json
import re
from tqdm import tqdm

from typing import List, Dict, Tuple
import onnxruntime as ort


SPECIAL_SYMBOLS_REGEX = r"([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)])"
SPECIAL_SYMBOLS_REGEX_2 = r"(\b\w+\b|[.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)])"

def add_spaces(text: str) -> str:
    return re.sub(SPECIAL_SYMBOLS_REGEX, r' \1 ', text)

def sub_special_symbols(text: str) -> str:
    text = re.sub(r'\n', ' __newline__ ', text)
    text = re.sub(r'\t', ' __tab__ ', text)
    return text

def tokenize(text: str) -> List[str]:
    results = re.findall(SPECIAL_SYMBOLS_REGEX_2, text.lower())
    return results

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

    return np.array(vector, dtype=np.float32) # workaround in Python, maybe in C++ would be out of the box - should be type float here


TFIDF_PATH = "tf_idf_new_data_15102023_1500.json"
MODEL_PATH = "svm_model_15102023_1500.onnx"


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

print(input_string_list)
float_input = generate_vector(input_string_list, tfidf_mapping)[None, :] # shape of the vector is 1 x TFIDF_DICT_SIZE - 2d array

input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: float_input})

print(pred_onnx[0])

### Evaluating over the val set
# val_arrays = val_set.text.apply(tokenize)
# true_val_set = val_arrays.apply(lambda v: generate_vector(v, tfidf_mapping))

# predictions = [sess.run(None, {input_name: float_input[None, :]}) for float_input in tqdm(true_val_set)]

# predictions_true = [pred[0] for pred in predictions]
# print(f1_score(val_y, predictions_true, average='macro'))
