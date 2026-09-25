from typing import List
import numpy as np

def batch_list(input_list: List[str], batch_size: int):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

def calculate_mean_vector(input_file_path: str, embedding_helper) -> np.ndarray:
    input_lines = []
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip().split('\t', 1)[1]
            input_lines.append(sentence)

    vectors = []

    for batch in batch_list(input_lines, 64):
        batch_embeddings = embedding_helper.get_token_embeddings([s.strip() for s in batch])
        for sent_vecs in batch_embeddings:
            vectors.extend(sent_vecs)
    vectors_array = np.array(vectors, dtype='float32')
    mean_vector = np.mean(vectors_array, axis=0)
    return mean_vector