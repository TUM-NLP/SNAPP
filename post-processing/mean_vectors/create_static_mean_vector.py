import argparse
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple

import torch
import numpy as np

class LanguageModelClass:
    def __init__(self, model_path: str, layer: int = 8, device: str = 'cuda'):
        self.device = device
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, output_hidden_states=True) 
        self.model.to(self.device) 
        self.model.eval() 

    # method returns embeddings for each token of a sentence
    @torch.no_grad() 
    def get_token_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        inputs = self.tokenizer(sentences,padding=True,truncation=True, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()} 
        hidden_states = self.model(**inputs).hidden_states 
        layer_embeddings = hidden_states[self.layer] 
        attention_mask = inputs['attention_mask']
        output_vectors = []
        for i in range(len(sentences)):
            real_tokens_count = attention_mask[i].sum()
            sentence_vectors = layer_embeddings[i, 1 : real_tokens_count - 1, :]
            output_vectors.append(sentence_vectors.cpu().numpy())
        return output_vectors

def batch_list(input_list: List[str],batch_size: int):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

def main(input_file_path_train: str, mean_vector_output_path: str, language_model: str):
    embedding_helper = LanguageModelClass(model_path=language_model)
    input_lines = []
    with open(input_file_path_train, 'r', encoding='utf-8') as file:
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
    np.savetxt(mean_vector_output_path, mean_vector, fmt="%.15f")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        input_file_path_train=args.input_file_path,
        mean_vector_output_path=args.output_file_path,
        language_model=args.model_name
    )