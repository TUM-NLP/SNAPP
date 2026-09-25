from simalign import SentenceAligner

# ---------------------------------------------------------------
# Setup SimAlign
# Reference: 
# Authors: Jalili Sabet, Masoud  and Dufter, Philipp  and Yvon, Fran{\c{c}}ois  and Sch{\"u}tze, Hinrich
# Title: {S}im{A}lign: High Quality Word Alignments Without Parallel Training Data Using Static and Contextualized Embeddings
# Year: 2020
# ---------------------------------------------------------------


from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Set, Tuple

import sys
import string
import argparse
import numpy as np
import re
import torch
import nltk
from nltk.corpus import stopwords
from create_mean_vector_dynamically import calculate_mean_vector

PUNCTUATION_SET = set(string.punctuation)
NEW_WORD_PREFIX = "\u2581"          # NOTE: perhaps different for other language models
nltk.download('stopwords', quiet=True)


# ---------------------------------------------------------------
# Implementation of the segment alignment algorithm (with small adjustments e.g., SimAlign instead of Dictionaries) described in:
# Authors: Viktor Hangya and Alexander Fraser
# Title: Unsupervised Parallel Sentence Extraction with Parallel Segment Detection Helps Machine Translation
# Year: 2019
# Re-implemented independently based on the paper's methodology
# ---------------------------------------------------------------


class LanguageModelClass:

    # initialization of language model
    def __init__(self, model_path: str, layer: int = 8, device: str = 'cuda'):
        self.device = device
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) # load tokenizer
        self.model = AutoModel.from_pretrained(model_path, output_hidden_states=True) # load language model --> second parameter is responsible for accessing the intermediate results (e.g. layer 8)
        self.model.to(self.device) # --> loading model into hardware/ on GPU
        self.model.eval() #--> model is used for predictions instead of training


    # separates a sentence into its tokens using the initialized language model/ its tokenizer
    def get_tokens_for_sentence(self, sentence: str) -> List[str]:
        return self.tokenizer.tokenize(sentence)


    # method should return embedding for each token of sentence
    @torch.no_grad() # deactivates the calculation of the gradient for faster inference
    def get_token_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt") # generates tokens in form of vectors --> return_tensors="pt" generates output as PyTorch-tensors which is the correct format for the PyTorch models
        inputs = {key: val.to(self.device) for key, val in inputs.items()} # transfers the data to the correct hardware (cpu to gpu)
        hidden_states = self.model(**inputs).hidden_states # tokenized sentences are fed into the neural network --> forward pass is executed by the model
        layer_embeddings = hidden_states[self.layer] #chooses results from layer 8 (specified in __init__ function)
        attention_mask = inputs['attention_mask']
        output_vectors = []
        for i in range(len(sentences)):
            real_tokens_count = attention_mask[i].sum()

            # NOTE: this assumes CLS and SEP from Glot500 --> check if this need potentially be adjusted for the model you are using (except for Glot500 and probably XLM-R)
            sentence_vectors = layer_embeddings[i, 1 : real_tokens_count - 1, :]

            output_vectors.append(sentence_vectors.cpu().numpy())

        return output_vectors


def load_custom_stopwords(file_path: str) -> set:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError as e:        
        print(f"Stopword file not found: {e}")
        sys.exit(1)


# function extracts training/ test data from file and returns dictionary
def getSentencesFromFile(file_path: str) -> Dict[int,str]:
    """
    function extracts training/ test data from file and returns dictionary
    """
    res = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                id_part, sentence = parts
                key_str = id_part.split('-')[1]
                key = int(key_str)
                value = sentence.strip()
                res[key] = value
    return res


# this function processes the sentence mapping file (output) from the pipeline
def getMappedNumbers(file_path: str) -> List[Tuple[int, int]]:
    """
    function extracts sentence numbers which are mapped to each other (src -> trg) from file and returns tuple list
    """
    res = list()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                id_src, id_trg = parts
                key_src_str = id_src.split('-')[1]
                key_trg_str = id_trg.strip().split('-')[1]
                res.append((int(key_src_str), int(key_trg_str)))
    return res


# NOTE: the extended version of this function has the assumption that as soon as a punctuation mark is within a token, the next token should be a new word
# this function transfers the embedding from token level to word level
def transfer_tknembedding_to_word_level_embedding(token_vectors: np.ndarray, sentence_tkns: List[str], apply_mean_subtraction: bool, word_mean: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    res_word_list = []
    res_vector_list = []
    temp_concat_tokens_str = ""
    temp_concat_vectors = []
    for i, tkn_temp in enumerate(sentence_tkns):
        if any(char in PUNCTUATION_SET for char in tkn_temp):
            if len(temp_concat_tokens_str) > 0:
                res_word_list.append(temp_concat_tokens_str)
                res_vector_list.append(np.mean(temp_concat_vectors, axis=0))
                temp_concat_vectors = []
                temp_concat_tokens_str = ""
            if tkn_temp.startswith(NEW_WORD_PREFIX):
                tkn_temp = tkn_temp[1:]
            res_word_list.append(tkn_temp)
            res_vector_list.append(token_vectors[i])
        elif tkn_temp.startswith(NEW_WORD_PREFIX):
            if len(temp_concat_tokens_str) > 0:
                res_word_list.append(temp_concat_tokens_str)
                res_vector_list.append(np.mean(temp_concat_vectors, axis=0))
                temp_concat_vectors = []
            temp_concat_tokens_str = tkn_temp[1:]
            temp_concat_vectors.append(token_vectors[i])
        else:
            temp_concat_tokens_str += tkn_temp
            temp_concat_vectors.append(token_vectors[i])

    if len(temp_concat_tokens_str) > 0:
        res_word_list.append(temp_concat_tokens_str)
        res_vector_list.append(np.mean(temp_concat_vectors, axis=0))

    array_vector = np.array(res_vector_list)

    if apply_mean_subtraction:
        centered_embeddings = array_vector - word_mean
    else:
        centered_embeddings = array_vector

    return (centered_embeddings, res_word_list)


#calculates the cosine similarity between two vectors and returns the similarity as a float number
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


# iterate through every token alignment provided by SimAlign --> indices in token list wher no alignment word in trg was found will appear in cos_scores as 0.0
def get_similarity_scores(alignments_tuple_list: List[Tuple], token_vectors_from: np.ndarray, token_vectors_to: np.ndarray, number_src_tkns: int) -> List[float]:
    cos_scores = [0.0] * number_src_tkns
    for id_from, id_to in alignments_tuple_list:
        vector_from = token_vectors_from[id_from]
        vector_to = token_vectors_to[id_to]
        cos_scores[id_from] = cosine_similarity(vector_from, vector_to)
    return cos_scores


def remove_terminal_punctuation(sim_scores: List[float], sentence_words: List[str], alignments_from: List[Tuple[int, int]], alignments_to: List[Tuple[int, int]]) -> Tuple[List[float], List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:

    if len(sentence_words) == 0:
        return sim_scores, sentence_words, alignments_from, alignments_to
    
    last_word = sentence_words[-1].replace(NEW_WORD_PREFIX, "").strip()
        
    if all(ch in PUNCTUATION_SET for ch in last_word):
        last_idx = len(sentence_words) - 1
        
        sim_scores_new = sim_scores[:-1]
        sentence_words_new = sentence_words[:-1]
        
        alignments_from_new = [(my_id, other_id) for (my_id, other_id) in alignments_from if my_id != last_idx]
        alignments_to_new = [(other_id, my_id) for (other_id, my_id) in alignments_to if my_id != last_idx]
        
        return sim_scores_new, sentence_words_new, alignments_from_new, alignments_to_new
        
    return sim_scores, sentence_words, alignments_from, alignments_to


# averages the similarity scores based on a window
def average_sim_scores(similarity_list: List[float], window_size: int) -> List[float]:
    if window_size % 2 == 0:
        window_size += 1
    radius = (window_size - 1) // 2

    if radius == 0:
        return similarity_list
        
    n = len(similarity_list)
    scores_array = np.array(similarity_list)
    averaged_scores = []
    
    for i in range(n):
        start_index = max(0, i - radius)
        end_index = min(n, i + radius + 1)
        window = scores_array[start_index:end_index]
        avg_score = np.mean(window)
        averaged_scores.append(avg_score)
        
    return averaged_scores


# this function extracts the segments based on averaged similarity scores and a minimum segment length --> returns a list of (start,end) index tuples 
def extract_segments(averaged_similarity_score_list: List[float], token_list: List[str], minimum_segment_length: int, threshold: float) -> List[Tuple[int,int]]:
    if len(averaged_similarity_score_list) != len(token_list):
        raise Exception("Error in extract_segments function --> score list and token list have different lengths")
    res_list = []
    current_segment_start_index = -1
    for i in range(len(averaged_similarity_score_list)):
        if averaged_similarity_score_list[i] >= threshold:
            if current_segment_start_index == -1:
                current_segment_start_index = i
        else:
            if current_segment_start_index != -1:
                segment_len = i - current_segment_start_index
                if segment_len >= minimum_segment_length:
                    res_list.append((current_segment_start_index, i)) # (start, end_exclusive)
                current_segment_start_index = -1
    if current_segment_start_index != -1:
        segment_len = len(averaged_similarity_score_list) - current_segment_start_index
        if segment_len >= minimum_segment_length:
            res_list.append((current_segment_start_index, len(averaged_similarity_score_list)))
    return res_list


# this function pairs the potential segments and returns a list of tuples with the index tuples of the segments
def pair_segments(
        src_segment_list: List[Tuple[int,int]], 
        trg_segment_list: List[Tuple[int,int]],
        word_alignments: List[Tuple[int,int]], 
        max_length_ratio: float = 0.25) -> List[Tuple[Tuple[int,int],Tuple[int,int]]]:
    
    paired_segment_list = []
    available_trg_segments = list(trg_segment_list)

    for (s_start,s_end) in src_segment_list:
        best_match_count = 0
        best_match_index = -1
        s_len = s_end - s_start

        for i, (t_start,t_end) in enumerate(available_trg_segments):
            t_len = t_end - t_start

            # filter ii) from paper
            # if abs(s_len - t_len) > max_length_diff:
            #     continue

            # adapted filter ii) from paper --> now relative difference instead of total difference
            max_len = max(s_len, t_len)
            if max_len > 0 and (abs(s_len - t_len) / max_len) > max_length_ratio:
                continue

            current_match_count = 0
            for (src_align_index, trg_align_index) in word_alignments: # find all word/ token allignments which are relevant
                if(s_start <= src_align_index < s_end) and (t_start <= trg_align_index < t_end): #both matched words are within the currently considered potentially parallel segments
                    current_match_count += 1
            
            if current_match_count > best_match_count:
                best_match_count = current_match_count
                best_match_index = i
            
        if best_match_index > -1:
            best_trg_segment = available_trg_segments.pop(best_match_index)
            paired_segment_list.append( ((s_start, s_end), best_trg_segment) )

    return paired_segment_list


# returns the length of the longest src segment
def longest_segment_length(paired_segments: List[Tuple[Tuple[int,int],Tuple[int,int]]]) -> int:
    res = 0
    for (src_tuple, _) in paired_segments:
        start_index, end_index = src_tuple
        if (end_index - start_index) > res:
            res = end_index - start_index
    return res


# calculates the final similarity score of a sentence pair to decide afterwards if the pair should be kept or not
def final_sim_score_for_sentence_pair(src_sentence_words: List[str], trg_sentence_words: List[str], sim_scores_src_to_trg: List[float], paired_segments: List[Tuple[Tuple[int,int],Tuple[int,int]]]) -> float:
    if len(paired_segments) == 0:
        return 0.0
    sum_scores = sum(sim_scores_src_to_trg)
    length_of_longest_segment = longest_segment_length(paired_segments)
    number_of_src_words = len(src_sentence_words)
    number_of_trg_words = len(trg_sentence_words)
    return (sum_scores/number_of_trg_words) * (length_of_longest_segment/number_of_src_words)


# this function deletes all stopwords which didn't get aligned before --> score gets deleted, word gets deleted and tuple lists are modified accordningly
def delete_unaligned_stopwords(sim_scores_src_to_trg: List[float], src_sentence_words: List[str], alignments_tuple_list_trg: List[Tuple[int,int]], alignments_tuple_list_src: List[Tuple[int,int]], stop_word_set: set) -> Tuple[List[float], List[str]]:
    sim_scores_src_to_trg_new = []
    src_sentence_words_new = []
    old_to_new_map: Dict[int,int] = {}
    new_index = 0
    for old_id, (score,word) in enumerate(zip(sim_scores_src_to_trg, src_sentence_words)):
        if score == 0.0 and word.lower() in stop_word_set:
            continue
        else:
            sim_scores_src_to_trg_new.append(score)
            src_sentence_words_new.append(word)
            old_to_new_map[old_id] = new_index
            new_index += 1
    
    for i, ((src_id_src, old_trg_id_src),(old_trg_id_trg, src_id_trg)) in enumerate(zip(alignments_tuple_list_trg, alignments_tuple_list_src)):
        if old_trg_id_src in old_to_new_map:
            alignments_tuple_list_trg[i] = (src_id_src, old_to_new_map[old_trg_id_src])
        if old_trg_id_trg in old_to_new_map:
            alignments_tuple_list_src[i] = (old_to_new_map[old_trg_id_trg], src_id_trg)
    return (sim_scores_src_to_trg_new, src_sentence_words_new)


def track_segment_length_ratios(src_sentence_words: List[str], trg_sentence_words: List[str], paired_segments: List[Tuple[Tuple[int,int],Tuple[int,int]]], final_similarity_score: float, is_gold_pair: bool, output_file_handle):
    number_words_src_sentence = len(src_sentence_words)
    number_words_trg_sentence = len(trg_sentence_words)

    if number_words_src_sentence == 0 or number_words_trg_sentence == 0:
        return
    
    length_longest_src_segm = 0
    length_longest_trg_segm = 0

    for (src_tuple, trg_tuple) in paired_segments:

        src_from, src_to = src_tuple
        trg_from, trg_to = trg_tuple

        current_src_len = src_to - src_from
        current_trg_len = trg_to - trg_from

        if current_src_len > length_longest_src_segm:
            length_longest_src_segm = current_src_len
        if current_trg_len > length_longest_trg_segm:
            length_longest_trg_segm = current_trg_len

    src_ratio = length_longest_src_segm/number_words_src_sentence
    trg_ratio = length_longest_trg_segm/number_words_trg_sentence


    output_file_handle.write(f"{src_ratio:.4f}\t{trg_ratio:.4f},\tfinal score:{final_similarity_score},\tgold_pair:{is_gold_pair}\n")
    return

def get_language_name(file_name: str, full_name: bool) -> str:
    language_abbreviation = file_name.split(".")[-1]

    if not full_name:
       return language_abbreviation
    
    language_map = {
        'ru': 'russian', 
        'es': 'spanish', 
        'de': 'german'
    }

    if language_abbreviation not in language_map:
        raise ValueError(f"Error: Unknown language extracted: '{language_abbreviation}' from '{file_name}'")
    
    return language_map[language_abbreviation]


# implements main functionality of the post processing
def main(mapping_file_name: str, src_file_name: str, trg_file_name: str, model_path: str, output_file_name: str, 
         segments_output_file_name: str, window_size: int, min_segment_length: float, segment_detection_threshold: float, 
         pair_filtering_threshold: float, matching_method: str, token_type: str, mean_subtraction: bool, 
         filter_stopwords_src: bool, filter_stopwords_trg: bool, exclude_punctuation: bool, dynamic_mean: bool):

    MODEL_TO_USE = model_path # "cis-lmu/glot500-base" --> model name from Hugging Face Hub or path to pretrained model
    embedding_helper = LanguageModelClass(model_path=MODEL_TO_USE)

    src_language_abbreviation = get_language_name(src_file_name, False)
    trg_language_abbreviation = get_language_name(trg_file_name, False)
    trg_full_language_name = get_language_name(trg_file_name, True)

    if dynamic_mean:
        LRL_WORD_MEAN = calculate_mean_vector(src_file_name, embedding_helper)
        HRL_WORD_MEAN = calculate_mean_vector(trg_file_name, embedding_helper)
    else:
        LRL_WORD_MEAN = np.loadtxt(f"../mean_vectors/mean_vector_{src_language_abbreviation}.txt")
        HRL_WORD_MEAN = np.loadtxt(f"../mean_vectors/mean_vector_{trg_language_abbreviation}.txt")
    
    matching_method_char_map = {'inter': 'a', 'mwmf': 'm', 'itermax': 'i'}
    simalign_method_char = matching_method_char_map.get(matching_method, 'a')

    myaligner = SentenceAligner(model=MODEL_TO_USE, token_type=token_type, matching_methods=simalign_method_char)

    mapping_list = getMappedNumbers(mapping_file_name)
    src_sentences_dict = getSentencesFromFile(src_file_name)
    trg_sentences_dict = getSentencesFromFile(trg_file_name)

    new_mapping_list = []
    extracted_segments = []
    for i, (src_num, trg_num) in enumerate(mapping_list):
        src_sentence = src_sentences_dict[src_num]
        trg_sentence = trg_sentences_dict[trg_num]

        src_sentence_tkns = embedding_helper.get_tokens_for_sentence(src_sentence)
        trg_sentence_tkns = embedding_helper.get_tokens_for_sentence(trg_sentence)

        # returns list of vectors, one for each token in sentence --> for both src and trg sentence
        src_token_vectors, trg_token_vectors = embedding_helper.get_token_embeddings([src_sentence, trg_sentence])

        # this function transfers the embedding from token level to word level
        src_word_lvl_vectors, src_sentence_words = transfer_tknembedding_to_word_level_embedding(src_token_vectors, src_sentence_tkns, mean_subtraction, LRL_WORD_MEAN)
        trg_word_lvl_vectors, trg_sentence_words = transfer_tknembedding_to_word_level_embedding(trg_token_vectors, trg_sentence_tkns, mean_subtraction, HRL_WORD_MEAN)

        if len(src_sentence_tkns) != src_token_vectors.shape[0] or len(trg_sentence_tkns) != trg_token_vectors.shape[0]:
            print("ATTENTION: length difference between tokenlist and vektor list for tokens. Skipping this sentence pair.")
            continue
        elif len(src_sentence_words) != src_word_lvl_vectors.shape[0] or len(trg_sentence_words) != trg_word_lvl_vectors.shape[0]:
            print("ATTENTION: length difference between wordlist and word list for words. Skipping this sentence pair.")
            continue

        """
        alignments is a dict[str, list[tuple[int,int]]]
        e.g.:
        mwmf (Match): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        inter (ArgMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        itermax (IterMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        """
        alignments_tuple_list_src = myaligner.get_word_aligns(src_sentence_words, trg_sentence_words)[matching_method]
        alignments_tuple_list_trg = myaligner.get_word_aligns(trg_sentence_words, src_sentence_words)[matching_method]

        sim_scores_src_to_trg = get_similarity_scores(alignments_tuple_list_src, src_word_lvl_vectors, trg_word_lvl_vectors,len(src_sentence_words))
        sim_scores_trg_to_src = get_similarity_scores(alignments_tuple_list_trg, trg_word_lvl_vectors, src_word_lvl_vectors,len(trg_sentence_words))

        if exclude_punctuation:
            sim_scores_src_to_trg, src_sentence_words, alignments_tuple_list_src, alignments_tuple_list_trg = remove_terminal_punctuation(
                sim_scores_src_to_trg, src_sentence_words, alignments_tuple_list_src, alignments_tuple_list_trg)
            sim_scores_trg_to_src, trg_sentence_words, alignments_tuple_list_trg, alignments_tuple_list_src = remove_terminal_punctuation(
                sim_scores_trg_to_src, trg_sentence_words, alignments_tuple_list_trg, alignments_tuple_list_src)

        if filter_stopwords_src:
            LRL_STOP_WORDS = load_custom_stopwords(f"../stop-words/{src_language_abbreviation}_generated_stopwords_final.txt")
            sim_scores_src_to_trg, src_sentence_words = delete_unaligned_stopwords(sim_scores_src_to_trg, src_sentence_words, alignments_tuple_list_trg, alignments_tuple_list_src, LRL_STOP_WORDS)
        if filter_stopwords_trg:
            HRL_STOP_WORDS = set(stopwords.words(trg_full_language_name))
            sim_scores_trg_to_src, trg_sentence_words = delete_unaligned_stopwords(sim_scores_trg_to_src, trg_sentence_words, alignments_tuple_list_src, alignments_tuple_list_trg, HRL_STOP_WORDS)

        avg_sim_scores_src_to_trg = average_sim_scores(sim_scores_src_to_trg, window_size)
        avg_sim_scores_trg_to_src = average_sim_scores(sim_scores_trg_to_src, window_size)

        src_len = len(src_sentence_words)
        trg_len = len(trg_sentence_words)

        if min_segment_length > 1.0 and float(min_segment_length).is_integer():
            abs_min_segment_length = int(min_segment_length)
        elif 0 < min_segment_length <= 1:
            abs_min_segment_length = max(int(min(src_len,trg_len)*min_segment_length), 1)
        else:
            raise ValueError("Invalid min_segment_length. Please provide a float in (0,1] or an integer greater 1.")

        segment_list_src_indices = extract_segments(avg_sim_scores_src_to_trg,src_sentence_words,abs_min_segment_length, segment_detection_threshold)
        segment_list_trg_indices = extract_segments(avg_sim_scores_trg_to_src,trg_sentence_words,abs_min_segment_length, segment_detection_threshold)

        paired_segments = pair_segments(segment_list_src_indices, segment_list_trg_indices, alignments_tuple_list_src, 0.25)

        final_similarity_score = final_sim_score_for_sentence_pair(src_sentence_words, trg_sentence_words, sim_scores_src_to_trg, paired_segments)

        extracted_segments_temp = []
        for (src_span, trg_span) in paired_segments:
            src_seg_text = " ".join(src_sentence_words[src_span[0]:src_span[1]])
            trg_seg_text = " ".join(trg_sentence_words[trg_span[0]:trg_span[1]])
            extracted_segments_temp.append((src_seg_text, trg_seg_text))
        extracted_segments.append(extracted_segments_temp)

        if final_similarity_score >= pair_filtering_threshold:
            new_mapping_list.append((src_num, trg_num, extracted_segments_temp))
    
    with open(output_file_name,'w',encoding='utf-8') as output_file:
        for (src_num, trg_num,_) in new_mapping_list:
            output_file.write(f"src-{src_num:07}\ttrg-{trg_num:07}\n")

    with open(segments_output_file_name, 'w', encoding='utf-8') as segments_file:
        for segments in extracted_segments:
            for src_seg_text, trg_seg_text in segments:
                segments_file.write(f"src-{src_num:07}\ttrg-{trg_num:07}\t{src_seg_text}\t{trg_seg_text}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run custom SimAlign post-processing on mined sentence pairs.")
    parser.add_argument("--mapping-file", type=str, required=True, help="Path to the .pred file from the main pipeline (e.g., xlmr.hsb-de.train.sim.pred).")
    parser.add_argument("--src-file", type=str, required=True, help="Path to the original source language data file (e.g., hsb-de.train.hsb).")
    parser.add_argument("--trg-file", type=str, required=True, help="Path to the original target language data file (e.g., hsb-de.train.de).")
    parser.add_argument("--model-path", type=str, required=True, help="Path or Hugging Face name of the model (e.g., 'cis-lmu/glot500-base').")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the new, filtered mapping file.")
    parser.add_argument("--segments-output-file", type=str, required=True, help="Path to save the extracted matched segments.")
    parser.add_argument("--window-size", type=int, default=7, help="Window size for average similarity scoring (default: 7).")
    parser.add_argument("--min-segment-length", type=float, default=0.3, help="Minimum segment length as a ratio of the shorter sentence (default: 0.3).")
    parser.add_argument("--segment-threshold", type=float, default=0.3, help="Similarity threshold for a token to be part of a segment (default: 0.3).")
    parser.add_argument("--filtering-threshold", type=float, default=0.3, help="Final threshold for the segment-weighted score to keep a sentence pair (default: 0.3).")
    parser.add_argument("--matching-method", type=str, choices=['inter', 'itermax', 'mwmf'], default='inter', help="SimAlign matching method (Argmax=inter, Itermax=itermax, Match=mwmf).")
    parser.add_argument("--token-type", type=str, choices=['bpe', 'word'], default='bpe', help="SimAlign token type.")
    parser.add_argument("--mean-subtraction", action="store_true", help="Enable Mean Vector Subtraction for isotropic embedding space.")
    parser.add_argument("--filter-stopwords-src", action="store_true", help="Delete unaligned stopwords for the source language (HSB).")
    parser.add_argument("--filter-stopwords-trg", action="store_true", help="Delete unaligned stopwords for the target language (DE).")
    parser.add_argument("--exclude-punctuation", action="store_true", help="Set similarity scores of punctuation marks strictly to 0.0.")
    parser.add_argument("--dynamic-mean", action="store_true", help="Calculate the mean vector dynamically instead of loading it statically.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        mapping_file_name=args.mapping_file,
        src_file_name=args.src_file,
        trg_file_name=args.trg_file,
        model_path=args.model_path,
        output_file_name=args.output_file,
        segments_output_file_name=args.segments_output_file,
        window_size=args.window_size,
        min_segment_length=args.min_segment_length,
        segment_detection_threshold=args.segment_threshold,
        pair_filtering_threshold=args.filtering_threshold,
        matching_method=args.matching_method,
        token_type=args.token_type,
        mean_subtraction=args.mean_subtraction,
        filter_stopwords_src=args.filter_stopwords_src,
        filter_stopwords_trg=args.filter_stopwords_trg,
        exclude_punctuation=args.exclude_punctuation,
        dynamic_mean=args.dynamic_mean
    )
