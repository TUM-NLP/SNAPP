# SNAPP
This repository contains the code for the article with the title "SNAPP: Segment-Level Neural Alignment Post-Processing for Low-Resource Parallel Sentence Mining".




## Setup
Before running the scripts, please ensure that you have [Conda](https://docs.conda.io/en/latest/) installed on your system or cluster. 


### Conda Environment
To ensure full reproducibility, all necessary dependencies (including deep learning frameworks, NLP libraries, and specific versions) are specified in the `environment.yml` file. 

You can recreate and activate the environment by executing the following commands from the root directory of this repository:

```bash
conda env create -f environment.yml
conda activate SNAPP_env
```

### Datasets

Please insert your datasets in the `post-processing/data/` folder. 
For each language pair, the dataset folder and file structure must be the same as shown by [BELOPSEM](https://github.com/shuokabe/Belopsem/tree/main/Belopsem_1) (e.g., see `chv-ru/` folder).  
If you want to add new languge pairs, simply insert new folders named `lang1-lang2/` and adjust the paths within the execution scripts accordingly.

### Initial Parallel Sentence Mining

Please execute a parallel sentence mining pipeline (e.g., the [PaSeMiLL](https://github.com/shuokabe/PaSeMiLL.git) mining pipeline) to retrieve an initial list of candidate pairs. 
Make the list of candidate pairs available in a sub-folder of `post-processing/results/`. 
Each line of the file must contain exactly one mapping in the following format: **src-xxxxxxx trg-xxxxxxx**  
The numbers behind src and trg are used to extract the respective sentences from the `data/` folder.
Our filtering pipeline will then access the mapping file for the post-processing steps. 

### Mean-Vector Computation

If you want to execute the pipeline with a static mean-vector subtraction, please compute the mean-vector separately, prior to execution, by adapting and executing the code provided in `post-processing/example_execution/mean_vector_computation.sh`.  
This execution will compute the mean-vector on the train data, which is contained in your `data/` folder (i.e., `post-processing/data/lang1-lang2/lang1-lang2.train.lang1` and `post-processing/data/lang1-lang2/lang1-lang2.train.lang2`).  

For the dynamic mean-vector configuration, there is nothing to note, prior to execution.


## Post-Processing Execution

You can find example executions for the post-processing in `post-processing/example_execution/`. 
Please note, that the sbatch scripts are adapted to the LRZ AI Cluster. 
You can adjust the `execute_post-processing_simalign.py` to your desired configuration.  
Please ensure, that there are initial mining results available in the corresponding results folders. 
For instance, you can use the [PaSeMiLL](https://github.com/shuokabe/PaSeMiLL.git) mining pipeline to compute candidate pairs. 


## MT Model Fine-Tuning 

In `MT/sample_execution/` you can find example scripts for executung the fine-tuning and the translation. 
To enable the fine-tuning of the MT model, please provide the sentence pairs in a jsonl file within the respective `MT/data/lang1-lang2` folder. 
Examples are provided in `MT/data/chv-ru/train_data.jsonl` and `MT/data/hsb-de/train_data.jsonl`. 
These files constitute the remaining candidate pairs after applying our post-processing pipeline to the [PaSeMiLL](https://github.com/shuokabe/PaSeMiLL.git) results, which are based on the [BELOPSEM](https://github.com/shuokabe/Belopsem/tree/main/Belopsem_1) dataset. 
Prior to the translation step, please insert the test sentences in the respective `MT/data/lang1-lang2` folder.



## Citations and Licences

### Data

The `MT/data/` folder contains the mining and filtering results (prepared for the fine-tuning step) for the language pairs Upper Sorbian-German (`hsb-de`) and Chuvash-Russian (`chv-ru`), which are based on the dataset originally introduced in the [Belopsem](https://github.com/shuokabe/Belopsem/tree/main/Belopsem_1) repository. 
In accordance with the original repository, this datasets are licensed under the **CC BY-NC-SA 4.0** (hsb-de) and under **CC BY** (chv-ru) license. 

If you use this dataset, please cite the following paper:
```bibtex
@inproceedings{okabe-etal-2025-improving,
    title = "Improving Parallel Sentence Mining for Low-Resource and Endangered Languages",
    author = {Okabe, Shu  and
      H{\"a}mmerl, Katharina  and
      Fraser, Alexander},
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-short.17/",
    doi = "10.18653/v1/2025.acl-short.17",
    pages = "196--205",
    ISBN = "979-8-89176-252-7",
}
```


### Code and Methodology

Please note that the `post-processing/code/` folder contains code from [PaSeMiLL](https://github.com/shuokabe/PaSeMiLL) in order to make the performance evaluation of the post-processing easier. 
Please cite usage of this code according to the [PaSeMiLL README](https://github.com/shuokabe/PaSeMiLL/blob/main/README.md) file.

Our post-processing pipeline is an independent partial re-implementation based on the algorithm description of [UnsupPSE](https://aclanthology.org/P19-1118.pdf):

```bibtex
@inproceedings{hangya-fraser-2019-unsupervised,
    title = "Unsupervised Parallel Sentence Extraction with Parallel Segment Detection Helps Machine Translation",
    author = "Hangya, Viktor  and
      Fraser, Alexander",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\`a}rquez, Llu{\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1118/",
    doi = "10.18653/v1/P19-1118",
    pages = "1224--1234",
    abstract = "Mining parallel sentences from comparable corpora is important. Most previous work relies on supervised systems, which are trained on parallel data, thus their applicability is problematic in low-resource scenarios. Recent developments in building unsupervised bilingual word embeddings made it possible to mine parallel sentences based on cosine similarities of source and target language words. We show that relying only on this information is not enough, since sentences often have similar words but different meanings. We detect continuous parallel segments in sentence pair candidates and rely on them when mining parallel sentences. We show better mining accuracy on three language pairs in a standard shared task on artificial data. We also provide the first experiments showing that parallel sentences mined from real life sources improve unsupervised MT. Our code is available, we hope it will be used to support low-resource MT research."
}
```

### Language Models and Alignment Tools

This repository relies on several external tools and pre-trained language models via the Hugging Face transformers library. Please consider citing the respective authors if you use them:

#### Language Models

- Glot500: Ayyoob Imani, Peiqin Lin, Amir Hossein Kargaran, Silvia Severini, Masoud Jalili Sabet, Nora Kassner, Chunlan Ma, Helmut Schmid, André Martins, François Yvon, and Hinrich Schütze. 2023. Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1082–1117, Toronto, Canada. Association for Computational Linguistics. [Link](https://aclanthology.org/2023.acl-long.61/)

- XLM-R: Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised Cross-lingual Representation Learning at Scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451, Online. Association for Computational Linguistics. [Link](https://aclanthology.org/2020.acl-main.747/)

#### Word alignment

- SimAlign: Masoud Jalili Sabet, Philipp Dufter, François Yvon, and Hinrich Schütze. 2020. SimAlign: High Quality Word Alignments Without Parallel Training Data Using Static and Contextualized Embeddings. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1627–1643, Online. Association for Computational Linguistics. [Link](https://aclanthology.org/2020.findings-emnlp.147/)  
