# A Tale of Two Linkings: Dynamically Gating between Schema Linking and Structural Linking for Text-to-SQL Parsing

This repo implements the dynamic gating mechanism described in our [COLING 2020 paper](https://arxiv.org/abs/2009.14809) on top of a graph neural network-based Text-to-SQL parser.
The implementation is built on top of this [repository](https://github.com/benbogin/spider-schema-gnn-global).



## Install & Configure

1. Install pytorch version 1.5.0 that fits your CUDA version 

2. Install the rest of required packages
    ```
    pip install -r requirements.txt
    ```
    
3. Run this command to install NLTK punkt.
```
python -c "import nltk; nltk.download('punkt')"
```

4. Download the dataset from the [official Spider dataset website](https://yale-lily.github.io/spider)

5. Edit the config file `train_configs/defaults.jsonnet` to update the location of the dataset:
```
local dataset_path = "dataset/";
```

6. **Before preprocessing the dataset, modify [two](https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/data/fields/knowledge_graph_field.py#L99) [lines](https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/data/fields/knowledge_graph_field.py#L109) in allennlp lib, to replace `self._tokenizer` with `_tokenizer`. This change greatly reduces the size of cache data and memory usage.** Also, change the number of processes in `dataset_readers/spider.py` according to your machine setting.

## Training and Inference

Run the following command to train a new model with or without the dynamic gating mechanism.
```
python run.py [--gated]
```

First time loading of the dataset might take a while (a few hours) since the model first loads values from SQL tables and calculates similarity features with the relevant question. It will then be cached for subsequent runs.

Run the following command to generate model predictions.
```
python run.py <path> --mode eval
```

The predictions can be further evaluated by the official [evaluation scripts](https://github.com/taoyds/spider) of the Spider dataset.

## Debugging

Refer to [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/using_a_debugger.md), use `run.py` for debugging.

## BibTeX

```tex
@inproceedings{chen2020tale,
    title={A Tale of Two Linkings: Dynamically Gating between Schema Linking and Structural Linking for Text-to-SQL Parsing},
    author={Sanxing Chen and Aidan San and Xiaodong Liu and Yangfeng Ji},
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    year={2020}
}
```