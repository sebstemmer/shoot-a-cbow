# Implementation of CBOW (Word2vec) using PyTorch

This repository contains an implementation of the **Continuous Bag of Words (CBOW)** model based on the paper

[Efficient Estimation of Word Representations in Vector Space - Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean](https://arxiv.org/abs/1301.3781)

A detailed explanation and additional context can be found in a [post on my personal blog](https://sebstemmer.com/data/science/implementation/2025/05/19/implementation-of-cbow-word2vec-using-pytorch.html).

All scripts log valuable information about what is happening at runtime.

Many utility functions, the model, and the PyTorch `Dataset` are covered by **unit tests**. **Type annotations** are used throughout the repository.

**Subsampling** is also supported.

**Note:** All scripts should be executed using

```
python -m ...
```

from the **root** of the repository.

## Structure

```
├── data/
│   ├── raw/                            
│   ├── preprocessing/
│   ├── training/
│
├── preprocessing/
├── training/
├── inference/
```

## Installation

After cloning the repository, create a virtual environment:

```
python -m venv venv
```

Activate the environment:

```
source ./venv/bin/activate
```

Install the required dependencies:

```
pip install -r requirements.txt
```

## Preprocessing

Create the `./data/` folder along with the subdirectory `./data/raw/`, and place your raw data file into it.

At the top of the `preprocessing_script.py` (in `./preprocessing/`), you can configure several parameters:

```
vocab_size: int = 30000
context_window_size: int = 4
preprocessing_run_label: str = "vs_30_cw_4"
```

* `vocab_size`: The number of most frequent words to include in the vocabulary.
* `context_window_size`: Number of words to look before and after the target word. The total number of context words will be `2 * context_window_size`.
* `preprocessing_run_label`: A label identifying this preprocessing run.

To begin preprocessing:

```
python -m preprocessing.preprocessing_script
```

This will create the `./data/preprocessing/` directory if it does not exist. Within it, a subfolder named after your `preprocessing_run_label` will be created, containing:

* `preprocessed_data.pickle`
* `vocab.pickle`

The preprocessing script uses many unit-tested utility functions. Run with:

```
python -m preprocessing.run_tests_script    
```

## Training

At the top of `training_script.py` (in `./training/`), configure training parameters:

```
preprocessing_run_label: str = "vs_30_cw_4"
training_run_label: str = "vs_30_cw_4_noss"
load_from_epoch: int = -1

batch_size: int = 2048
hidden_layer_size: int = 300

num_epochs: int = 100
learning_rate: int = 5

activate_subsampling: bool = False
subsampling_t: float = 1e-3
subsampling_pow: float = 1.0
```

* `preprocessing_run_label`: refers to the label previously set in the `preprocessing_script.py`. 
* `training_run_label`: refers to the label for this training run.
* `load_from_epoch`: If set to `-1`, training starts from scratch. Otherwise, it resumes from the given epoch. This is possible because model weights are persisted after each epoch.
* `batch_size`, `hidden_layer_size`, `num_epochs`, and `learning_rate` are self-explanatory. 

If `activate_subsampling` is `True` subsampling is applied using the formula:

```
p = 1 - (subsampling_t/f)^subsampling_pow​
```

where `f` is the word frequency and `p` is the probability of discarding the word.

To start training:

```
python -m training.training_script
```

This will create the `./data/training/` directory if it does not yet exist, and a subfolder named after your `training_run_label`. This subfolder will contain:

* A file with average epoch losses.
* A `model/` subfolder with model weights saved per epoch

The model and the PyTorch `Dataset` are unit tested. These tests can be run with:

```
python -m training.run_tests_script    
```

To plot average epoch loss over epochs:

```
python -m training.plot_epoch_losses_script
```

Specify `training_run_label` and `toIdx` (plot up to this epoch index) inside the script.

To help choose suitable subsampling parameters, run

```
python -m training.analyze_subsampling_params_script
```

This plots word occurrence log-frequency sorted by index.

## Inference

Use `inference_script.py` (in `./inference/`) to analyze trained word embeddings.

You can set the following parameters at the top of the script:

```
preprocessing_run_label: str = "vs_30_cw_4"
training_run_label: str = "vs_30_cw_4_noss"
epoch: int = 19
top: int = 5
```

* `preprocessing_run_label` and `training_run_label` specify which runs to analyze.
* `epoch`: The epoch whose embeddings should be analyzed.
* `top`: Number of most similar word embedding vectors (and associated words) to return.

With the help of two utility functions you can get the most similar words to a given word (here `cat`)

```
print("similar to: cat")
print(
    inference_utils.get_similar_words(
        word="cat",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)
```

and test analogies with

```
print("input: king - man + woman, expected: queen")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="king",
        word_b="man",
        word_c="woman",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)
```

here

```
queen ≈ king - man + woman
```

Run the inference unit tests with

```
python -m inference.run_tests_script
```