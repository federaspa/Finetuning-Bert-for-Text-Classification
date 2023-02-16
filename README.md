# Finetune Bert for Text Classification

In this notebook we will finetune a pretrained transformer to perform a NLP task. 

More precisely we will finetune a BERT model to perform text classification on the The Corpus of Linguistic Acceptability (CoLA) dataset from the GLUE benchmark.
This dataset consists of thousands of well-formed english sentences which have to be distinguished from the unacceptable ones. 

# Lbraries

- datasets
- scikit-learn
- torchmetrics
- scipy
- torch
- transformers
- torchtext
- setuptools
- tensorboard

# [The CoLA Dataset](https://nyu-mll.github.io/CoLA/)

The Corpus of Linguistic Acceptability dataset contains sentences annotated for acceptability (grammaticality) by their original authors. The public version contains 9594 sentences belonging to training and testing sets, and excludes 1063 sentences belonging to a held out test set.

# Preprocessing

Preprocessing the data involves:
*   Splitting the words in the text into *tokens*
*   Assigning a unique number to each token
*   Zero-Padding all the sentences to a fixed length
*   Defining the masks associated to valid tokens in the zero padded sentences

All of this is automatically done by the `AutoTokenizer`from the `transformer` library. Almost each transformer model has chosen a different tokenization strategy. For this reason the `AutoTokenizer` needs to know which model we are going to use (`bert-base-uncased`). We choose max_sequence_length = 128.

# Model

The pretrained model we use is [`bert-base-uncased`](https://huggingface.co/bert-base-uncased). Introduced in 2018, BERT is a transformers model pretrained on a large corpus of english text in a semi-supervised fashion. It's uncased version makes no distinction between cased and uncased words.

The model's configuration is set up with  `AutoConfig.from_pretrained` and the model is created automatically done by the `AutoModelForSequenceClassification` class.

# Training
We write a single `epoch` function which will manage both the training epoch and the validation one according to the `mode` used. The main difference consists in the optimization step that is performed only in *train* mode.
It is a more elegant and compact way of implementing the model's loop as per [huggingface documentation](https://huggingface.co/docs/transformers/training#train-in-native-pytorch
)


We also use:
*   [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) logger to check the loss and the accuracy of the model both at training and at test time.
*   A [tqdm](https://tqdm.github.io/) progress bar to check the evolution of the training process and estimate the total required time. 

# Matthews Correlation Coeffictient (MCC)
As a metric we will use the MCC, which is a good metric for binary classification tasks.

${\displaystyle {\text{MCC}}={\frac {{\mathit {TP}}\times {\mathit {TN}}-{\mathit {FP}}\times {\mathit {FN}}}{\sqrt {({\mathit {TP}}+{\mathit {FP}})({\mathit {TP}}+{\mathit {FN}})({\mathit {TN}}+{\mathit {FP}})({\mathit {TN}}+{\mathit {FN}})}}}}$ 

The coefficient takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. It returns a value between −1 and +1. 

# Adjusting dropout and adding learning rate scheduler

Attempting to reduce the overfitting, we try to both increase the model dropout and add weight decay.

1. The dropout probabilities can be adjusted in the [model configuration](https://huggingface.co/transformers/v3.0.2/model_doc/bert.html). There are two options, both defaulted at 0.1:
    * **hidden_dropout_prob** controls the dropout probability of all fully connected layers in the embeddings, encoder, and pooler 
    * **attention_probs_dropout_prob** controls dropout ratio for the attention
    probabilities
2. The learning rate scheduler used is a linear scheduler updated at each training step, as per [huggingface documentation](https://huggingface.co/docs/transformers/training#optimizer-and-learning-rate-scheduler)

# Conclusions:

The model already starts overfitting after a couple of epochs, most likely due to the huge number of parameters in BERT and the small size of the CoLA dataset. Adding a learning rate scheduler boosts the model's performance by roughly 2%, while increasing the dropout seems to have little positive impact.
Overall, we manage to get a Matthew Correlation Coefficient between 0.58 and 0.62.
