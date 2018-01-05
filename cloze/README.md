# Multi Word Cloze

RNN and GRU cell implementation in PyTorch, with their bidirectional language model.
The problem is to complete the missing words in a sentence (Cloze)
More info: http://mt-class.org/jhu/nn_hw4.html
Specify optimization method and training arguments in the training script.

## Running

1. Modify and run `train.sh` to generate the model file. Alternatively run `python train.py -h` to see all options.
2. Similary run the `ans.py` to generate the completions using the trained model which is passed through an argument.

## Training and Performance

Refer `training-report.pdf`
Final error with a single layered Bi-directional GRU language model is 0.396

