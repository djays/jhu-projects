# Neural Machine Translation

German to English neural machine translation system, implemented in PyTorch.  
It is a Sequence to Sequence model with Luong local Attention with Teacher enforcing training.
Done as part of Johns Hopkin's Machine Translation class.
More info: http://mt-class.org/jhu/nn_hw5.html  
An alternative task included converting graphemes to phonemes. (cmudict files)

## Architecture ##
The GRU is used as the recurrent unit as memory consumption is lower.  
Encoder:  Bi-directional GRU, whose outputs from either directions are  concatenated and fed into the decoder layer.  

Decoder:  Single GRU with Attention

## Running

1. Modify and run `train.sh` to generate the model file. Alternatively run `python train.py -h` to see all options.
2. Similary run the `Translator.py` to generate the translations using the trained model which is passed through an argument.

## Results
WER Accuracy 0.96

