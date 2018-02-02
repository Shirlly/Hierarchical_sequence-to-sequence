# Hierarchical_sequence-to-sequence

An implementation of hierarchical structure for sequence-to-sequence model. (https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Encoder:
1) words to sentence
2) setences to document

Decoder:
sentence

```
CUDA_VISIBLE_DEVICES=gpu_id python main.py --options option_values
```


## Required packages:
* Python 2.7
* Tensorflow 1.3
* Tensorgraph
* sklearn
* hdbscan
