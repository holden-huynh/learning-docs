# 1. Transformers
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
- https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/
- [The Transformer family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)
# BERT

* architecture: transformer encoder
* objective: masked_lm & NSP
* data
    * input: 
        * 1 sequence = 1 sentence or 1 sentence pair
        * first token: [CLS] -> output representation used for classification tasks
        * [SEP] token AND add a learned __segment embedding__ to every token indicating whether it belongs to sentence A or sentence B.
        * -> input representation = token emb + segment emb + positional emb
    * wordpiece tokenizer
    * 30k vocabulary
* pre-training data:
    * BooksCorpus (800m words)
    * English Wikipedia (2500m words)
    * 16GB in total
* pre-trained models:
    * base: L=12, D=768, H=12, params=110m
    * large: L=24, D=1024, H=16, params=340m
* evaluation
# RoBERTa

* architecture: BERT
* objective: masked_lm (remove NSP)
* data
    * use natural sentences instead of pairs of segments
    * BPE tokenizer (GPT-2)
    * 50k subword units
* pre-training data:
    * BooksCorpus (800m words)
    * English Wikipedia (2500m words)
    * CC-NEWS
    * OPENWEBTEXT
    * STORIES
    * 160GB in total
* pre-trained models
# BART

* architecture: encoder-decoder
* objective: reconstruction criterion
* noising functions
    * randomly shuffle order of original sentences
    * mask arbitrary span of text by MASK
* pre-training data:
* pre-trained models
    * base: L=6 encoders & decoders
    * large: L=12 encoders & decoders
# DeBERTa

* disentangled attention
* enhanced mask decoder
# T5

* architecture: encoder-decoder
* objective: 
* pre-trained models
# GPT
# Metrics
- [Perplexity](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)
- https://towardsdatascience.com/perplexity-in-language-models-87a196019a94