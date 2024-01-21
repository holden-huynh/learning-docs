* GLUE
* SuperGLUE
* Squadv1
* Squadv2
* LAMBADA
    * predict the last word of sentences
* Winograd
    * match pronoun with right word
* StoryCloze
* WMT
---
# GLUE
General Language Understanding Evaluation
* 9 sentence- or sentence-pair language understanding tasks
* built on established existing datasets and selected to cover a diverse range of dataset sizes, text genres, and degrees of difficulty
* A diagnostic dataset designed to evaluate and analyze model performance with respect to a wide range of linguistic phenomena found in natural language
## Tasks
## CoLA
The Corpus of Linguistic Acceptability: https://nyu-mll.github.io/CoLA/
* total 10657 sentences from 23 linguistics publications, expertly annotated for acceptability (grammaticality) by their original authors
* 9594 sentences belonging to training and development sets
    * 8551 train
* 1063 sentences belonging to a held out test set
## QNLI
* See SQuAD2.0
## Stanford Sentiment Treebank
* sentiment analysis
## Microsoft Research Paraphrase Corpus	
## Semantic Textual Similarity Benchmark	
* STS: http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
## MultiNLI
Multi-Genre Natural Language Inference
* https://cims.nyu.edu/~sbowman/multinli/
* 433k sentence pairs annotated with textual entailment information
* See SNLI: https://nlp.stanford.edu/projects/snli/
    * 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral.
## Winograd NLI
The Winograd Schema Challenge: https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html
---
# SuperGLUE
https://super.gluebenchmark.com/

#### Tasks:
https://super.gluebenchmark.com/tasks

#### Diagnostic dataset:
https://gluebenchmark.com/diagnostics
## MultiRC
Multi-Sentence Reading Comprehension	
## WSC
The Winograd Schema Challenge: https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html (same as GLUE)
---
# SQUAD
Stanford Question Answering Dataset

answer the question by extracting the relevant span from the context
---
# Grammar Error Checking & Correction
## Datasets

* http://nlpprogress.com/english/grammatical_error_correction.html
* https://paperswithcode.com/task/grammatical-error-correction
* https://ai.googleblog.com/2021/08/the-c4200m-synthetic-dataset-for.html
* W&I - LOCNESS
    * [WI-LOCNESS Dataset | Papers With Code](https://paperswithcode.com/dataset/locness-corpus)
    * https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
    * W & I: 3,600 annotated submissions to W&I across 3 different CEFR levels: A (beginner), B (intermediate), C (advanced)
    * LOCNESS: 100 annotated native (N) essays from LOCNESS
* BEA 2019: https://www.cl.cam.ac.uk/research/nl/bea2019st/
* CoNLL 2014: [Shared Task: Grammatical Error Correction](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
    * Annotated test data: https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
* NUCLE: https://www.comp.nus.edu.sg/~nlp/corpora.html
    * The corpus consists of about 1,400 essays written by university students at the National University of Singapore on a wide range of topics, such as environmental pollution, healthcare, etc. It contains over one million words which are completely annotated with error tags and corrections. All annotations have been performed by professional English instructors at the NUS CELC.
* Ten Sets of Multiply Annotated Essays for Grammatical Error Correction
    * Ten native speakers were each asked to correct 50 essays (~600 words per essay) written by non-native English speakers for grammatial correctness. Each edit was also classified according to the error classification scheme used in the CoNLL-2014 shared task. This corpus was used in the ACL 2015 paper of Christopher Bryant and Hwee Tou Ng, titled "How Far are We from Fully Automatic High Quality Grammatical Error Correction?"
* Lang-8
    * Lang-8 is an online language learning website which encourages users to correct each other's grammar. The Lang-8 Corpus of Learner English is a somewhat-clean, English subsection of this website.
* FCE: https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
    * The First Certificate in English (FCE) corpus is a subset of the Cambridge Learner Corpus (CLC) that contains 1,244 written answers to FCE exam questions
* CoLA (GLUE)
* Falko-MERLIN (German)
* Synthetic dataset:

## Tools:
* https://github.com/chrisjbryant/errant