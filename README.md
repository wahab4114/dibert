# DIBERT
DIBERT stands for Dependency Injected Bidirectional Encoder Representations from Transformers

DIBERT is a variation of the BERT-Base model with an additional objective called Parent Prediction (PP). PP utilizes dependency parse tree of a sentence
and ask model to predict the parent of each input token. Adding the additional syntactic information in the form of PP objective in BERT model improves the
performance of the model. Results show that DIBERT outperforms BERT model.

* model.py includes the architecture of DIBERT model
* datapreparation.py includes the creation of labels for MLM, NSP and PP objectives on WikiText-103 dataset and stores the data
* dataset.py creates dataset class for the prepared wikitext-103 dataset
* dibert/Downstreamtask/ contains the tasks on which finetuning is done
  * Each task has been fine-tuned on both DIBERT and BERT model
  * dibert/Downstreamtask/QNLI/results/full_text/params/training_logs.txt file contains the averaged result of both models on QNLI dataset and this pattern is same for other tasks as well
  * dibert/Downstreamtask/results.txt contains the result on every task
* dibert.yml file can be used for installing the dependencies needed to run the code
* utils.py contains the helper functions
