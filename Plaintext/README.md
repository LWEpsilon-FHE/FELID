# The continued-pretraining method for retrieving a kernel attention model.
## 0. Installation

```
cd Plaintext
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 1. Download the pretrained BERT modeal
google/bert_uncased_L-2_H-768_A-12:
https://huggingface.co/google/bert_uncased_L-2_H-768_A-12

## 2. Start the training

```
bash run.sh
```