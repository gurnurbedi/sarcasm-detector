
# Sarcasm Detector using LLM Models

This project detects sarcasm using AI model (LSTM + Attention) and LLM models (RoBERTa, XLNET). These NLP models have been fine tuned specifically to detect sarcasm in the user entered entered sentences.

While Django has been used for the integration of Python and the UI. 

Bootstrap 4.3.1 has been used for the creation of UI.

## Dataset

The Dataset used is a combination of the following
 - [Hugging Face Dataset](https://huggingface.co/datasets/siddhant4583agarwal/sarcasm-detection-dataset)
 - [GitHub Dataset](https://github.com/AmirAbaskohi/SemEval2022-Task6-Sarcasm-Detection/tree/main/Data/Cleaned%20Datasets)
 

## Run Locally

Clone the project

```bash
  git clone https://https://github.com/gurnurbedi/sarcasm-detector-LLM-model.git
```

Install dependencies

```bash
bert-embedding==1.0.1
numpy==1.19.5
tqdm==4.62.3
pandas==1.3.1
mxnet-cu101==1.9.0 
sentence-transformers==2.2.0
tensorflow==2.8.0 
tensorflow-gpu==2.8.0 
tensorflow-hub==0.12.0 
bert==2.2.0
torch==1.10.2 
pytorch-lightning==1.5.10
transformers==4.16.2
scikit-learn==0.24.2
ipywidgets==7.6.5
bertviz==1.3.0
json5==0.9.6
mxnet==1.9.0 
nltk==3.7 
gensim==4.1.2
sentencepiece==0.1.96 
fairseq==0.10.2 
datasets==1.18.3
collections2==0.3.0 
```

Start the Django server
```bash
python manage.py runserver
```

## Acknowledgements

- [Models](https://github.com/amirabaskohi/semeval2022-task6-sarcasm-detection)


