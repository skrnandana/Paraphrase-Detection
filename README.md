# CS 521 - Statistical natural language Processing : Final Semester Project (Spring 2024)

# Paraphrase-Detection using Transformer architecture based language model. 
# by - Prajwal Athreya Jagadish and Kavya Rama Nandana Sidda (images to be added here after description)

## Introduction
In the field of Natural Language Processing (NLP), the ability to accurately detect paraphrases is essential for applications such as plagiarism detection, question answering systems, and machine translation. The task of identifying paraphrases—texts that convey the same meaning but are expressed differently—is challenging due to the intricate and diverse nature of human language. Traditional models in NLP have often fallen short in effectively handling the nuances and complexities of paraphrase detection. Recognizing these challenges, this project focuses on enhancing the efficacy of paraphrase detection through the utilization of Bidirectional Encoder Representations from Transformers (BERT), a state-of-the-art model in NLP.

BERT's architecture, which leverages deep learning techniques to process words in relation to all the other words in a sentence (as opposed to one-directional reading), provides a robust framework for understanding the contextual relationships within text. Our study aims to explore how fine-tuning BERT on a carefully curated dataset containing a wide range of paraphrase variations can improve its performance in detecting paraphrases. By integrating BERT's advanced capabilities with a targeted training approach, we seek to set a new standard for accuracy in paraphrase detection, paving the way for more reliable applications in various domains of NLP.

## Environment setup

## Dataset used
For this research project, we utilized the Paraphrase Adversaries from Word Scrambling (PAWS) dataset, as described in the arXiv:1904.01130 paper. PAWS is meticulously designed to test the robustness of paraphrase detection models against complex sentence structures and contexts, thereby providing a comprehensive platform for evaluating our fine-tuned BERT model.

The dataset consists of pairs of sentences, each evaluated for paraphrasing. Each sentence pair is marked as either "Paraphrased" if the sentences are paraphrases of each other, or "Not paraphrased" if they are not. The PAWS dataset includes 100,000 sentence pairs, affording a rich set of data for both training and validation purposes.

## Custom dataset class

## Tokenization


## Dataloader for sequence classification



## Training Loop



## Evaluation on Validation dataset



## Evaluation on Test dataset



## Saving the model by giving a path



## Model for inference


## Fine-tuned model and example cases



## Analytics and Evaluation

### Output:
<p align="center">
<img src="Analytics/losses.png" />
</p>
 
## Conclusion
