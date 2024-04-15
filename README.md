# CS 521 - Statistical natural language Processing : Final Semester Project (Spring 2024)
Paraphrase-Detection using Transformer architecture based language model. 
by - Prajwal Athreya Jagadish and Kavya Rama Nandana Sidda (images to be added here after description)

## Introduction
Accurate paraphrasing identification is crucial for Natural Language Processing (NLP) applications, which include machine translation, question-answering systems, and plagiarism detection. Finding paraphrases or writings with the same content but conveyed differently may be difficult since human language is complex and varied. NLP's traditional models have frequently failed to capture the subtleties and complexity of paraphrase detection. Taking note of these difficulties, this study aims to apply the state-of-the-art NLP model Bidirectional Encoder Representations from Transformers (BERT) to improve the effectiveness of paraphrase identification. The design of BERT offers a strong foundation for comprehending the contextual relationships found in the text since it uses deep learning techniques to analyze words in relation to every other word in a sentence (as opposed to one-directional reading). We aim to investigate how BERT's detection performance of paraphrases might be enhanced by fine-tuning it on a carefully selected dataset with various paraphrase variants. We want to raise the bar for paraphrase detection accuracy by combining BERT's sophisticated features with a focused training strategy, opening the door to more dependable applications across various NLP areas.

## Environment setup
We set up our development environment to ensure all the libraries and dependencies were installed. This incorporated PyTorch for deep learning, implementing the BERT model, which included the Transformers library, and other necessary libraries, such as Scikit-learn and Pandas.

## Dataset used
We used the Paraphrase Adversaries from Word Scrambling (PAWS) dataset for this study, as detailed in the arXiv:1904.01130 publication. A thorough environment for assessing our optimized BERT model, PAWS is painstakingly created to evaluate the resilience of paraphrase detection techniques against intricate sentence structures and situations. The sentences in the sample are paired off and assessed for paraphrase. If two sentences are paraphrases of one another, they are labeled as "Paraphrased"; if not, they are labeled as "Not paraphrased." With 100,000 sentence pairings in the PAWS dataset, a wealth of information is available for training and validation. 

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
