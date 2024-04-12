#CS521 - Final Semester Project Work
#Prajwal Athreya Jagadish & Kavya Rama Nandana Sidda

#importing libraries
import torch
import sklearn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm.notebook import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# for GPU
model = model.to('cuda')

#PAWS dataset split into validation, train and test
train_df = pd.read_csv('/Users/prajwaljagadish/Desktop/Paraphrase-Detection/Datasource/labeled_final_train.csv')
val_df = pd.read_csv('/Users/prajwaljagadish/Desktop/Paraphrase-Detection/Datasource/labeled_final_validation.csv')
test_df = pd.read_csv('/Users/prajwaljagadish/Desktop/Paraphrase-Detection/Datasource/labeled_final_test.csv')

# Check the dataframe
print(train_df.head())
print(val_df.head())
print(test_df.head())

# We are defining a dataset class

class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, sentences1, sentences2, labels):
        self.tokenizer = tokenizer
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokenized_input = self.tokenizer(self.sentences1[idx], self.sentences2[idx], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized_input['input_ids'].squeeze(0)
        attention_mask = tokenized_input['attention_mask'].squeeze(0)
        token_type_ids = tokenized_input['token_type_ids'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, label
    
#tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ParaphraseDataset(tokenizer, train_df['sentence1'].tolist(), train_df['sentence2'].tolist(), train_df['label'].tolist())
val_dataset = ParaphraseDataset(tokenizer, val_df['sentence1'].tolist(), val_df['sentence2'].tolist(), val_df['label'].tolist())



#Dataloader and initializing our model
batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

#Model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


#First we evaluate our trained model on validation dataset
model.eval()  # Evaluation mode
predictions, true_labels = [], []

for batch in val_dataloader:
    batch = tuple(t.to(device) for t in batch)  
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=-1).tolist())
    true_labels.extend(batch[3].tolist())

accuracy = accuracy_score(true_labels, predictions)
print(f'Validation Accuracy: {accuracy}')


#evaluation on test dataset
test_dataset = ParaphraseDataset(tokenizer, test_df['sentence1'].tolist(), test_df['sentence2'].tolist(), test_df['label'].tolist())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
test_predictions, test_true_labels = [], []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    test_predictions.extend(torch.argmax(logits, dim=-1).tolist())
    test_true_labels.extend(batch[3].tolist())

test_accuracy = accuracy_score(test_true_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy}')

#creating the path to save the model
model_path = "/content/bert_paraphrase_detection"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

#inference
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)

#testing our model with few example sentences
def predict_paraphrase(sentence1, sentence2):
    model.eval()
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Paraphrase" if prediction == 1 else "Not Paraphrase"

#example1
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast, dark-colored fox leaps above a slow-moving dog."
print(predict_paraphrase(sentence1, sentence2))

#example2
sentence3 = "My name is Sam and I am a good boy."
sentence4 = "I'm Sam and I consider myself to be a well-behaved young man.."
print(predict_paraphrase(sentence3, sentence4))

#example3
sentence5 = "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:"
sentence6 = "BERT is a transformer-based model that was pre-trained using a vast collection of English text in a self-supervised manner. This entails that it underwent pre-training solely on raw texts without any human-annotated labels, allowing it to leverage a vast array of publicly accessible data through an automated process that creates inputs and labels directly from the texts. Specifically, it was designed with two primary pre-training objectives."
print(predict_paraphrase(sentence5, sentence6))

#example4
sentence7 = "The conclusion that explicit part descriptions enhance abstract reasoning in both humans and models is well-supported by the data."
sentence8 = "The data strongly supports the idea that detailed descriptions of parts improve abstract reasoning in both humans and computational models."
print(predict_paraphrase(sentence7, sentence8))

