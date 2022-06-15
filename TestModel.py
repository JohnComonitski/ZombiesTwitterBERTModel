import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import TextClassificationPipeline
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

#Prep Modal
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=6,
    output_attentions=False,
    output_hidden_states=False
)
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
model.to(device)
#model_name="BERT Pretrained"
model_name = "BERT_ft_epoch4.model"
path = "Models/"+ model_name
model.load_state_dict(
    torch.load(
        path,
        map_location=torch.device('cpu')
))
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

#Get All Labels
df2 = pd.read_csv(
    './datasets/smile-annotations-final.csv',
    names=['id', 'text', 'category']
)
df2.set_index('id', inplace=True)
df2 = df2[~df2.category.str.contains('\|')]
df2 = df2[df2.category != 'nocode']
possible_labels = df2.category.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

#Get tweets
df = pd.read_csv(
    './datasets/tweets-with-irrelevant.csv',
    names=[ 'id','text', 'tweetid', 'category']
)
df.set_index('id', inplace=True)
df.loc[df["category"] == 0, "category"] = "angry"
df.loc[df["category"] == 1, "category"] = "happy"
df.loc[df["category"] == 2, "category"] = "not-relevant"

print("------------------------------------------------------")
print("Testing Model: {}".format(model_name))
#Testing Tweets
correct = 0
total_tweets = 1000
for i in tqdm (range (total_tweets), desc="Testing"):
    tweet = df.iloc[i]
    res = pipe(tweet["text"])
    sentiment = max(range(len(res[0])), key=lambda index: res[0][index]['score'])

    if(possible_labels[sentiment] == tweet["category"]):
        correct = correct + 1

print("Score: {}/{}".format(correct,total_tweets))
