from transformers import AutoTokenizer
from torchmetrics import F1Score, Accuracy
from model import CustomBertClassifier
from data_preprocessing import bert_process
import torch
import json
from tqdm import tqdm
import torch.nn as nn
import numpy as np

n_epochs = 80
class_factor = 1.4
sum_factor = 0.8
normalizing_factor = 1
accuracy_factor = 8
num_of_output = 2

def load_data(path):

    data = []
    for x in open(path, "r"):
        data.append(json.loads(x))
    return data

# checking devices
device = None
if torch.cuda.is_available():
    print("Cuda is available, using CUDA")
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    print("MacOS acceleration is available, using MPS")
    device = torch.device('mps')
else:
    print("No acceleration device detected, using CPU")
    device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('./scibert_scivocab_uncased')
# print(tokenizer(['citation']))

ACL_TRAIN_PATH = './acl-arc/train.jsonl'
# ACL_TEST_PATH = './acl-arc/test.jsonl'
ACL_TEST_PATH = './output.jsonl'
ACL_DEV_PATH = './acl-arc/dev.jsonl'

train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)

def process_intents(data):

    data = [item for item in data if item['intent'] != 'HalfExtend']

    for item in data:
        if item['intent'] == 'Extends' or item['intent'] == 'Extend':
            item['intent'] = 'Extend'
        elif item['intent'] == 'HalfExtend':
            data.remove(item)
        else:
            item['intent'] = 'NotExtend'
    return data
train_data = process_intents(train_data)
test_data = process_intents(test_data)
dev_data = process_intents(dev_data)

# print(train_data[0]['intent'])

# train_data, test_data, dev_data = train_data[:40], test_data, dev_data
bz = 290
# bertmodel_name = 'bert-large-uncased'
bertmodel_name = 'allenai/scibert_scivocab_uncased'
# bertmodel_name = 'bert-base-uncased'

if bertmodel_name == 'bert-base-uncased':
    bert_dim_size = 768
elif bertmodel_name == 'allenai/scibert_scivocab_uncased':
    bert_dim_size = 768
else:
    bert_dim_size = 1024

dev = bert_process(dev_data, batch_size=bz, pretrained_model_name=bertmodel_name)
dev_loader = dev.data_loader

test = bert_process(test_data, batch_size=bz, pretrained_model_name=bertmodel_name)
test_loader = test.data_loader

network = CustomBertClassifier(hidden_dim= 100, bert_dim_size=bert_dim_size, num_of_output=2, model_name=bertmodel_name)

def evaluate_model(network, data, data_object):
    batch_size = 0
    f1s = []
    losses = []
    accus = []

    c = {"Extend": 0, "NotExtend": 0, "HalfExtend": 0}
    p = {"Extend": 0, "NotExtend": 0, "HalfExtend": 0}

    threshold = 0

    f1 = F1Score(num_classes=3, average='macro').to(device)
    accuracy = Accuracy(num_classes=3, average='macro').to(device)

    for batch in tqdm(data):
        x, y = batch
        network.eval()
        y = y.type(torch.LongTensor)
        y = y.to(device)
        sentences, citation_idxs, mask, token_id_types = x
        sentences, citation_idxs, mask, token_id_types = sentences.to(device), citation_idxs.to(device), mask.to(device), token_id_types.to(device)
        
        output = network(sentences, citation_idxs, mask, token_id_types, device=device)
        
        probabilities = torch.softmax(output, dim=1)
        
        prob_diff = torch.abs(probabilities[:, 0] - probabilities[:, 1])
        final_predicted = torch.where(
            prob_diff < threshold,
            torch.tensor(2).to(device),
            torch.argmax(probabilities, dim=1)
        )

        for true_label in y.cpu().detach().tolist():
            if true_label == 0:
                c["NotExtend"] += 1
            elif true_label == 1:
                c["Extend"] += 1
            else:
                c["HalfExtend"] += 1

        for pr in final_predicted.cpu().detach().tolist():
            if pr == 0:
                p["NotExtend"] += 1
            elif pr == 1:
                p["Extend"] += 1
            else:
                p["HalfExtend"] += 1

        loss_fn = nn.NLLLoss()

        print(y)
        
        loss = loss_fn(output, y)  

        f1 = F1Score(num_classes=3, average='macro').to(device)
        accuracy = Accuracy(task="multiclass", num_classes=3, top_k=1).to(device)
        
        f1_score = f1(final_predicted, y)
        acc = accuracy(final_predicted, y)
        
        f1s.append(f1_score.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        accus.append(acc.cpu().detach().numpy())

    print('y_true: ', c)  
    print('y_pred: ',p)
    print('y_types: ',data_object.output_types2idx) 
    print("Loss : %f, f1 : %f, accuracy: %f" % (loss, np.mean(f1s), np.mean(accus)))
    return f1

network.load_state_dict(torch.load("./best_models/bestmodel.npy", weights_only=True))
# print("The best dev f1 is ", best_f1)
network.to(device)
network.eval()

print("dev loss and f1")
curr_f1 = evaluate_model(network, dev_loader, dev)
print("The test f1 is")
evaluate_model(network, test_loader, test)
