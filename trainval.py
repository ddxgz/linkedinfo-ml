import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertPreTrainedModel, AutoTokenizer, AutoModel,
                          BertForSequenceClassification, AdamW, BertModel,
                          BertTokenizer, BertConfig, get_linear_schedule_with_warmup)

import dataset
from mltb.mltb import nlp as mnlp
from mltb.mltb import bert as mbert
from mltb.mltb.metrics import best_prec_score, classification_report_avg


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_loader, optimizer, scheduler=None):
    model.train()

    total_train_loss = 0

    for step, (input_ids, masks, labels) in enumerate(train_loader):
        input_ids, masks, labels = input_ids.to(
            DEVICE), masks.to(DEVICE), labels.to(DEVICE)

        model.zero_grad()
        loss, logits = model(input_ids, token_type_ids=None,
                             attention_mask=masks, labels=labels)

        total_train_loss += loss.item()
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print("Train loss: {0:.2f}".format(avg_train_loss))


def val(model, test_loader):
    model.eval()

    val_loss = 0

    y_pred, y_true = [], []
    # Evaluate data for one epoch
    for (input_ids, masks, labels) in test_loader:

        input_ids, masks, labels = input_ids.to(
            DEVICE), masks.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            (loss, logits) = model(input_ids,
                                   token_type_ids=None,
                                   attention_mask=masks,
                                   labels=labels)

        val_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        y_pred += logits.tolist()
        y_true += label_ids.tolist()

    bes_val_prec, bes_val_prec_thr = best_prec_score(
        np.array(y_true), np.array(y_pred))
    y_predicted = (np.array(y_pred) > 0.5)

    avg_val_loss = val_loss / len(test_loader)

    print("Val loss: {0:.2f}".format(avg_val_loss))
    print("best prec: {0:.4f}, thr: {1}".format(
        bes_val_prec, bes_val_prec_thr))
    print(classification_report_avg(y_true, y_predicted))


def add_bert_vocab(ds, col_text, tfidf_param: dict,
                   model_name: str = "google/bert_uncased_L-4_H-256_A-4"):
    tag_terms = dataset.tag_terms(ds)

    if '' in tag_terms:
        tag_terms.remove('')

    top_terms = mnlp.top_tfidf_terms(ds.data[col_text], tfidf_param, top_n=300)

    unique_terms = set(tag_terms + top_terms['term'].tolist())
    unique_terms = pd.DataFrame(unique_terms, columns=['term'])

    model_name = mbert.download_once_pretrained_transformers(
        model_name)

    bert_vocab = mbert.get_bert_tokens(model_name)

    new_tokens = unique_terms[~unique_terms['term'].isin(
        bert_vocab)]['term'].tolist()

    for i, line in enumerate(bert_vocab):
        if len(new_tokens) == 0:
            break
        if line.startswith('[unused'):
            bert_vocab[i] = new_tokens.pop()

    mbert.save_bert_vocab(bert_vocab, model_name=model_name)
    return model_name, bert_vocab


def fine_tune_bert(model_param: dict, train_features, train_labels, test_features,
                   test_labels, col_text, exp_name, model=None):
    n_classes = model_param['n_classes']
    model_name: int = model_param['model_name']
    epochs: int = model_param['epochs']
    batch_size: int = model_param['batch_size']
    lr = model_param.get('learning_rate', 5e-5)
    eps = model_param.get('eps', 1e-8)

    if not model:
        model = mbert.BertForSequenceMultiLabelClassification.from_pretrained(
            model_name,
            num_labels=n_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

    model.to(DEVICE)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.1,
         },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

    tokenizer, model_notuse = mbert.get_tokenizer_model(model_name)

    input_ids, attention_mask = mbert.bert_tokenize(
        tokenizer, train_features, col_text=col_text)
    input_ids_test, attention_mask_test = mbert.bert_tokenize(
        tokenizer, test_features, col_text=col_text)

    train_set = torch.utils.data.TensorDataset(
        input_ids, attention_mask, torch.Tensor(train_labels))
    test_set = torch.utils.data.TensorDataset(
        input_ids_test, attention_mask_test, torch.Tensor(test_labels))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=RandomSampler(train_set))
    test_loader = torch.utils.data.DataLoader(
        test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    for ep in range(epochs):
        print(f'-------------- Epoch: {ep+1}/{epochs} --------------')
        train(model, train_loader, optimizer, scheduler)
        val(model, test_loader)

    print('-------------- Completed --------------')

    # exp_name = 'bert_finetuned_tagthr_20_new_vocab_24ep_slr_chop'

    output_dir = f"./data/models/{exp_name}/"

    from transformers import WEIGHTS_NAME, CONFIG_NAME, BertTokenizer

    def save_model_tuned(model, tokenizer, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

        # If we have a distributed model, save only the encapsulated model
        # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)

    save_model_tuned(model, tokenizer, output_dir)

    print('------------- Saved model --------------')
    print(output_dir)
    return output_dir
