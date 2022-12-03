import argparse
import os
import json

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, \
    Trainer, Seq2SeqTrainingArguments

from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t2t', action='store_true')
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--mode', type=str, help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir', type=str, help='root directory to save model & summaries')
    parser.add_argument('--expt_name', type=str, help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--data_dir', type=str, help='directory containing train, eval, test files')
    parser.add_argument('--model', type=str, help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--lr', type=float, help='learning rate', default=2e-5)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--acc_step', type=int, help='gradient accumulation steps', default=1)

    args = parser.parse_args()

    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, 'train.json')
    val_file = os.path.join(data_dir, 'val.json')
    test_file = os.path.join(data_dir, 'test.json')
    output_dir = args.expt_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.mode == 'train':
        with open(train_file, 'r') as f:
            train_data = json.loads(f.read())
            if args.t2t:
                train_instances = [t2t_preprocess_data(x, tokenizer) for x in train_data]
            else:
                train_instances = [preprocess_data(x, tokenizer) for x in train_data]
        with open(val_file, 'r') as f:
            val_data = json.loads(f.read())
            if args.t2t:
                val_instances = [t2t_preprocess_data(x, tokenizer) for x in val_data]
            else:
                val_instances = [preprocess_data(x, tokenizer) for x in val_data]
        with open(test_file, 'r') as f:
            test_data = json.loads(f.read())
            if args.t2t:
                test_instances = [t2t_preprocess_data(x, tokenizer) for x in test_data]
            else:
                test_instances = [preprocess_data(x, tokenizer) for x in test_data]

        if not args.t2t:
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        name = f"{args.expt_name}_bs_{batch_size}_lr_{lr}_epoch_{n_epochs}"
        run_name = os.path.join(output_dir, name)

        if not args.t2t:
            dataset = DatasetDict({
                'train': Dataset.from_list(train_instances),
                'valid': Dataset.from_list(val_instances),
                'test': Dataset.from_list(test_instances),
            })
            print(sum([x['labels'] for x in train_instances]))
            train_args = TrainingArguments(
                output_dir=run_name,
                do_train=True,
                do_eval=True,
                do_predict=True,
                warmup_ratio = 0.3,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                gradient_accumulation_steps=args.acc_step,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                num_train_epochs=n_epochs,
                load_best_model_at_end=True,
                save_total_limit=2,
            )
            trainer = Trainer(
                model,
                train_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["valid"],
                tokenizer=tokenizer,
            )
        else:

            dataset = DatasetDict({
                'train': Dataset.from_list(train_instances),
                'valid': Dataset.from_list(val_instances),
                'test': Dataset.from_list(test_instances),
            })
            train_args = Seq2SeqTrainingArguments(
                output_dir=run_name,
                do_train=True,
                do_eval=True,
                do_predict=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                gradient_accumulation_steps=args.acc_step,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                num_train_epochs=n_epochs,
                load_best_model_at_end=True,
                save_total_limit=2,
            )
            trainer = Trainer(
                model,
                train_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["valid"],
                tokenizer=tokenizer,
            )

        trainer.train()
        results_dir = os.path.join(run_name, f'{name}_predictions.csv')
        preds = []
        references = []
        pred_numbers = []
        if not args.t2t:
            for d in dataset['test']:
                input_ids = torch.Tensor(d['input_ids']).to(torch.int).reshape(1, -1).to('cuda')
                attn_mask = torch.Tensor(d['attention_mask']).to(torch.int).reshape(1, -1).to('cuda')
                result = model(input_ids=input_ids, attention_mask=attn_mask)
                result = torch.argmax(result.logits).item()
                narrative, question, answer, human_score, label = get_important_parts(d)
                preds.append((f'"{narrative}"', f'"{question}"', f'"{answer}"', str(human_score), str(label), str(result)))
                pred_numbers.append(result)
                references.append(label)
            with open(results_dir, 'w') as out:
                out.write('narrative,question,answer,human_score_avg,prediction,label' + '\n')
                for p in preds:
                    out.write(','.join(p) + '\n')
            accuracy = accuracy_score(references, pred_numbers)
            print(accuracy)
        else:
            for d in dataset['test']:
                input_ids = torch.Tensor(d['input_ids']).to(torch.int).reshape(1, -1).to('cuda')
                result = tokenizer.decode(model.generate(input_ids=input_ids)[0])
                narrative, question, answer, human_score = get_important_parts(d['meta'])
                preds.append((f'"{narrative}"', f'"{question}"', f'"{answer}"', str(human_score), str(result)))
            with open(results_dir, 'w') as out:
                out.write('narrative,question,answer,human_score_avg,prediction,label' + '\n')
                for p in preds:
                    out.write(','.join(p) + '\n')


if __name__ == '__main__':
    main()
