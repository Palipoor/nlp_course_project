import argparse
import os
import json

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, \
    Trainer

from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t2t', action='store_true')
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
    accumulation_steps = args.acc_step
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, 'train.json')
    val_file = os.path.join(data_dir, 'val.json')
    test_file = os.path.join(data_dir, 'test.json')
    output_dir = args.expt_dir
    if args.mode == 'train':
        with open(train_file, 'r') as f:
            train_data = json.loads(f.read())
            train_instances = [create_nli_instance(x) for x in train_data]
        with open(val_file, 'r') as f:
            val_data = json.loads(f.read())
            val_instances = [create_nli_instance(x) for x in val_data]
        with open(test_file, 'r') as f:
            test_data = json.loads(f.read())
            test_instances = [create_nli_instance(x) for x in test_data]

        if not args.t2t:
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if not args.t2t:
            train_data_points = [preprocess_data(x, tokenizer) for x in train_instances]
            val_data_points = [preprocess_data(x, tokenizer) for x in val_instances]
            test_data_points = [preprocess_data(x, tokenizer) for x in test_instances]
            dataset = DatasetDict({
                'train': Dataset.from_list(train_data_points),
                'valid': Dataset.from_list(val_data_points),
                'test': Dataset.from_list(test_data_points),
            })

        train_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"{args.expt_name}_bs_{batch_size}_lr_{lr}_epoch_{n_epochs}"),
            do_train=True,
            do_eval=True,
            do_predict=True,
            evaluation_strategy="epoch",
            save_strategy = "epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=n_epochs,
            load_best_model_at_end=True,
            save_total_limit = 2,
        )

        trainer = Trainer(
            model,
            train_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            tokenizer=tokenizer,
        )

        trainer.train()
        results_dir = os.path.join(output_dir, 'predictions.csv')
        preds = []
        for d in dataset['test']:
            input_ids = torch.Tensor(d['input_ids']).to(torch.int).reshape(1, -1).to('cuda')
            attn_mask = torch.Tensor(d['attention_mask']).to(torch.int).reshape(1, -1).to('cuda')
            result = model(input_ids=input_ids, attention_mask=attn_mask)
            preds.append((d['meta'], torch.argmax(result.logits).item()))

        with open(results_dir, 'w') as out:
            for p in preds:
                out.write(p[0] + ',' + str(p[1]) + '\n')


if __name__ == '__main__':
    main()
