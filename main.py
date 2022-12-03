import argparse
import os
import json

import torch
from scipy import stats
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, \
    Trainer, Seq2SeqTrainingArguments

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

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 3.0, 1.5, 1.0, 4.0])).to('cuda')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

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
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=5)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        name = f"{args.expt_name}_bs_{batch_size}_lr_{lr}_epoch_{n_epochs}"
        run_name = os.path.join(output_dir, name)

        if not args.t2t:
            train_data_points = [preprocess_data(x, tokenizer) for x in train_instances]
            val_data_points = [preprocess_data(x, tokenizer) for x in val_instances]
            test_data_points = [preprocess_data(x, tokenizer) for x in test_instances]
            dataset = DatasetDict({
                'train': Dataset.from_list(train_data_points),
                'valid': Dataset.from_list(val_data_points),
                'test': Dataset.from_list(test_data_points),
            })
            print(sum([x['labels'] for x in train_data_points]))
            train_args = TrainingArguments(
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
            trainer = CustomTrainer(
                model,
                train_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["valid"],
                tokenizer=tokenizer,
            )
        else:
            train_data_points = [t2t_preprocess_data(x, tokenizer) for x in train_instances]
            val_data_points = [t2t_preprocess_data(x, tokenizer) for x in val_instances]
            test_data_points = [t2t_preprocess_data(x, tokenizer) for x in test_instances]
            dataset = DatasetDict({
                'train': Dataset.from_list(train_data_points),
                'valid': Dataset.from_list(val_data_points),
                'test': Dataset.from_list(test_data_points),
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
                result =  torch.argmax(result.logits).item()
                narrative, question, answer, human_score = get_important_parts(d['meta'])
                preds.append((f'"{narrative}"', f'"{question}"', f'"{answer}"', human_score, str(result)))
                pred_numbers.append(result)
                references.append(int(human_score))
            with open(results_dir, 'w') as out:
                out.write('narrative,question,answer,human_score_sum,prediction' + '\n')
                for p in preds:
                    out.write(','.join(p) + '\n')
            corr = stats.spearmanr(references, pred_numbers)
            print(corr)
        else:
            for d in dataset['test']:
                input_ids = torch.Tensor(d['input_ids']).to(torch.int).reshape(1, -1).to('cuda')
                result = tokenizer.decode(model.generate(input_ids=input_ids)[0])
                narrative, question, answer, human_score = get_important_parts(d['meta'])
                preds.append((f'"{narrative}"', f'"{question}"', f'"{answer}"', human_score, str(result)))
            with open(results_dir, 'w') as out:
                out.write('narrative,question,answer,human_score_sum,prediction' + '\n')
                for p in preds:
                    out.write(','.join(p) + '\n')


if __name__ == '__main__':
    main()
