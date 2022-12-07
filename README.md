# Code Base for NLP (CSE 538) Course Project - Fall 2022 - Stony Brook University



## Original source code

We implemented the code from scratch, using [Huggingface](https://huggingface.co/) documentations and Google search for some minor issues. The argument parsing part was losely copied from [this repo](https://github.com/PlusLabNLP/Com2Sense).

## Code Explanation
The code is pretty straightforward. There are two python modules in this code:
- `main.py`: Contains code for training and evaluating models.
- `utils.py`: Contains code for pre-processing data.

### Arguments

To finetune models using this code you need to provide a few arguments. Most of these have default values, but setting them yourself gives you more control over the code.
- `-t2t` Flag, default to False: whether or not you're fine-tuning a seq2seq model.
- `hp` Short for: hypothesis-permise: use this when you want to pre-process data as pairs of hypothesis and premise.
- `--mode` Defaults to: train. Unfortunately we didn't implement the test mode seapartely, but if you wanted to edit the code and add a test mode to load and test a model on a data you can use this argument.
- `--expt_dir` Required. This is the directory to save the experiment logs and checkpoints.
- `--expt_name` Required. A name for this experiment.
- `--data_dir` Required. A directory which contains files: `train.json`, `val.json`, `test.json`
- `--model` Required. The name of the model. This will be used to load a pre-trained model from Huggingface hub. 
- `--lr` Base learning rate. Defaults to 2e-5.
- `--epochs` Number of epochs to train the model. Defaults to 10.
- `--batch_size`Batch size used in training. Defaults to 8.
- `--acc_step` Number of steps to accumulate gradient before updating the model parameters. Defaults to 1.

### Example Script
```bash
python main.py 
--mode train
--model roberta-large
-hp
--expt_dir './project_runs'
--epochs 10
--lr 5e-5
--batch_size 8
--expt_name 'roberta_large_demo'
--data_dir './new_data'
```
### Requirements

This project was run with python==3.9.12, pytorch ==1.13.0, transformers==4.24.0, datasets==2.7.0, scikit-learn==1.1.1. 
