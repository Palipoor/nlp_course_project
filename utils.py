import argparse
import sys
import pickle
import pandas as pd
import json
import re


# def create_nli_instance(qa_instance):
#     context = qa_instance['narrative']
#     effect = qa_instance['original_sentence_for_question']
#     cause = qa_instance['answer']
#
#     premise = context
#     if 'because' in cause:
#         hypothesis = cause
#     else:
#         hypothesis = effect[:-1] + ' because ' + cause
#     return {
#         'premise': premise,
#         'hypothesis': hypothesis,
#         'meta': qa_instance
#     }


# def preprocess_label(instance):
#     scores = instance['meta']['val_ann']
#     if sum(scores) >= 3:
#         return sum(scores) - 2
#     return 0

#
def t2t_preprocess_label(instance):
    score = instance['label']
    if score == 1:
        return 'valid'
    else:
        return 'invalid'


def t2t_preprocess_data(instance, tokenizer):
    input_text = 'tmw question: ' + instance['question'] + ' context: ' + instance['narrative'] + ' answer: ' + \
                 instance['answer']
    encoded = tokenizer(input_text, padding='max_length', max_length=256)
    tokenized_input = {'input_ids': encoded.input_ids}
    label = t2t_preprocess_label(instance)
    output = tokenizer(label, padding='max_length', max_length=5).input_ids
    tokenized_input.update({'labels': output, 'meta': {'answer': instance['answer'],
                                                       'question': instance['question'],
                                                       'narrative': instance['narrative'],
                                                       'human_score': instance['avg_score'],
                                                       'label': instance['label']}})
    return tokenized_input


def preprocess_data(instance, tokenizer):
    tokenized_input = tokenizer(instance['narrative'], instance['question'], instance['answer'],
                                add_special_tokens=True, padding=True)
    label = instance['label']
    tokenized_input.update({'labels': label, 'meta': {'answer': instance['answer'],
                                                      'question': instance['question'],
                                                      'narrative': instance['narrative'],
                                                      'human_score': instance['avg_score'],
                                                      'label': instance['label']}})
    return tokenized_input


def get_important_parts(data):
    meta = data['meta']
    return meta['narrative'], meta['question'], meta['answer'], meta['human_score'], meta['label']
