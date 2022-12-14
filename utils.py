def t2t_hp_preprocess_label(instance):
    score = instance['label']
    if score == 1:
        return 'entailment'
    else:
        return 'contradiction'


def t2t_preprocess_label(instance):
    score = instance['label']
    if score == 1:
        return 'valid answer'
    else:
        return 'invalid answer'


def t2t_preprocess_data(instance, tokenizer, hyp_premise):
    if hyp_premise:
        context = instance['narrative']
        effect = instance['original_sentence']
        cause = instance['answer']

        premise = context
        if 'because' in cause:
            hypothesis = cause
        else:
            hypothesis = effect[:-1] + ' because ' + cause
        input_text = 'hypothesis: ' + hypothesis + '\n' + 'premise: ' + premise
        encoded = tokenizer(input_text, padding='max_length', max_length=128)
        tokenized_input = {'input_ids': encoded.input_ids}
        label = t2t_hp_preprocess_label(instance)
    else:
        input_text = 'answer verification narrative: ' + instance['narrative'] + '\nquestion: ' + instance[
            'question'] + '\nanswer: ' + \
                     instance['answer']
        encoded = tokenizer(input_text, padding='max_length', max_length=128)
        tokenized_input = {'input_ids': encoded.input_ids}
        label = t2t_preprocess_label(instance)
    output = tokenizer(label, padding='max_length', max_length=10).input_ids
    tokenized_input.update({'labels': output, 'meta': {'answer': instance['answer'],
                                                       'question': instance['question'],
                                                       'narrative': instance['narrative'],
                                                       'human_score': instance['avg_score'],
                                                       'label': label}})
    return tokenized_input


def preprocess_data(instance, tokenizer, hyp_premise=False):
    if hyp_premise:
        context = instance['narrative']
        effect = instance['original_sentence']
        cause = instance['answer']

        premise = context
        if 'because' in cause:
            hypothesis = cause
        else:
            hypothesis = effect[:-1] + ' because ' + cause
        tokenized_input = tokenizer(premise, hypothesis,
                                    add_special_tokens=True, padding=True)
    else:
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
