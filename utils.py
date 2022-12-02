def create_nli_instance(qa_instance):
    context = qa_instance['narrative']
    effect = qa_instance['original_sentence_for_question']
    cause = qa_instance['answer']

    premise = context
    if 'because' in cause:
        hypothesis = cause
    else:
        hypothesis = effect[:-1] + ' because ' + cause
    return {
        'premise': premise,
        'hypothesis': hypothesis,
        'meta': qa_instance
    }


def preprocess_label(instance):
    scores = instance['meta']['val_ann']
    if sum(scores) >= 3:
        return sum(scores) - 2
    return 0


def t2t_preprocess_label(instance):
    scores = instance['meta']['val_ann']
    if sum(scores) >= 4:
        return 'entailment'
    return 'not_entailment'


def t2t_preprocess_data(instance, tokenizer):
    input_text = 'tmw hypothesis: ' + instance['hypothesis'] + ' premise: ' + instance['premise']
    encoded = tokenizer(input_text, padding='max_length', max_length=256)
    tokenized_input = {'input_ids': encoded.input_ids}
    label = t2t_preprocess_label(instance)
    output = tokenizer(label, padding='max_length', max_length=10).input_ids
    tokenized_input.update({'labels': output, 'meta': instance['meta']['question_meta']})
    return tokenized_input


def preprocess_data(instance, tokenizer):
    tokenized_input = tokenizer(instance['premise'], instance['hypothesis'], add_special_tokens=True, padding=True)
    label = preprocess_label(instance)
    tokenized_input.update({'labels': label, 'meta': instance['meta']['question_meta']})
    return tokenized_input
