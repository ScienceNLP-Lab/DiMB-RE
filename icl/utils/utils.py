import logging
import re
import random
from collections import defaultdict
from shared.data_structures import Dataset

logger = logging.getLogger('root')

random.seed(42)

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))
    return doc_sent, sub, obj

def generate_data(entity_data, sent_level=True):

    logger.info('Generate relation data from %s'%(entity_data))

    data = Dataset(entity_data)

    num_null = 0
    num_rel = 0

    samples = []
    for doc in data:
        for i, sent in enumerate(doc):

            num_rel += len(sent.relations)

            if sent_level:
                sent_sample = {}
                sent_sample['docid'] = doc._doc_key
                sent_sample['id'] = '%s@%d'%(doc._doc_key, sent.sentence_ix)
                sent_sample['sentence'] = ' '.join(sent.text)
                sent_sample['sentence_ix'] = sent.sentence_ix
                sent_sample['relations'] = []
                sent_sample['entities'] = defaultdict(set)
            else:
                sent_sample = []
            
            sent_ner = sent.ner

            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label
                if sent_level:
                    ner_mention = ' '.join(sent.text[ner.span.start_sent:ner.span.end_sent+1])
                    sent_sample['entities'][ner.label].add(ner_mention)
            
            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = (rel.label, rel.certainty)
            
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text
            
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label, factuality = gold_rel.get((sub.span, obj.span), ('no_relation', ''))

                    if label == 'no_relation':
                        num_null += 1

                    sample = {}
                    sample['docid'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                    sample['relation'] = label
                    sample['factuality'] = factuality
                    sample['subj_start'] = sub.span.start_sent + sent_start
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['subj_mention'] = ' '.join(tokens[sub.span.start_sent:sub.span.end_sent+1])
                    
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_sent + sent_start
                    sample['obj_type'] = obj.label
                    sample['obj_mention'] = ' '.join(tokens[obj.span.start_sent:obj.span.end_sent+1])

                    sample['token'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end

                    if sent_level:
                        sent_sample['relations'].append(sample)
                    else:
                        sent_sample.append(sample)

            if sent_level:
                samples.append(sent_sample)
            else:
                samples.extend(sent_sample)
    
    tot = len(samples)  # Num of sents
    logger.info('#samples: %d'%(tot))
    logger.info('Proportion of NULL examples: %.4f'%(num_null/tot))

    return data, samples


def generate_subset(prompts, samples, method, subset_ratio, null_ratio):
    total_len = len(samples)
    null_list = []
    valid_list = []
    for idx, (prompt, sample) in enumerate(zip(prompts, samples)):
        if sample['relation'] == 'no_relation':
            null_list.append((idx, prompt, sample))
        else:
            valid_list.append((idx, prompt, sample))

    if method == 'random':
        random.shuffle(null_list)
        random.shuffle(valid_list)

    null_subset = null_list[:int(total_len*subset_ratio*null_ratio)]
    valid_subset = valid_list[:int(total_len*subset_ratio*(1-null_ratio))]

    logger.info(f'# Null subset: {len(null_subset)}')
    logger.info(f'# Valid subset: {len(valid_subset)}')

    prompt_subset = []
    sample_subset = []
    subset_indices = []
    for subset in null_subset:
        subset_indices.append(subset[0])
        prompt_subset.append(subset[1])
        sample_subset.append(subset[2])
    for subset in valid_subset:
        subset_indices.append(subset[0])
        prompt_subset.append(subset[1])
        sample_subset.append(subset[2])

    logger.info(subset_indices[:10])   

    # # Just for check
    # i = 0
    # for sample, prompt, idx in zip(sample_subset, prompt_subset, subset_indices):
    #     if i >= 5:
    #         break
    #     logger.info(f"Sample-idx >>> {idx}")
    #     logger.info(f"Sample >>> {sample}")
    #     logger.info(f"Prompt >>> {prompt}")
    #     i += 1
    # # Check in reverse
    # for sample, prompt, idx in zip(sample_subset[::-1], prompt_subset, subset_indices):
    #     if i >= 10:
    #         break
    #     logger.info(f"Sample-idx >>> {idx}")
    #     logger.info(f"Sample >>> {sample}")
    #     logger.info(f"Prompt >>> {prompt}")
    #     i += 1 

    return prompt_subset, sample_subset, subset_indices


def generate_prompt(demo_samples, query_samples, structure, verbalizer, prompt_dir, nli, few_shot, ranks):

    # load Task Description
    with open(prompt_dir, 'r') as f:
        desc = f.read()

    prompts = []

    # zero-shot settings based on NLI-style multiple choices
    for idx, sample in enumerate(query_samples):

        prompt = desc

        # Few-shot retrieval
        if few_shot:
            if ranks is not None:
                target_rank = ranks[idx]
                for rank in target_rank:
                    most_similar = demo_samples[int(rank)]
                    relations = most_similar['relations']
                    flag = False
                    for annotation in relations:
                        if (annotation['subj_type'] == sample['subj_type']) \
                            and annotation['obj_type'] == sample['obj_type']:
                            prompt = build_demos(
                                annotation, prompt, 
                                structure, verbalizer, nli, 
                                is_test_input=False
                            )
                            flag = True
                            break
                    if flag:
                        break
            else:
                while True:
                    relations = random.sample(demo_samples, 1)[0]['relations']
                    if relations:
                        annotation = random.sample(relations, 1)[0]
                        prompt = build_demos(
                            annotation, prompt, 
                            structure, verbalizer, nli, 
                            is_test_input=False
                        )
                        break
            
        # Test input
        prompt = build_demos(
            sample, prompt, 
            structure, verbalizer, nli, 
            is_test_input=True
        )
        prompts.append(prompt)

    return prompts


def build_demos(sample, prompt, structure, verbalizer, nli, is_test_input):

    multiple_choices = []
    pair = '-'.join([sample['subj_type'], sample['obj_type']])
    for rel_type, pairs in structure.items():
        if pair in pairs:
            if nli:
                factual = verbalizer[rel_type]['Factual'].format(sample['subj_mention'], sample['obj_mention'])
                negated = verbalizer[rel_type]['Negated'].format(sample['subj_mention'], sample['obj_mention'])
                unknown = verbalizer[rel_type]['Unknown'].format(sample['subj_mention'], sample['obj_mention'])
                multiple_choices.extend([factual, negated, unknown])
            else:
                if rel_type != 'no_relation':
                    multiple_choices.append(rel_type)

    if (not is_test_input) \
        and (sample['relation'] != 'no_relation') \
        and (sample['relation'] not in multiple_choices):
        multiple_choices.append(sample['relation'])

    rel2idx = {line: idx for idx, line in enumerate(multiple_choices, 1)}

    if nli:
        line = verbalizer['no_relation'].format(sample['subj_mention'], sample['obj_mention'])
        multiple_choices.append(line)

    multiple_choices = [f"{idx}. {line}" for idx, line in enumerate(multiple_choices, 1)]

    tokens = []
    for i, token in enumerate(sample['token']):
        if i == sample['subj_start']:
            subj_start = len(tokens)
            tokens.append(f"<SUBJ_START={sample['subj_type']}>")
        if i == sample['obj_start']:
            obj_start = len(tokens)
            tokens.append(f"<OBJ_START={sample['obj_type']}>")
        tokens.append(token)
        if i == sample['subj_end']:
            subj_end = len(tokens)
            tokens.append(f"<SUBJ_END={sample['subj_type']}>")
        if i == sample['obj_end']:
            obj_end = len(tokens)
            tokens.append(f"<OBJ_END={sample['obj_type']}>")

    sent = ' '.join(tokens)
    prompt += '\n' + f"Sentence: {sent}"
    prompt += '\n'

    if not nli:
        q1 = "Question 1: Following the above sentence, find out carefully whether there is a relation between '{}' (role: Subject, type: {}) and '{}' (role: Object, type: {}). A relation exists if and only if its type is included in the Guideline.".format(
                sample['subj_mention'], sample['subj_type'],
                sample['obj_mention'], sample['obj_type']
            )
        c1 = "1. Relation exists\n2. No relation"
        q2 = "Question 2: If there is a relation, please select an option number and relation type below that best describes the relation of given entity pair."
        prompt += '\n'.join([q1, c1, q2])

    for line in multiple_choices:
        prompt += '\n' + line

    fact2idx = {"Factual": 1, "Negated": 2, "Unknown": 3}

    if not nli:
        prompt += '\n'
        q3 = "Question 3: Which of these options best describe the certainty level of the relation between the two entities?"
        c3 = "1. Factual: It is used for relations that are expressed as an assertion or a fact.\n2. Negated: It is used when relations are asserted as a negative fact.\n3. Unknown: Less or no certainty about the relationship."
        prompt += '\n'.join([q3, c3])

        prompt += '\n\n'
        desc1 = "Please make sure that you should respond to Question 2 and 3 if you answer '1. There is a relation between the entity pair' for Question 1. If you answer A1: 2. No relation, then you should not answer Question 2 and 3."
        desc2 = "When answering the questions, you should follow this format: 'A1: 1. Relation exists | A2: 1. RELATION-TYPE | A3: 2. Factuality-level', or 'A1: 2. No relation'. Please use the bar sign to separate answers for each question." 
        desc3 = "Note that when there are no choice options in Question 2, you should only reply 2. No relation for Question 1."
        prompt += '\n'.join([desc1, desc2, desc3])

        if not is_test_input:
            a1 = a2 = a3 = ''
            if sample['relation'] != 'no_relation':
                a1 = 'A1: 1. Relation exists'
                a2 = f"A2: {rel2idx[sample['relation']]}. {sample['relation']}"
                a3 = f"A3: {fact2idx[sample['factuality']]}. {sample['factuality']}"
            else:
                a1 = 'A1: 2. No relation'
            answer = ' | '.join([a1, a2, a3]) if a2 else a1
            prompt += '\n' + answer
            prompt += '\n\n'

    return prompt

def reconstruct_samples(demo_samples, query_samples):
    # Collect target sentences
    demo_texts = [sample['sentence'] for sample in demo_samples]
    # Reconstruct context
    query_texts = []
    for sample in query_samples:
        sentence = ' '.join(sample['token'])
        # TODO: Add entity types?
        prefix = f"The relation between '{sample['subj_mention']}' and '{sample['obj_mention']}' in the context: "
        query_texts.append(prefix + sentence)

    return demo_texts, query_texts


def replace_inside_apostrophes(input_string):
    # Replace content inside apostrophes with {}
    pattern = r"'(.*?)'"
    replaced_string = re.sub(pattern, "\'{}\'", input_string)
    return replaced_string

def remove_period_pattern(input_string):
    # Remove numbers followed by period pattern
    pattern_numbers_period = r'\b\d+\.'
    replaced_string = re.sub(pattern_numbers_period, '', input_string)
    return replaced_string

def extract_predictions(responses, statement2label, rel_labels, two_step=True):
    # Define a regular expression pattern 
    # to match any number followed by a dot 
    # and capture the rest of the sentence
    predictions = []
    pattern = re.compile(r'\d+\..*')
    for response in responses:
        if '\n' in response:
            response = response.split('\n')[0]  # Take only first response
        if two_step:
            response = response.split('|')
            if len(response) == 3:
                pred = []
                pred_rel, pred_fact = response[1:]

                if pattern.findall(pred_rel):
                    pred_rel = pattern.findall(pred_rel)[0]
                    pred_rel = remove_period_pattern(pred_rel).strip()
                else:
                    pred_rel = pred_rel.replace('A2:  ', '').strip()

                # Manual correction
                pred_rel = pred_rel.upper()
                if pred_rel == 'NEGATIVELY ASSOCIATED WITH':
                    pred_rel = 'NEG_ASSOCIATED_WITH'
                elif pred_rel == 'POSITIVELY ASSOCIATED WITH':
                    pred_rel = 'POS_ASSOCIATED_WITH'
                elif pred_rel == "ASSOCIATED WITH":
                    pred_rel = 'ASSOCIATED_WITH'

                pred_fact = pattern.findall(pred_fact)[0]
                pred_fact = remove_period_pattern(pred_fact).strip()

                # TODO: remove hard-coded syntax
                if pred_rel not in rel_labels or pred_fact not in ['Factual', 'Negated', 'Unknown']:
                    pred = ('no_relation', '')
                else:
                    pred = (pred_rel, pred_fact)
            else:
                pred = ('no_relation', '')

            predictions.append(pred)

        else:
            statement = pattern.findall(response)[0]
            statement = replace_inside_apostrophes(statement).rstrip('.').strip()
            statement = remove_period_pattern(statement).strip()
            if statement in statement2label:
                pred = statement2label[statement]
                pred = pred.split('-')
                if len(pred) > 1:
                    predictions.append(tuple(pred))
                else:
                    predictions.append(('no_relation', ''))  # Placeholder
            else:
                predictions.append(('no_relation', ''))
    return predictions
