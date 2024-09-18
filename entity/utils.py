import numpy as np
import json
import logging
from collections import Counter, defaultdict
from transformers import AutoTokenizer

logger = logging.getLogger('root')

def batchify(samples_all, batch_size, model_name):

    """
    Batchify samples with a batch size
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Exclude samples whose subtokens are more than 512
    samples = []
    for i in range(0, len(samples_all)):
        sub_tokens = []
        for token in samples_all[i]['tokens']:
            sub_tokens.extend(tokenizer.tokenize(token))
        if len(sub_tokens) <= 510:
            samples.append(samples_all[i])

    num_samples = len(samples)

    list_samples_batches = []
    
    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)
    
    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'], samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False

# def convert_dataset_to_samples(dataset, task, max_span_length, ner_label2id=None, context_window=0):
#     """
#     Extract sentences and gold entities from a dataset
#     """
#     samples = []
#     num_ner = 0
#     num_trg = 0
#     max_len = 0
#     max_ner = 0
#     max_trg = 0
#     num_overlap = 0
#     label_cnt_dict = Counter()

#     for c, doc in enumerate(dataset):
#         for i, sent in enumerate(doc):
#             num_ner += len(sent.ner)
#             num_trg += len(sent.triggers)
#             sample = {
#                 'doc_key': doc._doc_key,
#                 'sentence_ix': sent.sentence_ix,
#             }
#             if context_window != 0 and len(sent.text) > context_window:
#                 logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))
#                 # print('Exclude:', sample)
#                 # continue
#             sample['tokens'] = sent.text
#             sample['sent_length'] = len(sent.text)
#             sent_start = 0
#             sent_end = len(sample['tokens'])

#             max_len = max(max_len, len(sent.text))
#             max_ner = max(max_ner, len(sent.ner))
#             max_trg = max(max_trg, len(sent.triggers))

#             if context_window > 0:
#                 add_left = (context_window-len(sent.text)) // 2
#                 add_right = (context_window-len(sent.text)) - add_left
                
#                 # add left context
#                 j = i - 1
#                 while j >= 0 and add_left > 0:
#                     context_to_add = doc[j].text[-add_left:]
#                     sample['tokens'] = context_to_add + sample['tokens']
#                     add_left -= len(context_to_add)
#                     sent_start += len(context_to_add)
#                     sent_end += len(context_to_add)
#                     j -= 1

#                 # add right context
#                 j = i + 1
#                 while j < len(doc) and add_right > 0:
#                     context_to_add = doc[j].text[:add_right]
#                     sample['tokens'] = sample['tokens'] + context_to_add
#                     add_right -= len(context_to_add)
#                     j += 1

#             sample['sent_start'] = sent_start
#             sample['sent_end'] = sent_end
#             sample['sent_start_in_doc'] = sent.sentence_start
            
#             sent_ner = {}
#             for ner in sent.ner:
#                 sent_ner[ner.span.span_sent] = ner.label
#                 label_cnt_dict[ner.label] += 1.0

#             # Add Trigger mentions to NER task
#             if "trg" in task:
#                 for trg in sent.triggers:
#                     if task.endswith("dummy"):
#                         sent_ner[trg.span.span_sent] = "TRIGGER"  # Dummification
#                         label_cnt_dict["TRIGGER"] += 1.0
#                     else:
#                         sent_ner[trg.span.span_sent] = trg.label
#                         label_cnt_dict[trg.label] += 1.0

#             span2id = {}
#             sample['spans'] = []
#             sample['spans_label'] = []

#             # sent.text
#             # ['Effects', 'of', 'enriched', 'seafood', 'sticks', '(', 'heat', '-', 'inactivated', 'B', '.', 'animalis', '.']

#             for i in range(len(sent.text)):
#                 for j in range(i, min(len(sent.text), i+max_span_length)):
#                     sample['spans'].append((i+sent_start, j+sent_start, j-i+1))
#                     span2id[(i, j)] = len(sample['spans'])-1
#                     if (i, j) not in sent_ner:
#                         sample['spans_label'].append(0)
#                     else:
#                         sample['spans_label'].append(ner_label2id[sent_ner[(i, j)]])
#             samples.append(sample)
            
#     avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
#     max_length = max([len(sample['tokens']) for sample in samples])

#     logger.info('# Overlap: %d'%num_overlap)
#     logger.info('Extracted %d samples from %d documents, with %d NER labels, %d TRG labels, %.3f avg input length, %d max length'%(len(samples), len(dataset), num_ner, num_trg, avg_length, max_length))
#     logger.info('Max Length: %d, max NER: %d, max TRG: %d'%(max_len, max_ner, max_trg))

#     return samples, num_ner, num_trg, label_cnt_dict

def convert_dataset_to_samples(args, dataset, ner_label2id=None):
    """
    Extract sentences and gold entities from a dataset
    """
    max_len = 0
    num_ner = 0
    num_trg = 0
    max_ner = 0
    max_trg = 0
    label_cnt_dict = Counter()

    samples = []
    span_len_entity = defaultdict(int)
    span_len_trigger = defaultdict(int)

    for c, doc in enumerate(dataset):
        for i, sent in enumerate(doc):

            ner_not_nested = []
            ner_nested = []

            # If you want to remove nested entities
            if args.remove_nested:
                checker = [0]*len(sent.text)
                ner_sorted = sorted(sent.ner, key=lambda x: (x.span.start_sent, -(x.span.end_sent-x.span.start_sent)))
                for ner in ner_sorted:
                    if all(idx == 0 for idx in checker[ner.span.start_sent:ner.span.end_sent]):
                        checker[ner.span.start_sent:ner.span.end_sent] = [1]*(ner.span.end_sent-ner.span.start_sent+1)
                        ner_not_nested.append(ner)
                    else:
                        ner_nested.append(ner)
                sent.ner = ner_not_nested

            num_ner += len(sent.ner)
            if args.extract_trigger:
                num_trg += len(sent.triggers)

            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
            }

            if args.context_window != 0 and len(sent.text) > args.context_window:
                logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))
                # print('Exclude:', sample)
                # continue
            sample['tokens'] = sent.text
            sample['sent_length'] = len(sent.text)
            sent_start = 0
            sent_end = len(sample['tokens'])

            max_len = max(max_len, len(sent.text))
            max_ner = max(max_ner, len(sent.ner))
            max_trg = max(max_trg, len(sent.triggers))

            if args.context_window > 0:
                add_left = (args.context_window-len(sent.text)) // 2
                add_right = (args.context_window-len(sent.text)) - add_left
                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1
                # add right context
                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start
            
            sent_ner = {}
            for ner in sent.ner:
                sent_ner[ner.span.span_sent] = ner.label
                label_cnt_dict[ner.label] += 1.0
                span_len = ner.span.end_sent - ner.span.start_sent + 1
                span_len_entity[span_len] += 1

            sent_trg = {}
            if args.extract_trigger:
                for trg in sent.triggers:
                    if args.untyped_trigger:
                        sent_trg[trg.span.span_sent] = "TRIGGER"  # Untyped TRIGGER
                        label_cnt_dict["TRIGGER"] += 1.0
                    else:
                        sent_trg[trg.span.span_sent] = trg.label
                        label_cnt_dict[trg.label] += 1.0
                    span_len = trg.span.end_sent - trg.span.start_sent + 1
                    span_len_trigger[span_len] += 1

            span2id = {}
            sample['spans'] = []
            sample['spans_label'] = []

            for i in range(len(sent.text)):
                for j in range(i, min(len(sent.text), i+max(args.max_span_length_entity, args.max_span_length_trigger))):
                    sample['spans'].append(
                        (i+sent_start, j+sent_start, j-i+1)
                    )
                    span2id[(i, j)] = len(sample['spans'])-1

                    if (i, j) in sent_ner and (j-i+1) <= args.max_span_length_entity:
                        sample['spans_label'].append(ner_label2id[sent_ner[(i, j)]])
                    elif (i, j) in sent_trg and (j-i+1) <= args.max_span_length_trigger:
                        sample['spans_label'].append(ner_label2id[sent_trg[(i, j)]])
                    else:
                        sample['spans_label'].append(0)

                    # if args.extract_trigger and args.dual_classifier:
                    #     if (i, j) not in sent_ner:
                    #         sample['spans_label_entity'].append(0)
                    #     else:
                    #         sample['spans_label_entity'].append(entity_label2id[sent_ner[(i, j)]])

                    #     if (i, j) not in sent_trg:
                    #         sample['spans_label_trigger'].append(0)
                    #     else:
                    #         sample['spans_label_trigger'].append(trigger_label2id[sent_trg[(i, j)]])
                    # else:
                    #     if (i, j) in sent_ner:
                    #         sample['spans_label_entity'].append(entity_label2id[sent_ner[(i, j)]])
                    #     elif (i, j) in sent_trg:
                    #         sample['spans_label_entity'].append(entity_label2id[sent_trg[(i, j)]])
                    #     else:
                    #         sample['spans_label_entity'].append(0)

            samples.append(sample)
            
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    span_len_entity = dict(sorted([(k, v) for k, v in span_len_entity.items()]))
    span_len_trigger = dict(sorted([(k, v) for k, v in span_len_trigger.items()]))

    logger.info('Extracted %d samples from %d documents, with %d NER labels, %d TRG labels, %.3f avg input length, %d max length'%(len(samples), len(dataset), num_ner, num_trg, avg_length, max_length))
    logger.info('Max Length: %d, max NER: %d, max TRG: %d'%(max_len, max_ner, max_trg))
    logger.info(f'Span Length of Entities >>> {span_len_entity}')
    logger.info(f'Span Length of Triggers >>> {span_len_trigger}')

    return samples, num_ner, num_trg, label_cnt_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data
