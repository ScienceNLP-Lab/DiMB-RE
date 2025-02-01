from collections import Counter
import logging

from shared.data_structures import Dataset

logger = logging.getLogger('root')

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))
    return doc_sent, sub, obj

def generate_relation_data(entity_data, training=True, use_gold=False, use_trigger=True, context_window=0):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0, 0
    nrel_sent_level = 0
    max_sentsample = 0
    label_cnt_dict = Counter()
    # certainty_cnt_dict = Counter()

    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []

            nner += len(sent.ner)
            nrel += len(sent.relations)
            if use_gold:
                sent_ner = sent.ner
                if use_trigger:
                    sent_triplet = sent.triplets
            else:
                sent_ner = sent.predicted_ner
                if use_trigger:
                    sent_triplet = sent.predicted_triplets
            
            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label
            
            gold_rel = {}
            # gold_certainty = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
                label_cnt_dict[rel.label] += 1.0
                # gold_certainty[rel.pair] = rel.certainty
                # certainty_cnt_dict[rel.certainty] += 1.0

            if use_trigger:
                triplets_lookup = {}
                if training:
                    for pair in sent.triplets:    # From GOLD triggers
                        span1, span2, span_trg = pair.triplet
                        triplets_lookup[(span1, span2)] = span_trg
                else:
                    for pair in sent_triplet:    # From predicted triggers
                        span1, span2, span_trg = pair.triplet
                        triplets_lookup[(span1, span2)] = span_trg
            
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1
                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1
            
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span, obj.span), 'no_relation')

                    span_result = None
                    if label != "no_relation":
                        nrel_sent_level += 1
                    if use_trigger:
                        span_result = triplets_lookup.get((sub.span, obj.span))

                    sample = {}
                    sample['docid'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                    sample['relation'] = label
                    sample['subj_start'] = sub.span.start_sent + sent_start
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_sent + sent_start
                    sample['obj_type'] = obj.label

                    if use_trigger and span_result is not None:
                        sample['trg_start'] = span_result.start_sent + sent_start
                        sample['trg_end'] = span_result.end_sent + sent_start

                    sample['token'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end
                    sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    nrel_invalid = nrel-nrel_sent_level
    if not nrel_invalid:
        print(f"All Relations are within Sent-level!")
    else: 
        print(f"# of Invalid Relations within Sent-level: {nrel_invalid}({round(nrel_invalid/nrel*100, 4)}%)")

    return data, samples, nrel, label_cnt_dict


# def generate_relation_data(entity_data, use_gold=False, context_window=0):
#     """
#     Prepare data for the relation model
#     If training: set use_gold = True
#     """
#     logger.info('Generate relation data from %s'%(entity_data))
#     data = Dataset(entity_data)

#     nner, nrel = 0, 0
#     nrel_sent_level = 0
#     max_sentsample = 0
#     samples = []
#     for doc in data:
#         for i, sent in enumerate(doc):
#             sent_samples = []

#             nner += len(sent.ner)
#             nrel += len(sent.relations)
#             if use_gold:
#                 sent_ner = sent.ner
#             else:
#                 sent_ner = sent.predicted_ner
            
#             gold_ner = {}
#             for ner in sent.ner:
#                 gold_ner[ner.span] = ner.label
            
#             gold_rel = {}
#             for rel in sent.relations:
#                 gold_rel[rel.pair] = rel.label
            
#             sent_start = 0
#             sent_end = len(sent.text)
#             tokens = sent.text

#             triggers_start = [ner.span.start_sent for ner in sent_ner if ner.label == "Trigger"]
#             triggers_end = [ner.span.end_sent for ner in sent_ner if ner.label == "Trigger"]

#             if context_window > 0:
#                 add_left = (context_window-len(sent.text)) // 2
#                 add_right = (context_window-len(sent.text)) - add_left

#                 j = i - 1
#                 while j >= 0 and add_left > 0:
#                     context_to_add = doc[j].text[-add_left:]
#                     tokens = context_to_add + tokens
#                     add_left -= len(context_to_add)
#                     sent_start += len(context_to_add)
#                     sent_end += len(context_to_add)
#                     j -= 1

#                 j = i + 1
#                 while j < len(doc) and add_right > 0:
#                     context_to_add = doc[j].text[:add_right]
#                     tokens = tokens + context_to_add
#                     add_right -= len(context_to_add)
#                     j += 1
            
#             for x in range(len(sent_ner)):
#                 for y in range(len(sent_ner)):
#                     if x == y or "TRIGGER" in sent_ner[x].label + sent_ner[y].label:    # To exclude Trigger entites from the entity pairs
#                         continue
#                     sub = sent_ner[x]
#                     obj = sent_ner[y]
#                     label = gold_rel.get((sub.span, obj.span), 'no_relation')

#                     if label != "no_relation":
#                         nrel_sent_level += 1

#                     sample = {}
#                     sample['docid'] = doc._doc_key
#                     sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
#                     sample['relation'] = label
#                     sample['subj_start'] = sub.span.start_sent + sent_start
#                     sample['subj_end'] = sub.span.end_sent + sent_start
#                     sample['subj_type'] = sub.label
#                     sample['obj_start'] = obj.span.start_sent + sent_start
#                     sample['obj_end'] = obj.span.end_sent + sent_start
#                     sample['obj_type'] = obj.label

#                     sample['trg_start'] = triggers_start
#                     sample['trg_end'] = triggers_end

#                     sample['token'] = tokens
#                     sample['sent_start'] = sent_start
#                     sample['sent_end'] = sent_end

#                     sent_samples.append(sample)


#             max_sentsample = max(max_sentsample, len(sent_samples))
#             samples += sent_samples
    
#     tot = len(samples)
#     logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

#     nrel_invalid = nrel-nrel_sent_level
#     print(f"# of Invalid Relations within Setence-level:{nrel_invalid} - {round(nrel_invalid/nrel, 4)*100}%")

#     return data, samples, nrel


# def generate_relation_data(entity_data, use_gold=False, context_window=0):
#     """
#     Prepare data for the relation model
#     If training: set use_gold = True
#     """
#     logger.info('Generate relation data from %s'%(entity_data))
#     data = Dataset(entity_data)

#     nner, nrel = 0, 0
#     max_sentsample = 0
#     samples = []
#     for doc in data:
#         for i, sent in enumerate(doc):
#             sent_samples = []

#             nner += len(sent.ner)
#             nrel += len(sent.relations)
#             if use_gold:
#                 sent_ner = sent.ner
#             else:
#                 sent_ner = sent.predicted_ner
            
#             gold_ner = {}
#             for ner in sent.ner:
#                 gold_ner[ner.span] = ner.label
            
#             gold_rel = {}
#             for rel in sent.relations:
#                 gold_rel[rel.pair] = rel.label
            
#             sent_start = 0
#             sent_end = len(sent.text)
#             tokens = sent.text

#             if context_window > 0:
#                 add_left = (context_window-len(sent.text)) // 2
#                 add_right = (context_window-len(sent.text)) - add_left

#                 j = i - 1
#                 while j >= 0 and add_left > 0:
#                     context_to_add = doc[j].text[-add_left:]
#                     tokens = context_to_add + tokens
#                     add_left -= len(context_to_add)
#                     sent_start += len(context_to_add)
#                     sent_end += len(context_to_add)
#                     j -= 1

#                 j = i + 1
#                 while j < len(doc) and add_right > 0:
#                     context_to_add = doc[j].text[:add_right]
#                     tokens = tokens + context_to_add
#                     add_right -= len(context_to_add)
#                     j += 1
            
#             for x in range(len(sent_ner)):
#                 for y in range(len(sent_ner)):
#                     if x == y:
#                         continue
#                     sub = sent_ner[x]
#                     obj = sent_ner[y]
#                     label = gold_rel.get((sub.span, obj.span), 'no_relation')
#                     sample = {}
#                     sample['docid'] = doc._doc_key
#                     sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
#                     sample['relation'] = label
#                     sample['subj_start'] = sub.span.start_sent + sent_start
#                     sample['subj_end'] = sub.span.end_sent + sent_start
#                     sample['subj_type'] = sub.label
#                     sample['obj_start'] = obj.span.start_sent + sent_start
#                     sample['obj_end'] = obj.span.end_sent + sent_start
#                     sample['obj_type'] = obj.label
#                     sample['token'] = tokens
#                     sample['sent_start'] = sent_start
#                     sample['sent_end'] = sent_end

#                     sent_samples.append(sample)

#             max_sentsample = max(max_sentsample, len(sent_samples))
#             samples += sent_samples
    
#     tot = len(samples)
#     logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

#     return data, samples, nrel
