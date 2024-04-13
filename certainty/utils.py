from collections import Counter
import logging

from shared.data_structures_copied import Dataset

logger = logging.getLogger('root')

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))
    return doc_sent, sub, obj

def generate_certainty_data(relation_data, ref_data=None, training=True, use_gold=False, use_trigger=True, context_window=0):
    """
    Prepare data for the negation detection model
    If training: set use_gold = True
    """
    logger.info('Generate certainty data from %s'%(relation_data))
    data = Dataset(relation_data)
    
    if ref_data is not None:
        ref_data = Dataset(ref_data)

    nrel = 0
    nrel_sent_level = 0
    max_sentsample = 0
    label_cnt_dict = Counter()

    samples = []
    for doc, ref_doc in zip(data, ref_data):
        for i, (sent, ref_sent) in enumerate(zip(doc, ref_doc)):

            sent_samples = []
            nrel += len(ref_sent.relations)

            if use_gold:
                sent_relations = ref_sent.relations
                sent_ner = ref_sent.ner
                sent_triggers = ref_sent.triggers
                sent_triplets = ref_sent.triplets
            else:
                sent_relations = sent.predicted_relations
                sent_ner = sent.predicted_ner
                sent_triggers = sent.predicted_triggers
                sent_triplets = sent.predicted_triplets
            
            gold_certainty = {}
            for rel in ref_sent.relations:
                gold_certainty[rel.pair] = rel.certainty
                label_cnt_dict[rel.certainty] += 1.0

            span_ner = {}
            for ner in sent_ner:
                # set doc-level span as key, ner class as label
                span_ner[ner.span.span_doc] = ner

            span_triggers = {}
            for trg in sent_triggers:
                span_triggers[trg.span.span_doc] = trg
            
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
            
            # print("TRIGGERS >>> ", span_triggers)
            # print("TRIPLETS >>> ", sent_triplets)
            # print("RELATIONS >>> ", sent_relations)
            for idx, relation in enumerate(sent_relations):
                # cross-sentence relation
                if not (relation.pair[0].start_doc, relation.pair[0].end_doc) in span_ner:
                    continue
                if not (relation.pair[1].start_doc, relation.pair[1].end_doc) in span_ner:
                    continue

                sub = span_ner[(relation.pair[0].start_doc, relation.pair[0].end_doc)]
                obj = span_ner[(relation.pair[1].start_doc, relation.pair[1].end_doc)]
                
                label = gold_certainty.get(relation.pair, 'no_certainty')
                if label != "no_certainty":
                    nrel_sent_level += 1
                
                span_result = None
                if use_trigger:
                    # print(idx)
                    
                    # start and end span of trigger for the relation
                    span_result = span_triggers[sent_triplets[idx].triplet[-1].span_doc]

                sample = {}
                sample['docid'] = doc._doc_key
                sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                sample['certainty'] = label
                sample['subj_start'] = sub.span.start_sent + sent_start
                sample['subj_end'] = sub.span.end_sent + sent_start
                sample['subj_type'] = sub.label
                sample['obj_start'] = obj.span.start_sent + sent_start
                sample['obj_end'] = obj.span.end_sent + sent_start
                sample['obj_type'] = obj.label
                sample['rel_type'] = relation.label

                if use_trigger:
                    sample['trg_start'] = span_result.span.start_sent + sent_start
                    sample['trg_end'] = span_result.span.end_sent + sent_start

                sample['token'] = tokens
                sample['sent_start'] = sent_start
                sample['sent_end'] = sent_end
                sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    nrel_invalid = nrel - nrel_sent_level
    if not nrel_invalid:
        print(f"All Relations are within Sent-level!")
    else: 
        print(f"# of Invalid Relations within Sent-level: {nrel_invalid}({round(nrel_invalid/nrel*100, 4)}%)")

    return data, samples, nrel, label_cnt_dict

