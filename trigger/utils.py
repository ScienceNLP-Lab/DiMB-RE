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
    trg = (int(pair[2][1:-1].split(',')[0]), int(pair[2][1:-1].split(',')[1]))
    return doc_sent, sub, obj, trg

def generate_trigger_data(entity_data, ref_data=None, use_gold=False, context_window=0, binary_classification=False):
    """
    Prepare data for the trigger-entity pairs model
    If training: set use_gold = True
    """
    logger.info('Generate Trigger-Entity Triplets from %s'%(entity_data))
    data = Dataset(entity_data)

    if ref_data is not None:
        ref_data = Dataset(ref_data)

    ntriplets = 0
    ntriplets_sent_level = 0
    max_sentsample = 0
    label_cnt_dict = Counter()

    samples = []
    for doc, ref_doc in zip(data, ref_data):
        for i, (sent, ref_sent) in enumerate(zip(doc, ref_doc)):

            sent_samples = []

            ntriplets += len(sent.triplets)

            if use_gold:
                sent_ner = ref_sent.ner
                sent_trg = ref_sent.triggers
            else:
                sent_ner = sent.predicted_ner
                sent_trg = sent.predicted_triggers
            
            gold_triplet = {}
            for triplet in ref_sent.triplets:
                if not binary_classification:
                    gold_triplet[triplet.triplet] = triplet.label
                    label_cnt_dict[triplet.label] += 1.0
                else:
                    gold_triplet[triplet.triplet] = "Entailment"
                    label_cnt_dict["Entailment"] += 1.0
            
            # Context Window
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
            
            for x in range(len(sent_trg)):
                trg = sent_trg[x]
                for y in range(len(sent_ner)):
                    for z in range(len(sent_ner)):

                        if y == z:
                            continue

                        sub = sent_ner[y]
                        obj = sent_ner[z]
                        label = gold_triplet.get((sub.span, obj.span, trg.span), 'no_relation')

                        if label != "no_relation":
                            ntriplets_sent_level += 1

                        sample = {}
                        sample['docid'] = doc._doc_key
                        sample['id'] = '%s@%d::(%d,%d)-(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, \
                                                                 sub.span.start_doc, sub.span.end_doc, \
                                                                    obj.span.start_doc, obj.span.end_doc, \
                                                                        trg.span.start_doc, trg.span.end_doc)
                        sample['relation'] = label
                        sample['subj_start'] = sub.span.start_sent + sent_start
                        sample['subj_end'] = sub.span.end_sent + sent_start
                        sample['subj_type'] = sub.label
                        sample['obj_start'] = obj.span.start_sent + sent_start
                        sample['obj_end'] = obj.span.end_sent + sent_start
                        sample['obj_type'] = obj.label
                        sample['trg_start'] = trg.span.start_sent + sent_start
                        sample['trg_end'] = trg.span.end_sent + sent_start

                        if binary_classification:
                            sample['trg_type'] = trg.label
                        else:
                            sample['trg_type'] = "TRIGGER"

                        sample['token'] = tokens
                        sample['sent_start'] = sent_start
                        sample['sent_end'] = sent_end

                        sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    ntriplets_invalid = ntriplets - ntriplets_sent_level
    if not ntriplets_invalid:
        print(f"All Triggers are within Sent-level!")
    else: 
        print(f"# of Invalid Triggers within Sent-level: {ntriplets_invalid}({round(ntriplets_invalid/ntriplets*100, 4)}%)")
        print("Only useful with GOLD ENTITIES")

    return data, samples, ntriplets, label_cnt_dict
