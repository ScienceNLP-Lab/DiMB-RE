from collections import Counter
import logging

from shared.data_structures import Dataset

logger = logging.getLogger('root')

CLS = "[CLS]"
SEP = "[SEP]"


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx, trg_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
        self.trg_idx = trg_idx


def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))
    trg = (int(pair[2][1:-1].split(',')[0]), int(pair[2][1:-1].split(',')[1]))
    return doc_sent, sub, obj, trg


def generate_trigger_data(
        entity_data, 
        ref_data=None, 
        use_gold=False, 
        context_window=100, 
        binary_classification=True
):
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
                        sample['id'] = '%s@%d::(%d,%d)-(%d,%d)-(%d,%d)'%(
                            doc._doc_key, sent.sentence_ix, 
                            sub.span.start_doc, sub.span.end_doc,
                            obj.span.start_doc, obj.span.end_doc,
                            trg.span.start_doc, trg.span.end_doc
                        )
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


def convert_examples_to_triplet_features(examples, label2id, tokenizer, special_tokens, args, unused_tokens=True):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>')
        return special_tokens[w]

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]

        TRIGGER_START_NER = get_special_token("TRG_START=%s"%example['trg_type'])
        TRIGGER_END_NER = get_special_token("TRG_END=%s"%example['trg_type'])
        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])

        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            if i == example['trg_start']:
                trg_idx = len(tokens)
                tokens.append(TRIGGER_START_NER)
            # for sub_token in tokenizer.tokenize(token):
            for sub_token in tokenizer.tokenize(token.lower()):
                tokens.append(sub_token)
            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
            if i == example['trg_end']:
                tokens.append(TRIGGER_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > args.max_seq_length:
            tokens = tokens[:args.max_seq_length]
            if sub_idx >= args.max_seq_length:
                sub_idx = 0
            if obj_idx >= args.max_seq_length:
                obj_idx = 0
            if trg_idx >= args.max_seq_length:
                trg_idx = 0
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example['id']))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example['relation'], label_id))
                logger.info("sub_idx, obj_idx, trg_idx: %d, %d, %d" % (sub_idx, obj_idx, trg_idx))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                sub_idx=sub_idx,
                obj_idx=obj_idx,
                trg_idx=trg_idx
            )
        )
        
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d"%max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), args.max_seq_length))
    
    return features

