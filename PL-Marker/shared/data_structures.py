"""
This code is based on DYGIE++'s codebase
"""
import json
import copy
import os
from collections import Counter
import numpy as np
from shared.const import task_ner_labels, task_rel_labels

def fields_to_batches(d, keys_to_ignore=[]):
    keys = [key for key in d.keys() if key not in keys_to_ignore]
    lengths = [len(d[k]) for k in keys]
    assert len(set(lengths)) == 1
    length = lengths[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res

def get_sentence_of_span(span, sentence_starts, doc_tokens):
    """
    Return the index of the sentence that the span is part of.
    """
    # Inclusive sentence ends
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [span[0] >= start and span[1] <= end
                  for start, end in zip(sentence_starts, sentence_ends)]
    assert sum(in_between) == 1
    the_sentence = in_between.index(True)
    return the_sentence

def find_overlapped_spans(span1, span2):
    if span1[1] >= span2[0] and span1[0] <= span2[1]:
        return True
    return False

class Dataset:
    def __init__(self, json_file, pred_file=None, doc_range=None):
        self.js = self._read(json_file, pred_file)
        if doc_range is not None:
            self.js = self.js[doc_range[0]:doc_range[1]]
        self.documents = [Document(js) for js in self.js]

    def update_from_js(self, js):
        self.js = js
        self.documents = [Document(js) for js in self.js]

    def _read(self, json_file, pred_file=None):
        gold_docs = [json.loads(line) for line in open(json_file)]
        if pred_file is None:
            return gold_docs

        pred_docs = [json.loads(line) for line in open(pred_file)]
        merged_docs = []
        for gold, pred in zip(gold_docs, pred_docs):
            assert gold["doc_key"] == pred["doc_key"]
            assert gold["sentences"] == pred["sentences"]
            merged = copy.deepcopy(gold)
            for k, v in pred.items():
                if "predicted" in k:
                    merged[k] = v
            merged_docs.append(merged)

        return merged_docs

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)


class Document:
    def __init__(self, js):
        self._doc_key = js["doc_key"]
        entries = fields_to_batches(
            js, ["doc_key", "clusters", "predicted_clusters", "section_starts"]
        )
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        self.sentence_starts = sentence_starts
        self.sentences = [Sentence(entry, sentence_start, sentence_ix)
                          for sentence_ix, (entry, sentence_start)
                          in enumerate(zip(entries, sentence_starts))]

    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))

    def find_cluster(self, entity, predicted=True):
        """
        Search through erence clusters and return the one containing the query entity, if it's
        part of a cluster. If we don't find a match, return None.
        """
        clusters = self.predicted_clusters if predicted else self.clusters
        for clust in clusters:
            for entry in clust:
                if entry.span == entity.span:
                    return clust

        return None

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix):
        self.sentence_start = sentence_start
        self.text = entry["sentences"]
        self.sentence_ix = sentence_ix

        # Gold
        if "ner_flavor" in entry:
            self.ner = [NER(this_ner, self.text, sentence_start, flavor=this_flavor)
                        for this_ner, this_flavor in zip(entry["ner"], entry["ner_flavor"])]
        elif "ner" in entry:
            self.ner = [NER(this_ner, self.text, sentence_start)
                        for this_ner in entry["ner"]]
        if "triggers" in entry:
            self.triggers = [NER(this_trg, self.text, sentence_start) 
                             for this_trg in entry["triggers"]]
        if "relations" in entry:
            self.relations = [Relation(this_relation, self.text, sentence_start) for
                              this_relation in entry["relations"]]
        # if "triplets" in entry:
        #     self.triplets = [Triplet(this_triplet, self.text, sentence_start) for 
        #                                   this_triplet in entry["triplets"]]

        # Predicted
        if "predicted_ner" in entry:
            self.predicted_ner = [NER(this_ner, self.text, sentence_start, flavor=None) for
                                  this_ner in entry["predicted_ner"]]
        else:
            self.predicted_ner = self.ner
        if "predicted_triggers" in entry:
            self.predicted_triggers = [NER(this_trg, self.text, sentence_start, flavor=None) for
                                  this_trg in entry["predicted_triggers"]]
        else:
            self.predicted_triggers = self.triggers            
        if "predicted_relations" in entry:
            self.predicted_relations = [Relation(this_relation, self.text, sentence_start) for
                                        this_relation in entry["predicted_relations"]]
        # if "predicted_triplets" in entry:
        #     self.predicted_triplets = [Triplet(this_pair, self.text, sentence_start) for 
        #                                   this_pair in entry["predicted_triplets"]]


    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)

    def get_flavor(self, argument):
        the_ner = [x for x in self.ner if x.span == argument.span]
        if len(the_ner) > 1:
            print("Weird")
        if the_ner:
            the_flavor = the_ner[0].flavor
        else:
            the_flavor = None
        return the_flavor


class Span:
    def __init__(self, start, end, text, sentence_start):
        self.start_doc = start
        self.end_doc = end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = start - sentence_start
        self.end_sent = end - sentence_start
        self.span_sent = (self.start_sent, self.end_sent)
        if self.start_sent >= 0 and self.end_sent >= 0:
            self.text = text[self.start_sent:self.end_sent + 1]
        else:
            self.text = "Out-of-sentence"

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.text == other.text)

    def __hash__(self):
        tup = self.span_doc + self.span_sent + (" ".join(self.text),)
        return hash(tup)


class Token:
    def __init__(self, ix, text, sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))

class Argument:
    def __init__(self, span, role, event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return self.span.__repr__()[:-1] + ", " + self.event_type + ", " + self.role + ")"

    def __eq__(self, other):
        return (self.span == other.span and
                self.role == other.role and
                self.event_type == other.event_type)

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))


class NER:
    def __init__(self, ner, text, sentence_start, flavor=None):
        self.span = Span(ner[0], ner[1], text, sentence_start)
        self.label = ner[2]
        self.flavor = flavor

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label and
                self.flavor == other.flavor)
    
    def __hash__(self):
        return hash((self.span, self.label))
    
class Triplet:
    def __init__(self, triplet, text, sentence_start):
        start1, end1 = triplet[0], triplet[1]
        start2, end2 = triplet[2], triplet[3]
        start_trg, end_trg = triplet[4], triplet[5]
        span1 = Span(start1, end1, text, sentence_start)
        span2 = Span(start2, end2, text, sentence_start)
        span_trg = Span(start_trg, end_trg, text, sentence_start)
        self.triplet = (span1, span2, span_trg)

        if len(triplet) > 6:
            self.label = triplet[6]

    def __repr__(self):
        return self.triplet[0].__repr__() + ";" + self.triplet[1].__repr__() + ";" + self.triplet[2].__repr__()

    def __eq__(self, other):
        return (self.triplet == other.triplet) and (self.label == other.label)


class Relation:
    def __init__(self, relation, text, sentence_start):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1, end1, text, sentence_start)
        span2 = Span(start2, end2, text, sentence_start)
        self.pair = (span1, span2)
        self.flipped_pair = (span2, span1)
        self.label = label
        if len(relation) == 5:
            self.certainty = ""  # Placeholder
        else:
            self.certainty = relation[5]

    def __repr__(self):
        return self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label

    def __eq__(self, other):
        if isinstance(other, Relation):
            return (self.pair == other.pair) and (self.label == other.label)
        return False
    
    def __hash__(self):
        # Return a hash value based on a tuple of the attributes
        return hash((self.pair[0], self.pair[1]))
    
    def flipped_match(self, other, mode='strict'):
        if mode == 'strict':
            return (self.flipped_pair == other.pair) and (self.label == other.label)
        if mode == 'relaxed':
            return self.find_overlap(other, mode='flipped') and (self.label == other.label)
        
    def find_overlap(self, other, mode='ordered'):
        if mode == 'ordered':
            if find_overlapped_spans(self.pair[0].span_sent, other.pair[0].span_sent) and \
            find_overlapped_spans(self.pair[1].span_sent, other.pair[1].span_sent):
                return True            
        else:
            if find_overlapped_spans(self.pair[1].span_sent, other.pair[0].span_sent) and \
            find_overlapped_spans(self.pair[0].span_sent, other.pair[1].span_sent):
                return True
        return False


####################################################################################################

# Code to do evaluation of predictions for a loaded dataset.

def safe_div(num, denom):
    if denom > 0:
        return round(num/denom, 4)
    else:
        return 0

def compute_f1(predicted, gold, matched):
    # F1 score.
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    result = dict(
        precision=precision, 
        recall=recall, 
        f1=f1,
        n_gold=gold,
        n_pred=predicted,
        n_correct=matched
    )
    return result

    
def remove_nested_same_type(sent):
    '''Choose longer mention if nested mention has the same label'''

    predicted_ner_sorted = sorted(
        [ner for ner in sent.predicted_ner],
        key=lambda x: x.span.span_sent[1] - x.span.span_sent[0],
        reverse=True
    )
    predicted_trg_sorted = sorted(
        [trg for trg in sent.predicted_triggers],
        key=lambda x: x.span.span_sent[1] - x.span.span_sent[0],
        reverse=True
    )

    checker = [[] for _ in range(len(sent.text))]

    predicted_ner_processed = []
    for ner in predicted_ner_sorted:
        dup_flag = False
        for lst in checker[ner.span.span_sent[0]:ner.span.span_sent[1] + 1]:
            if ner.label in lst:
                dup_flag = True
                break
        if not dup_flag:
            predicted_ner_processed.append(ner)
            for lst in checker[ner.span.span_sent[0]:ner.span.span_sent[1] + 1]:
                lst.append(ner.label)

    predicted_trg_processed = []
    for trg in predicted_trg_sorted:
        dup_flag = False
        for lst in checker[trg.span.span_sent[0]:trg.span.span_sent[1] + 1]:
            if trg.label in lst:
                dup_flag = True
                break
        if not dup_flag:
            predicted_trg_processed.append(trg)
            for lst in checker[trg.span.span_sent[0]:trg.span.span_sent[1] + 1]:
                lst.append(trg.label)

    return predicted_ner_processed


def evaluate_sent(
        sent, 
        counts, 
        ner_result_by_class, 
        rel_result_by_class, 
        dataset_name,
        do_eval_rel=True
):
    
    errors = {
        'tp-ner-em':[],
        'fp-ner-em':[],
        'fn-ner-em':[],
        'tp-ner-relaxed':[],
        'fp-ner-relaxed':[],
        'fn-ner-relaxed':[],
        'tp-trg-em':[],
        'fp-trg-em':[],
        'fn-trg-em':[],
        'tp-trg-relaxed':[],
        'fp-trg-relaxed':[],
        'fn-trg-relaxed':[],        
        'tp-rel-em':[],
        'fp-rel-em':[],
        'fn-rel-em':[],
        'tp-rel-relaxed':[],
        'fp-rel-relaxed':[],
        'fn-rel-relaxed':[]
    }

    ##################################################
    ####### Evaluate NER & Trigger Extraction ########
    ##################################################

    correct_ner = set()
    correct_ner_partial = set()

    # TODO: keep it or not?
    # If activated, predictions exclude the nested mentions with same entity type
    # sent.predicted_ner, sent.predicted_triggers = remove_nested_same_type(sent=sent)
    
    gold_ner_em = {}
    pred_ner_em = {}
    gold_ner_relaxed = {}
    pred_ner_relaxed = {}

    # Entities and Triggers
    for ner in sent.ner:
        ner_result_by_class[ner.label]['gold'] += 1.0
        gold_ner_em[ner] = "FN"
        gold_ner_relaxed[ner] = "FN"
    for ner in sent.predicted_ner:
        ner_result_by_class[ner.label]['pred'] += 1.0
        ner_result_by_class[ner.label]['pred_relaxed'] += 1.0
        pred_ner_em[ner] = "FP"
        pred_ner_relaxed[ner] = "FP"

    gold_trg_em = {}
    pred_trg_em = {}
    gold_trg_relaxed = {}
    pred_trg_relaxed = {}

    for ner in sent.triggers:
        if dataset_name.endswith('dummy'):
            ner.label = "TRIGGER"
        ner_result_by_class[ner.label]['gold'] += 1.0
        gold_trg_em[ner] = "FN"
        gold_trg_relaxed[ner] = "FN"
        
    for ner in sent.predicted_triggers:
        ner_result_by_class[ner.label]['pred'] += 1.0
        ner_result_by_class[ner.label]['pred_relaxed'] += 1.0
        pred_trg_em[ner] = "FP"
        pred_trg_relaxed[ner] = "FP"

    counts["ner_gold"] += len(sent.ner)
    counts["ner_predicted"] += len(sent.predicted_ner)
    counts["ner_relaxed_predicted"] += len(sent.predicted_ner)

    counts["trigger_gold"] += len(sent.triggers)
    counts["trigger_predicted"] += len(sent.predicted_triggers)
    counts["trigger_relaxed_predicted"] += len(sent.predicted_triggers)

    ### Entity Evaluation ###

    # tp_ner_relaxed = []
    # fp_ner_relaxed = []
    # fn_ner_relaxed = []
    # tp_ner_em = []
    # fp_ner_em = []
    # fn_ner_em = []

    partial_match_ner = []
    for prediction in sent.predicted_ner:

        flag_ner_em, flag_ner_pm = False, False

        # Strict Matching
        for actual in sent.ner:
            if prediction == actual:
                counts["ner_matched"] += 1
                ner_result_by_class[prediction.label]['correct'] += 1.0
                correct_ner.add(prediction.span)
                flag_ner_em = True
                # tp_ner_em.append((
                #     prediction.span.start_sent, 
                #     prediction.span.end_sent, 
                #     prediction.label
                # ))
                gold_ner_em[actual] = "TP"
                pred_ner_em[prediction] = "TP"
                break

        # Partial Matching
        # Remove duplicated prediction -> 1:1 mapping for every gold standard
        # But consider nested gold cases whose concepts are same
        is_duplicated = False
        for actual in sent.ner:
            if (actual.label == prediction.label) and \
            find_overlapped_spans(actual.span.span_sent, prediction.span.span_sent):
                if actual not in partial_match_ner:
                    partial_match_ner.append(actual)
                    counts["ner_partial_matched"] += 1
                    ner_result_by_class[prediction.label]['correct_relaxed'] += 1.0
                    correct_ner_partial.add(prediction.span)  # Add span of prediction for RE evaluation
                    is_duplicated = False
                    flag_ner_pm = True
                    # if not flag_ner_em:
                    #     tp_ner_relaxed.append((
                    #         actual.span.start_sent, 
                    #         actual.span.end_sent, 
                    #         actual.label
                    #     ))
                    gold_ner_relaxed[actual] = "TP"
                    pred_ner_relaxed[prediction] = "TP"
                    break
                else:
                    # This can happen even though we remove nested prediction with same type
                    # e.g. PRED >> [120, 120, "Physiology"], [123, 124, "Physiology"]
                    # GOLD >> [120, 124, "Physiology"]
                    is_duplicated = True
                    correct_ner_partial.add(prediction.span)

        if is_duplicated:
            counts["ner_relaxed_predicted"] -= 1
            ner_result_by_class[prediction.label]['pred_relaxed'] -= 1.0
            del pred_ner_relaxed[prediction]

        # Below codes are for error analysis text file
        # if not flag_ner_em:
        #     if not flag_ner_pm:
        #         fp_ner_relaxed.append((
        #             prediction.span.start_sent, 
        #             prediction.span.end_sent, 
        #             prediction.label
        #         ))
        #     else:
        #         fp_ner_em.append((
        #             prediction.span.start_sent, 
        #             prediction.span.end_sent, 
        #             prediction.label
        #         ))

    # for actual in sent.ner:
    #     decomposed = (
    #         actual.span.start_sent, 
    #         actual.span.end_sent, 
    #         actual.label
    #     )
    #     if decomposed not in tp_ner_em:
    #         if decomposed not in tp_ner_relaxed:
    #             fn_ner_relaxed.append(decomposed)
    #         else:
    #             fn_ner_em.append(decomposed)

    tp_ner_relaxed = [(
        ent.span.start_sent,
        ent.span.end_sent,
        ent.label
    ) for ent, tag, in pred_ner_relaxed.items() if tag == "TP"]
    fp_ner_relaxed = [(
        ent.span.start_sent,
        ent.span.end_sent,
        ent.label
    ) for ent, tag, in pred_ner_relaxed.items() if tag == "FP"]
    fn_ner_relaxed = [(
        ent.span.start_sent,
        ent.span.end_sent,
        ent.label
    ) for ent, tag, in gold_ner_relaxed.items() if tag == "FN"]
    tp_ner_em = [(
        ent.span.start_sent,
        ent.span.end_sent,
        ent.label
    ) for ent, tag, in gold_ner_em.items() if tag == "TP"]
    fp_ner_em = [(
        ent.span.start_sent,
        ent.span.end_sent,
        ent.label
    ) for ent, tag, in pred_ner_em.items() if tag == "FP"]
    fn_ner_em = [(
        ent.span.start_sent,
        ent.span.end_sent,
        ent.label
    ) for ent, tag, in gold_ner_em.items() if tag == "FN"]

    errors['tp-ner-relaxed'].extend(tp_ner_relaxed)
    errors['fp-ner-relaxed'].extend(fp_ner_relaxed)
    errors['fn-ner-relaxed'].extend(fn_ner_relaxed)
    errors['tp-ner-em'].extend(tp_ner_em)
    errors['fp-ner-em'].extend(fp_ner_em)
    errors['fn-ner-em'].extend(fn_ner_em)

                
    ### Trigger Evaluation ###
    partial_match_trg = []
    for prediction in sent.predicted_triggers:

        flag_trg_em, flag_trg_pm = False, False

        # Strict Matching: Same as NER
        for actual in sent.triggers:
            if prediction == actual:
                counts["trigger_matched"] += 1
                ner_result_by_class[prediction.label]['correct'] += 1.0
                flag_trg_em = True
                gold_trg_em[actual] = "TP"
                pred_trg_em[prediction] = "TP"
                break

        # Partial Matching: Same as NER
        is_duplicated = False
        for actual in sent.triggers:
            if (actual.label == prediction.label) and \
            find_overlapped_spans(actual.span.span_sent, prediction.span.span_sent):
                if actual not in partial_match_trg:
                    partial_match_trg.append(actual)
                    counts["trigger_partial_matched"] += 1
                    ner_result_by_class[prediction.label]['correct_relaxed'] += 1.0
                    is_duplicated = False
                    flag_trg_pm = True
                    gold_trg_relaxed[actual] = "TP"
                    pred_trg_relaxed[prediction] = "TP"
                    break
                else:
                    is_duplicated = True

        if is_duplicated:
            counts["ner_relaxed_predicted"] -= 1
            ner_result_by_class[prediction.label]['pred_relaxed'] -= 1.0


    tp_trg_relaxed = [(
        trg.span.start_sent,
        trg.span.end_sent,
        trg.label
    ) for trg, tag, in pred_trg_relaxed.items() if tag == "TP"]
    fp_trg_relaxed = [(
        trg.span.start_sent,
        trg.span.end_sent,
        trg.label
    ) for trg, tag, in pred_trg_relaxed.items() if tag == "FP"]
    fn_trg_relaxed = [(
        trg.span.start_sent,
        trg.span.end_sent,
        trg.label
    ) for trg, tag, in gold_trg_relaxed.items() if tag == "FN"]
    tp_trg_em = [(
        trg.span.start_sent,
        trg.span.end_sent,
        trg.label
    ) for trg, tag, in gold_trg_em.items() if tag == "TP"]
    fp_trg_em = [(
        trg.span.start_sent,
        trg.span.end_sent,
        trg.label
    ) for trg, tag, in pred_trg_em.items() if tag == "FP"]
    fn_trg_em = [(
        trg.span.start_sent,
        trg.span.end_sent,
        trg.label
    ) for trg, tag, in gold_trg_em.items() if tag == "FN"]

    errors['tp-trg-relaxed'].extend(tp_trg_relaxed)
    errors['fp-trg-relaxed'].extend(fp_trg_relaxed)
    errors['fn-trg-relaxed'].extend(fn_trg_relaxed)
    errors['tp-trg-em'].extend(tp_trg_em)
    errors['fp-trg-em'].extend(fp_trg_em)
    errors['fn-trg-em'].extend(fn_trg_em)


    # partial_match_trg = []
    # for actual in sent.triggers:
    #     if actual.label == "TRIGGER":
    #         # Strict Matching
    #         if any([prediction.span.span_sent == actual.span.span_sent for prediction in sent.predicted_triggers]):
    #             counts["trigger_matched"] += 1
    #             ner_result_by_class[actual.label]['correct'] += 1.0
    #         # Partial Matching: Find overlapped spans
    #         for prediction in sent.predicted_triggers:
    #             if find_overlapped_spans(actual.span.span_sent, prediction.span.span_sent):
    #                 if actual not in partial_match_trg:
    #                     partial_match_trg.append(actual)
    #                     counts["trigger_partial_matched"] += 1
    #                     ner_result_by_class[actual.label]['correct_relaxed'] += 1.0
    #                 else:
    #                     counts["trigger_relaxed_predicted"] -= 1
    #                     ner_result_by_class[actual.label]['pred_relaxed'] -= 1.0                        
    #                 break
    #     else:
    #         # Strict Matching
    #         if any([prediction == actual for prediction in sent.predicted_triggers]):
    #             counts["trigger_matched"] += 1
    #             ner_result_by_class[actual.label]['correct'] += 1.0
    #         # Partial Matching: Find overlapped spans
    #         for prediction in sent.predicted_triggers:
    #             if find_overlapped_spans(actual.span.span_sent, prediction.span.span_sent) \
    #                 and actual.label == prediction.label:
    #                 if actual not in partial_match_trg:
    #                     partial_match_trg.append(actual)
    #                     counts["trigger_partial_matched"] += 1
    #                     ner_result_by_class[actual.label]['correct_relaxed'] += 1.0
    #                 else:
    #                     # This can happen even though we remove nested prediction with same type
    #                     counts["trigger_relaxed_predicted"] -= 1
    #                     ner_result_by_class[actual.label]['pred_relaxed'] -= 1.0                        
    #                 break


    ##################################################
    ################## Evaluate RE ###################
    ##################################################

    if not do_eval_rel:
        return counts, ner_result_by_class, rel_result_by_class, errors

    gold_ner = {}
    pred_ner = {}

    gold_rel_em = {}
    pred_rel_em = {}
    gold_rel_relaxed = {}
    pred_rel_relaxed = {}

    for ner in sent.ner:
        gold_ner[(ner.span.start_sent, ner.span.end_sent)] = ner.label
    for ner in sent.predicted_ner:
        pred_ner[(ner.span.start_sent, ner.span.end_sent)] = ner.label

    for rel in sent.relations:
        rel_result_by_class[rel.label]['gold'] += 1.0
        counts["relations_gold"] += 1
        gold_rel_em[rel] = "FN"
        gold_rel_relaxed[rel] = "FN"
    for rel in sent.predicted_relations:
        rel_result_by_class[rel.label]['pred'] += 1.0
        rel_result_by_class[rel.label]['pred_relaxed'] += 1.0
        counts["relations_predicted"] += 1
        counts["relations_relaxed_predicted"] += 1
        pred_rel_em[rel] = "FP"
        pred_rel_relaxed[rel] = "FP"

    relaxed_match_actuals = []
    true_positives_bidi_strict = []
    true_positives_bidi_relaxed = []
    for prediction in sent.predicted_relations:

        relaxed_match_actual = ""
        strict_match_actual = ""

        pred_sub = prediction.pair[0]
        pred_obj = prediction.pair[1]

        # if pred is bidirectional and already counted,
        # we don't count it to avoid duplicated cases
        flag_relaxed_match = True
        for tp in true_positives_bidi_relaxed:
            if prediction.flipped_match(tp):
                flag_relaxed_match = False
                # print(f"Already counted as TP-bidi-relaxed >> PRED:{prediction}, TP:{tp}")
                counts["relations_relaxed_predicted"] -= 1
                rel_result_by_class[prediction.label]['pred_relaxed'] -= 1.0
                del pred_rel_relaxed[prediction]
                break

        flag_strict_match = True
        for tp in true_positives_bidi_strict:
            if prediction.flipped_match(tp):
                flag_strict_match = False
                # print(f"Already counted as TP-bidi-strict >> PRED:{prediction}, TP:{tp}")
                counts["relations_predicted"] -= 1
                rel_result_by_class[prediction.label]['pred'] -= 1.0
                del pred_rel_em[prediction]
                break

        if flag_relaxed_match:
            is_duplicated = False
            for actual in sent.relations:

                gold_sub = actual.pair[0]
                gold_obj = actual.pair[1]

                # First, do Relaxed Match
                # 1) Relation Label Match
                # 2) Relaxed boundary match of entity spans
                # 3) Evaluate whether it's correct entity
                # 4) Entity Label Match

                if actual.label == prediction.label and \
                    actual.find_overlap(prediction) and \
                    (pred_sub in correct_ner_partial) \
                    and (pred_obj in correct_ner_partial) \
                    and (pred_ner[
                        (pred_sub.start_sent, pred_sub.end_sent)
                    ] == gold_ner[
                        (gold_sub.start_sent, gold_sub.end_sent)
                    ]) \
                    and (pred_ner[
                        (pred_obj.start_sent, pred_obj.end_sent)
                    ] == gold_ner[
                        (gold_obj.start_sent, gold_obj.end_sent)
                    ]):

                    if actual not in relaxed_match_actuals:
                        # print(f"## Soft match >>> PRED:{prediction}, GOLD:{actual}")
                        relaxed_match_actuals.append(actual)
                        relaxed_match_actual = actual
                        counts["relaxed_relations_matched"] += 1
                        rel_result_by_class[prediction.label]['correct_relaxed'] += 1.0
                        
                        gold_rel_relaxed[actual] = "TP"
                        pred_rel_relaxed[prediction] = "TP"                        
                        
                        # Factuality matching
                        if actual.certainty == prediction.certainty:
                            counts["relaxed_relations_matched_fact"] += 1
                            rel_result_by_class[prediction.label]['correct_relaxed_fact'] += 1.0
                            # gold_rel_relaxed[actual] = "TP"
                            # pred_rel_relaxed[prediction] = "TP"
                        is_duplicated = False
                        break
                    else:
                        is_duplicated = True

            # After looping all gold standards, we should remove a duplicated pred
            # if it is not matched with other gold standards.
            if is_duplicated:
                # print(f"## Duplicated soft match >> PRED:{prediction} | GOLD:{actual} | Duplicated:{relaxed_match_actuals}")
                counts["relaxed_relations_predicted"] -= 1
                rel_result_by_class[prediction.label]['pred_relaxed'] -= 1.0
                del pred_rel_relaxed[prediction]

            # Store TP if it finds bidirectional match in the gold standards.
            if relaxed_match_actual:
                for actual2 in sent.relations:
                    if relaxed_match_actual.flipped_match(actual2, mode='relaxed'):
                        # print(f"bidi-tp-relaxed >> PRED_Matched:{relaxed_match_actual}, GOLD:{actual2}")
                        true_positives_bidi_relaxed.append(relaxed_match_actual)
                        break

        # Strict boundary match
        if flag_strict_match:
            for actual in sent.relations:
                gold_sub = actual.pair[0]
                gold_obj = actual.pair[1] 
                if actual == prediction and \
                    (pred_sub in correct_ner) \
                    and (pred_obj in correct_ner) \
                    and (
                        pred_ner[
                            (pred_sub.start_sent, pred_sub.end_sent)
                        ] == gold_ner[
                            (gold_sub.start_sent, gold_sub.end_sent)
                        ]
                    ) \
                    and (
                        pred_ner[
                            (pred_obj.start_sent, pred_obj.end_sent)
                        ] == gold_ner[
                            (gold_obj.start_sent, gold_obj.end_sent)
                        ]
                    ):
                    # print(f"## Strict match >> PRED:{prediction}, GOLD:{actual}")
                    counts["strict_relations_matched"] += 1
                    rel_result_by_class[prediction.label]['correct'] += 1.0
                    
                    gold_rel_em[actual] = "TP"
                    pred_rel_em[prediction] = "TP"                    

                    if actual.certainty == prediction.certainty:
                        counts["strict_relations_matched_fact"] += 1
                        rel_result_by_class[prediction.label]['correct_fact'] += 1.0
                        # gold_rel_em[actual] = "TP"
                        # pred_rel_em[prediction] = "TP"
                        # # In case if you want to print relaxed-only analysis
                        # gold_rel_relaxed[actual] = "TP"
                        # pred_rel_relaxed[prediction] = "TP"    

                    strict_match_actual = actual
                    break

            if strict_match_actual:
                for actual2 in sent.relations:
                    if strict_match_actual.flipped_match(actual2):
                        # print(f"## bidi-tp-strict >> PRED_Matched:{strict_match_actual}, GOLD:{actual2}")
                        true_positives_bidi_strict.append(strict_match_actual)
                        break

    # Count the number of gold relations
    # to consider the bidi-relations as the same one
    counted_bidi = []
    for actual1 in sent.relations:
        if actual1 in counted_bidi:
            continue     
        for actual2 in sent.relations:
            if actual1 == actual2:
                continue
            if actual1.flipped_match(actual2):
                counted_bidi.append(actual2)
                # print("## Duplicated_bidi_gold >>>", actual1, actual2)
                counts["relations_gold"] -= 1
                rel_result_by_class[actual2.label]['gold'] -= 1.0
                del gold_rel_relaxed[actual2]
                del gold_rel_em[actual2]
                break

    tp_rel_relaxed = [(
        rel.label,
        rel.pair[0].text,
        rel.pair[1].text,
        rel.certainty
    ) for rel, tag in pred_rel_relaxed.items() if tag == "TP"] 
    fp_rel_relaxed = [(
        rel.label,
        rel.pair[0].text,
        rel.pair[1].text,
        rel.certainty
    ) for rel, tag in pred_rel_relaxed.items() if tag == "FP"] 
    fn_rel_relaxed = [(
        rel.label,
        rel.pair[0].text,
        rel.pair[1].text,
        rel.certainty
    ) for rel, tag in gold_rel_relaxed.items() if tag == "FN"] 
    tp_rel_em = [(
        rel.label,
        rel.pair[0].text,
        rel.pair[1].text,
        rel.certainty
    ) for rel, tag in gold_rel_em.items() if tag == "TP"] 
    fp_rel_em = [(
        rel.label,
        rel.pair[0].text,
        rel.pair[1].text,
        rel.certainty
    ) for rel, tag in pred_rel_em.items() if tag == "FP"] 
    fn_rel_em = [(
        rel.label,
        rel.pair[0].text,
        rel.pair[1].text,
        rel.certainty
    ) for rel, tag in gold_rel_em.items() if tag == "FN"]

    errors['tp-rel-relaxed'].extend(tp_rel_relaxed)
    errors['fp-rel-relaxed'].extend(fp_rel_relaxed)
    errors['fn-rel-relaxed'].extend(fn_rel_relaxed)
    errors['tp-rel-em'].extend(tp_rel_em)
    errors['fp-rel-em'].extend(fp_rel_em)
    errors['fn-rel-em'].extend(fn_rel_em)  

    return counts, ner_result_by_class, rel_result_by_class, errors
    

def evaluate_predictions(
    dataset, output_dir, task, dataset_name, do_eval_rel, use_gold, print_trigger
):

    counts = Counter()
    ner_labels = task_ner_labels[dataset_name]
    rel_labels = task_rel_labels[dataset_name]

    ner_result_by_class = {}
    for label in ner_labels:
        ner_result_by_class[label] = {"gold": 0.0, "pred": 0.0, "pred_relaxed": 0.0, 
                                      "correct": 0.0, "correct_relaxed": 0.0}
    rel_result_by_class = {}
    for label in rel_labels:
        rel_result_by_class[label] = {"gold": 0.0, "pred": 0.0, "pred_relaxed": 0.0, 
                                      "correct": 0.0, "correct_relaxed": 0.0, 
                                      "correct_fact": 0.0, "correct_relaxed_fact": 0.0}

    errors_doc = {}
    for doc in dataset:

        errors_sent = []

        for sent in doc:
            counts, ner_result_by_class, rel_result_by_class, errors = evaluate_sent(
                sent, 
                counts, 
                ner_result_by_class, 
                rel_result_by_class, 
                dataset_name, 
                do_eval_rel=do_eval_rel
            )
            errors_sent.append(errors)

        errors_doc[doc._doc_key] = errors_sent

    scores_ner = compute_f1(
        counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
    scores_ner_soft = compute_f1(
        counts["ner_relaxed_predicted"], counts["ner_gold"], counts["ner_partial_matched"])
    scores_trigger = compute_f1(
        counts["trigger_predicted"], counts["trigger_gold"], counts["trigger_matched"])
    scores_trigger_soft = compute_f1(
        counts["trigger_relaxed_predicted"], counts["trigger_gold"], counts["trigger_partial_matched"])
    scores_strict_relations = compute_f1(
        counts["relations_predicted"], counts["relations_gold"], counts["strict_relations_matched"])
    scores_relaxed_relations = compute_f1(
        counts["relations_relaxed_predicted"], counts["relations_gold"], counts["relaxed_relations_matched"])
    scores_strict_relations_fact = compute_f1(
        counts["relations_predicted"], counts["relations_gold"], counts["strict_relations_matched_fact"])
    scores_relaxed_relations_fact = compute_f1(
        counts["relations_relaxed_predicted"], counts["relations_gold"], counts["relaxed_relations_matched_fact"])

    for label in ner_result_by_class:
        counts_label = ner_result_by_class[label]
        counts_label["precision"] = safe_div(counts_label["correct"], counts_label["pred"])
        counts_label["recall"] = safe_div(counts_label["correct"], counts_label["gold"])
        counts_label["f1"] = safe_div(2*counts_label["precision"]*counts_label["recall"], \
                                      counts_label["precision"]+counts_label["recall"])       
        
        counts_label["precision_relaxed"] = safe_div(counts_label["correct_relaxed"], counts_label["pred_relaxed"])
        counts_label["recall_relaxed"] = safe_div(counts_label["correct_relaxed"], counts_label["gold"])
        counts_label["f1_relaxed"] = safe_div(2*counts_label["precision_relaxed"]*counts_label["recall_relaxed"], \
                                      counts_label["precision_relaxed"]+counts_label["recall_relaxed"])  
        
    with open(os.path.join(output_dir, f"{task}_ner_result_by_class_e2e.json"), 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(ner_result_by_class, indent=4))
    print("NER Result by class is saved!!!")

    for label in rel_result_by_class:
        counts_label = rel_result_by_class[label]
        counts_label["precision"] = safe_div(counts_label["correct"], counts_label["pred"])
        counts_label["recall"] = safe_div(counts_label["correct"], counts_label["gold"])
        counts_label["f1"] = safe_div(2*counts_label["precision"]*counts_label["recall"], \
                                      counts_label["precision"]+counts_label["recall"])       
        
        counts_label["precision_relaxed"] = safe_div(counts_label["correct_relaxed"], counts_label["pred_relaxed"])
        counts_label["recall_relaxed"] = safe_div(counts_label["correct_relaxed"], counts_label["gold"])
        counts_label["f1_relaxed"] = safe_div(2*counts_label["precision_relaxed"]*counts_label["recall_relaxed"], \
                                      counts_label["precision_relaxed"]+counts_label["recall_relaxed"])  
        
    with open(os.path.join(output_dir, f"{task}_rel_result_by_class_e2e.json"), 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(rel_result_by_class, indent=4))
    print("RE Result by class is saved!!!")

    if use_gold:
        output_file_name = f'{output_dir}/analysis_sorted_gold.txt'
    else:
        output_file_name = f'{output_dir}/analysis_sorted.txt'
    print_predictions(
        result=errors_doc,
        ner_label_result=ner_result_by_class,
        rel_label_result=rel_result_by_class,
        gold_file=dataset,
        output_file=output_file_name,
        print_trigger=print_trigger
    )

    result = dict(
        ner=scores_ner, 
        ner_soft=scores_ner_soft, 
        trigger=scores_trigger, 
        trigger_soft=scores_trigger_soft, 
        relaxed_relation=scores_relaxed_relations, 
        strict_relation=scores_strict_relations,
        relaxed_relation_fact=scores_relaxed_relations_fact, 
        strict_relation_fact=scores_strict_relations_fact
    )
    return result


def print_predictions(
        result, 
        ner_label_result, 
        rel_label_result,
        gold_file, 
        output_file,
        print_trigger
):
    
    with open(output_file, "w") as f:     
        
        ner_header = ["NER-ENTITY_TYPE", \
                  "Prec", "Rec", "F1", \
                  "Prec-relaxed", "Rec-relaxed", "F1-relaxed", \
                  "Gold", "Pred", "Correct", "Pred-relaxed", 'Correct-relaxed']
        f.write("\t".join(ner_header) + "\n")
        f.write("="*89 + "\n")
        
        # Record scores of NER/TRG by label
        for k, v in ner_label_result.items():
            record = [
                k, v['precision'], v['recall'], v['f1'], 
                v['precision_relaxed'], v['recall_relaxed'], v['f1_relaxed'], 
                int(v['gold']), int(v['pred']), int(v['correct']), 
                int(v['pred_relaxed']), int(v['correct_relaxed'])
            ]
            f.write("\t".join([str(r) for r in record]) + "\n")      
        f.write('\n')

        rel_header = ["REL_TYPE", \
                  "Prec", "Rec", "F1", \
                  "Prec-relaxed", "Rec-relaxed", "F1-relaxed", \
                  "Gold", "Pred", "Correct", "Pred-relaxed", 'Correct-relaxed']
        f.write("\t".join(rel_header) + "\n")
        f.write("="*89 + "\n")
        
        # Record scores of REL by label
        for k, v in rel_label_result.items():
            record = [
                k, v['precision'], v['recall'], v['f1'], 
                v['precision_relaxed'], v['recall_relaxed'], v['f1_relaxed'], 
                int(v['gold']), int(v['pred']), int(v['correct']), 
                int(v['pred_relaxed']), int(v['correct_relaxed'])
            ]
            f.write("\t".join([str(r) for r in record]) + "\n")      
        f.write('\n\n')
        
        for doc in gold_file:
            text = []
            abstract = []
            for idx, sent in enumerate(doc):
                text.extend(sent.text)
                if idx == 0:
                    title = ' '.join(sent.text)
                else:
                    abstract.extend(sent.text)

            f.write('|'.join([doc._doc_key, 't', title]))
            f.write('\n')
            abstract = ' '.join(abstract)
            f.write('|'.join([doc._doc_key, 'a', abstract]))
            f.write('\n\n')

            errors = result[doc._doc_key]

            assert len(doc) == len(errors)

            for idx, (sent, sent_errors) in enumerate(zip(doc, errors)):
                
                f.write('|'.join([doc._doc_key, f'S#{idx}', " ".join(sent.text)]))
                f.write("\n")
                    
                errors_ner = []
                errors_trg = []
                errors_rel = []
                for error_type, error_list in sent_errors.items():
                    # Just to print relaxed-only
                    if error_type.split('-')[-1] == 'em':
                        continue

                    if error_type.split('-')[1] == 'ner':
                        for sample in error_list:
                            errors_ner.append(
                                [error_type] + list(sample)
                            )
                    elif error_type.split('-')[1] == 'trg':
                        for sample in error_list:
                            errors_trg.append(
                                [error_type] + list(sample)
                            )                            
                    elif error_type.split('-')[1] == 'rel':
                        for sample in error_list:
                            errors_rel.append(
                                [error_type] + list(sample)
                            )

                errors_ner = sorted(errors_ner, key=lambda x: (x[1], x[0]))
                for idx, t in enumerate(errors_ner):
                    t = [
                        t[0].upper(),
                        # doc._doc_key,
                        str(t[1]), 
                        str(t[2]), 
                        ' '.join(sent.text[t[1]:t[2]+1]), t[3]
                    ]
                    f.write("\t".join(t))    
                    f.write("\n")       
                f.write("\n")

                if print_trigger:
                    errors_trg = sorted(errors_trg, key=lambda x: (x[1], x[0]))
                    for idx, t in enumerate(errors_trg):
                        t = [
                            t[0].upper(),
                            # doc._doc_key,
                            str(t[1]), 
                            str(t[2]), 
                            ' '.join(sent.text[t[1]:t[2]+1]), t[3]
                        ]
                        f.write("\t".join(t))    
                        f.write("\n")       
                    f.write("\n")

                errors_rel = sorted(errors_rel, key=lambda x: (x[1], x[0]))
                for idx, t in enumerate(errors_rel):
                    t = [
                        t[0].upper(),
                        # doc._doc_key,
                        t[1], 
                        " ".join(t[2]), 
                        " ".join(t[3]), 
                        t[4]
                    ]
                    f.write("\t".join(t))    
                    f.write("\n")       
                f.write("\n")
            f.write("\n")
                    

# def analyze_relation_coverage(dataset):
    
#     def overlap(s1, s2):
#         if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
#             return True
#         if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
#             return True
#         return False

#     nrel_gold = 0
#     nrel_pred_cover = 0
#     nrel_top_cover = 0

#     npair_pred = 0
#     npair_top = 0

#     nrel_overlap = 0

#     for d in dataset:
#         for s in d:
#             pred = set([ner.span for ner in s.predicted_ner])
#             top = set([ner.span for ner in s.top_spans])
#             npair_pred += len(s.predicted_ner) * (len(s.predicted_ner) - 1)
#             npair_top += len(s.top_spans) * (len(s.top_spans) - 1)
#             for r in s.relations:
#                 nrel_gold += 1
#                 if (r.pair[0] in pred) and (r.pair[1] in pred):
#                     nrel_pred_cover += 1
#                 if (r.pair[0] in top) and (r.pair[1] in top):
#                     nrel_top_cover += 1
                
#                 if overlap(r.pair[0], r.pair[1]):
#                     nrel_overlap += 1

#     print('Coverage by predicted entities: %.3f (%d / %d), #candidates: %d'%(nrel_pred_cover/nrel_gold*100.0, nrel_pred_cover, nrel_gold, npair_pred))
#     print('Coverage by top 0.4 spans: %.3f (%d / %d), #candidates: %d'%(nrel_top_cover/nrel_gold*100.0, nrel_top_cover, nrel_gold, npair_top))
#     print('Overlap: %.3f (%d / %d)'%(nrel_overlap / nrel_gold * 100.0, nrel_overlap, nrel_gold))
