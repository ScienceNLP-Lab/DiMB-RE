# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import time
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForACEBothOneDropoutSub,
                                  BertForACEBothOneDropoutSubFactuality,
                                  AlbertForACEBothSub,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  AlbertForACEBothOneDropoutSub,
                                  BertForACEBothOneDropoutSubNoNer
                                  )

from transformers import AutoTokenizer
# from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import timeit

from tqdm import tqdm

logger = logging.getLogger(__name__)


ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  AlbertConfig)), 
    ()
)

MODEL_CLASSES = {
    # 'bertsub': (BertConfig, BertForACEBothOneDropoutSub, BertTokenizer),
    'bertsub': (BertConfig, BertForACEBothOneDropoutSubFactuality, BertTokenizer),
    'bertnonersub': (BertConfig, BertForACEBothOneDropoutSubNoNer, BertTokenizer),
    'albertsub': (AlbertConfig, AlbertForACEBothOneDropoutSub, AlbertTokenizer),
}

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'DiMB-RE': ['Food', 'Nutrient', 'DietPattern', 'Microorganism', 'DiversityMetric', 'Metabolite', 'Physiology', 'Disease', 'Enzyme', 'Gene', 'Measurement', 'Chemical']
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'DiMB-RE': ['INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 'PREDISPOSES']
}

task_facuality_labels = {
    'DiMB-RE': ['Factual', 'Negated', 'Unknown']
}


class ACEDataset(Dataset):
    def __init__(
        self, 
        tokenizer, 
        args=None, 
        evaluate=False, 
        do_test=False, 
        max_pair_length=None, 
        use_gold=False
    ):
        if not evaluate:
            if do_test:
                file_path = os.path.join(args.data_dir, args.dev_file)
            else:
                file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                if use_gold:
                    file_path = os.path.join(args.data_dir, "test.json")
                elif args.test_file.find('models') == -1:
                    file_path = os.path.join(args.data_dir, args.test_file)
                else:
                    file_path = args.test_file
            else:
                if use_gold:
                    file_path = os.path.join(args.data_dir, "dev.json")
                elif args.dev_file.find('models') == -1 and not use_gold:
                    file_path = os.path.join(args.data_dir, args.dev_file)
                else:
                    file_path = args.dev_file

        assert os.path.isfile(file_path)

        self.file_path = file_path
                
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = max_pair_length
        self.max_entity_length = self.max_pair_length * 2

        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type
        self.no_sym = args.no_sym

        if args.data_dir.find('ace05')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
            if args.no_sym:
                label_list = ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS',  'PART-WHOLE']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('ace04')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
            if args.no_sym:
                label_list = ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('scierc')!=-1:      
            self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']
            if args.no_sym:
                label_list = ['CONJUNCTION', 'COMPARE', 'PART-OF', 'USED-FOR', 'FEATURE-OF',  'EVALUATE-FOR', 'HYPONYM-OF']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['PART-OF', 'USED-FOR', 'FEATURE-OF',  'EVALUATE-FOR', 'HYPONYM-OF']
                self.sym_labels = ['NIL', 'CONJUNCTION', 'COMPARE']
                self.label_list = self.sym_labels + label_list
                
        elif args.data_dir.find('DiMB-RE')!=-1:
            self.ner_label_list = [
                'NIL', 'Food', 'Nutrient', 'DietPattern', 'Microorganism', \
                'DiversityMetric', 'Metabolite', 'Physiology', 'Disease', \
                'Enzyme', 'Gene', 'Measurement', 'Chemical'
            ]
            self.rel_label_list = [
                'INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 
                'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 
                'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 
                'PREDISPOSES'
            ]                

            label_list = ['None', 'Factual', 'Negated', 'Unknown']
            # 'None' is for placeholder
            self.sym_labels = []
            self.label_list = self.sym_labels + label_list
    
        else:
            assert False  

        self.global_predicted_ners = {}
        self.global_predicted_triggers = {}
        self.use_trigger = args.use_trigger
        self.initialize()
        
        
    def get_sentence_index(self, entity_index, sentence_boundaries):
        """ Helper function to determine the sentence index for a given entity index. """
        for i in range(len(sentence_boundaries) - 1):
            if sentence_boundaries[i] <= entity_index < sentence_boundaries[i + 1]:
                return i
        return -1        
 
    def initialize(self):
        tokenizer = self.tokenizer
        # vocab_size = tokenizer.vocab_size
        # max_num_subwords = self.max_seq_length - 4  # for two marker
        
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}
        trg_label_map = {label: i for i, label in enumerate(self.rel_label_list)}

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        
        self.ner_tot_recall = 0
        self.trg_tot_recall = 0
        self.tot_recall = 0
        self.data = []
        self.ner_golden_labels = set([])
        self.trg_golden_labels = set([])
        self.golden_labels = set([])
        self.golden_labels_withner = set([])
        
        maxR = 0
        maxL = 0
        for l_idx, line in enumerate(f):
            data = json.loads(line)

            pmid = data['doc_key']
            sentences = data['sentences']
            
            
            
            if 'predicted_ner' in data:
               ners = data['predicted_ner']               
            else:
               ners = data['ner']
            std_ners = data['ner']
            
            if 'predicted_triggers' in data:
               triggers = data['predicted_triggers']               
            else:
               triggers = data['triggers']
            std_triggers = data['triggers']
            
            if 'predicted_triplets' in data:
               triplets = data['predicted_triplets']               
            else:
               triplets = data['triplets']
            std_triplets = data['triplets']
            
            if 'predicted_relations' in data:
               relations = data['predicted_relations']               
            else:
               relations = data['relations']
            std_relations = data['relations']
                    
            for sentence_relation in std_relations:
                for x in sentence_relation: 
                    self.tot_recall +=  1           

            sentence_boundaries = [0]
            words = []
            L = 0  # Num of tokens
            for i in range(len(sentences)):
                L += len(sentences[i])
                sentence_boundaries.append(L)
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]  # 2d array of sent-level subtokens
            subwords = [w for li in tokens for w in li]  # 1d array of subtokens
            maxL = max(maxL, len(subwords))
            
            subword2token = list(
                itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)])
            )
            token2subword = [0] + list(
                itertools.accumulate(len(li) for li in tokens)
            )  # similar with np.cumsum()
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [
                sum(len(li) for li in tokens[:p]) for p in sentence_boundaries
            ]

            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]  # Either GOLD or Predicted
                sentence_triggers = triggers[n]
                sentence_triplets = triplets[n]
                sentence_relations = relations[n]
                
                std_ner = std_ners[n]
                std_trigger = std_triggers[n]
                std_triplet = std_triplets[n]
                std_relation = std_relations[n]
                
                # span_triggers = {}
                # for trg in sentence_triggers:
                #     span_triggers[(trg[0], trg[1])] = trg  
    
                gold_factuality = {}
                for rel in std_relation:
                    gold_factuality[(rel[0], rel[1], rel[2], rel[3])] = rel[-1]

                std_entity_labels = {}
                self.ner_tot_recall += len(std_ner)

                for start, end, label in std_ner:
                    std_entity_labels[(start, end)] = label
                    # l_idx -> doc-level index
                    # n -> sent-level index
                    self.ner_golden_labels.add(
                        ((l_idx, n), (start, end), label)
                    )
                self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)
                
                std_trigger_labels = {}
                self.trg_tot_recall += len(std_trigger)

                for start, end, label in std_trigger:
                    std_trigger_labels[(start, end)] = label
                    self.trg_golden_labels.add(
                        ((l_idx, n), (start, end), label)
                    )
                self.global_predicted_triggers[(l_idx, n)] = list(sentence_triggers)                


                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]
                # Consider special tokens: CLS/SEP/Markers of SUBJ and TRG
                max_num_subwords = self.max_seq_length - (2+2+2)
                
                len_subwords = doc_sent_end - doc_sent_start
                # Check if the number of subtokens exceeds max_num_subwords
                if len_subwords > max_num_subwords:
                    print(
                        f"Skipping sentence due to exceeding subtoken limit: {len_subwords}"
                    )
                    continue

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if sentence_length < max_num_subwords:
                    if left_length < right_length:
                        left_context_length = min(left_length, half_context_length)
                        right_context_length = min(
                            right_length, 
                            max_num_subwords - left_context_length - sentence_length
                        )
                    else:
                        right_context_length = min(right_length, half_context_length)
                        left_context_length = min(
                            left_length, 
                            max_num_subwords - right_context_length - sentence_length
                        )

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[
                    doc_offset:(doc_sent_end + right_context_length)
                ]
                
                if not self.use_trigger:
                    target_tokens = [tokenizer.cls_token] \
                        + target_tokens[: self.max_seq_length - (2+2)] \
                        + [tokenizer.sep_token]
                    assert (len(target_tokens) <= self.max_seq_length - 2)
                else:
                    target_tokens = [tokenizer.cls_token] \
                        + target_tokens[: self.max_seq_length - (2+2+2)] \
                        + [tokenizer.sep_token]
                    assert (len(target_tokens) <= self.max_seq_length - 4)  
                    
                pos2label = {}
                for idx, x in enumerate(sentence_relations):
                    # Convert entity indices to subword indices
                    entity1_start_subword = token2subword[x[0]]
                    entity2_start_subword = token2subword[x[2]]
                    # Check if both entities are in the same sentence
                    sentence_index1 = self.get_sentence_index(
                        entity1_start_subword, subword_sentence_boundaries
                    )
                    sentence_index2 = self.get_sentence_index(
                        entity2_start_subword, subword_sentence_boundaries
                    )
                    if sentence_index1 == sentence_index2 and sentence_index1 == n:
                        trg_start, trg_end = sentence_triplets[idx][-3:-1]
                        
                        label = gold_factuality.get(
                            (x[0], x[1], x[2], x[3]), 'None'
                        )
                    
                        pos2label[
                            (x[0], x[1], x[2], x[3], trg_start, trg_end)
                        ] = label_map[label]
                        
                        if label != 'None':
                            self.golden_labels.add(
                                (
                                    (l_idx, n), 
                                    (x[0], x[1]), 
                                    (x[2], x[3]),
                                    (x[4]),
                                    label
                                )
                            )
                            self.golden_labels_withner.add(
                                (
                                    (l_idx, n), 
                                    (x[0], x[1], std_entity_labels[(x[0], x[1])]), 
                                    (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                    (x[4]),
                                    label
                                )
                            )
                        
                        if label != 'None' and (label in self.sym_labels[1:]):
                            self.golden_labels.add(
                                (
                                    (l_idx, n), 
                                    (x[2], x[3]), 
                                    (x[0], x[1]),
                                    (x[4]),
                                    label
                                )
                            )
                            self.golden_labels_withner.add(
                                (
                                    (l_idx, n), 
                                    (x[2], x[3], std_entity_labels[(x[2], x[3])]), 
                                    (x[0], x[1], std_entity_labels[(x[0], x[1])]), 
                                    (x[4]),
                                    label
                                )
                            )

                # entities = list(sentence_ners)

                # # Is inverse-relation applicable to triplet-based approach?
                # for x in sentence_triplets:
                #     w = (x[2], x[3], x[0], x[1], x[4], x[5])  # inverse sub-obj
                #     if w not in pos2label:
                #         if x[-1] in self.sym_labels[1:]:
                #             pos2label[w] = label_map[x[-1]]  # bug
                #         else:
                #             # Assign inverse relation
                #             pos2label[w] = label_map[x[-1]] + (len(label_map) - len(self.sym_labels))

                # # Ignore this for now
                # # Dummy entities for negative samples
                # if not self.evaluate:
                #     entities.append((10000, 10000, 'NIL')) # only for NER
                
                # print(pos2label)
                
                # 1. Add Solid markers for Subject entities
                # for sub in entities:
                for sub in sentence_ners:
                    cur_ins = []

                    if sub[0] < 10000:
                        # Initial subject positions
                        sub_s = token2subword[sub[0]] - doc_offset + 1
                        sub_e = token2subword[sub[1] + 1] - doc_offset
                        sub_label = ner_label_map[sub[2]]
                        
                        # Store original positions before adding markers
                        orig_sub_s = sub_s
                        orig_sub_e = sub_e
                        
                        # First add subject markers
                        # We use typed marker to compare with PURE settings
                        if self.use_typemarker:
                            # l_m = '[unused%d]' % (2 + sub_label)
                            # r_m = '[unused%d]' % (2 + sub_label + len(self.ner_label_list))
                            l_m = '[unused%d]' % (4 + sub_label)
                            r_m = '[unused%d]' % (4 + sub_label + len(self.ner_label_list))                            
                        else:  # Default setting from paper
                            l_m = '[unused0]'
                            r_m = '[unused1]'
                        
                        sub_tokens = target_tokens[:sub_s] \
                            + [l_m] \
                            + target_tokens[sub_s:sub_e+1] \
                            + [r_m] \
                            + target_tokens[sub_e+1:]
                            
                        # Track the actual positions of subject markers after adding them
                        sub_marker_left = sub_s  # Position of left marker
                        sub_marker_right = sub_e+2  # Position of right marker
                        sub_e += 2  # Adjust for added subject markers
                        
                        if sub_e >= self.max_seq_length-1:
                            print("SKIPPED SUBJ >> ", (pmid, n), sub, (sub_s, sub_e))
                            continue                        

                        # 2. Process each trigger separately for this subject
                        for trg in sentence_triggers:
                            cur_ins = []  # Reset for each subject-trigger pair
                            
                            # Get original trigger positions
                            trg_s = token2subword[trg[0]] - doc_offset + 1
                            trg_e = token2subword[trg[1] + 1] - doc_offset
                            trg_label = trg_label_map[trg[2]]
                            
                            # Adjust trigger positions based on subject markers
                            orig_trg_s = trg_s
                            orig_trg_e = trg_e
                            
                            if orig_trg_s >= orig_sub_s:
                                trg_s += 1
                                if orig_trg_s > orig_sub_e:
                                    trg_s += 1
                            if orig_trg_e >= orig_sub_s:   
                                trg_e += 1
                                if orig_trg_e > orig_sub_e:
                                    trg_e += 1
                            
                            # Type-specific markers for triggers
                            l_m_trg = '[unused%d]' % (100 + (2+trg_label))
                            # r_m_trg = '[unused%d]' % (50 + trg_label + len(self.label_list))
                            r_m_trg = '[unused%d]' % (100 + (2+trg_label) + len(trg_label_map))
                            
                            # Keep track of current subject marker positions
                            current_sub_s = sub_marker_left
                            current_sub_e = sub_marker_right
                            
                            tokens_with_markers = (
                                sub_tokens[:trg_s] +
                                [l_m_trg] + 
                                sub_tokens[trg_s:trg_e+1] +
                                [r_m_trg] +
                                sub_tokens[trg_e+1:]
                            )

                            # Adjust subject marker positions 
                            # considering inserted trigger mentions
                            if orig_sub_s > orig_trg_s:
                                current_sub_s += 1
                                if orig_sub_s > orig_trg_e:
                                    current_sub_s += 1
                            if orig_sub_e >= orig_trg_s:
                                current_sub_e += 1
                                if orig_sub_e > orig_trg_e:
                                    current_sub_e += 1
                                
                            trg_e += 2  # Adjust trigger end position for its own markers
                            if trg_e >= self.max_seq_length-1:
                                print("SKIPPED TRG >> ", (pmid, n), trg, (trg_s, trg_e))
                                continue 
                            
                            # Update the final subject positions for the item
                            sub_s = current_sub_s
                            sub_e = current_sub_e

                            # 3. Process all objects for this subject-trigger pair
                            for start, end, obj_label in sentence_ners:
                                if start == sub[0] and end == sub[1]:  # Skip if object is same as subject
                                    continue
                                
                                # print(std_triplet)
                                # print([sub[0], sub[1], trg[0], trg[1], start, end, trg[2]])
                                # assert 1==0
                                
                                # Filter out samples not having fact labels
                                # This is only for FD task
                                if [sub[0], sub[1], start, end, trg[0], trg[1], trg[2]] not in sentence_triplets:
                                    continue

                                # Get original object positions
                                orig_obj_s = token2subword[start] - doc_offset + 1
                                orig_obj_e = token2subword[end+1] - doc_offset
                                
                                # Start with original positions
                                left = orig_obj_s
                                right = orig_obj_e

                                if orig_obj_s >= orig_sub_s:
                                    left += 1
                                    if orig_obj_s > orig_sub_e:
                                        left += 1
                                if orig_obj_e >= orig_sub_s:   
                                    right += 1
                                    if orig_obj_e > orig_sub_e:
                                        right += 1
                                        
                                if orig_obj_s >= orig_trg_s:
                                    left += 1
                                    if orig_obj_s > orig_trg_e:
                                        left += 1
                                if orig_obj_e >= orig_trg_s:   
                                    right += 1
                                    if orig_obj_e > orig_trg_e:
                                        right += 1                                      
                                
                                if right >= self.max_seq_length-1:
                                    print("SKIPPED OBJ >> ", (pmid, n), start, end, obj_label, (left, right), (orig_obj_s, orig_obj_e))
                                    continue

                                obj = (start, end)
                                label = pos2label.get(
                                    (sub[0], sub[1], obj[0], obj[1], trg[0], trg[1]), 
                                    0
                                )

                                assert tokens_with_markers[left] == subwords[token2subword[start]], \
                                    f"Object left marker mismatch: {tokens_with_markers[left]} != {subwords[token2subword[start]]}, {subwords}, {left, orig_obj_s}"
                                assert tokens_with_markers[right] == subwords[token2subword[end+1]-1], \
                                    f"Object right marker mismatch: {tokens_with_markers[right]} != {subwords[token2subword[end+1]-1]}"
                                
                                cur_ins.append(
                                    (
                                        (left, right, ner_label_map[obj_label]),
                                        label,
                                        obj,
                                    )
                                )
                            
                            maxR = max(maxR, len(cur_ins))
                            dL = self.max_pair_length
                            
                            if self.args.shuffle:
                                np.random.shuffle(cur_ins)

                            for i in range(0, len(cur_ins), dL):
                                examples = cur_ins[i : i + dL]
                                # item is assigned per sub-trg pair
                                item = {
                                    'index': (l_idx, n),
                                    'sentence': tokens_with_markers,
                                    'examples': examples,
                                    'sub': (sub, (sub_s, sub_e), sub_label),
                                    'trg': (trg, (trg_s, trg_e), trg_label)
                                }                
                                self.data.append(item)
                            
                            # Assert subject & trigger markers
                            if self.args.debug:
                                self.log_debug_info(pmid, sub, trg, (sub_s, sub_e), (trg_s, trg_e), tokens_with_markers, cur_ins)  
                                
                            assert tokens_with_markers[sub_s].startswith('[unused'), "Subject left marker not found at start"
                            assert tokens_with_markers[sub_e].startswith('[unused'), "Subject right marker not found at end"
                            assert tokens_with_markers[trg_s].startswith('[unused'), "Trigger left marker not found at start"
                            assert tokens_with_markers[trg_e].startswith('[unused'), "Trigger right marker not found at end"                  
                                  
        logger.info('maxR: %s', maxR)
        logger.info('maxL: %s', maxL)
        
                
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # sub -> org. element (doc-level start/end, entity label)
        # sub_position -> subtoken indices:(sub_s, sub_e)
        # sub_label -> entity label
        
        sub, sub_position, sub_label = entry['sub']
        trg, trg_position, trg_label = entry['trg']
        
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))

        attention_mask = torch.zeros(
            (
              self.max_entity_length + self.max_seq_length, 
              self.max_entity_length + self.max_seq_length  
            ), dtype=torch.int64
        )
        attention_mask[:L, :L] = 1
        
        # examples are levitated markers
        if self.model_type.startswith('albert'):
            input_ids = input_ids + [30002] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [30003] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug
        else:
            input_ids = input_ids + \
                [self.tokenizer.encode('[unused2]', add_special_tokens=False)[0]] * (len(entry['examples'])) + \
                [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + \
                [self.tokenizer.encode('[unused3]', add_special_tokens=False)[0]] * (len(entry['examples'])) + \
                [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug

        labels = []
        ner_labels = []
        mention_pos = []
        mention_2 = []
        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length 
        num_pair = self.max_pair_length

        for x_idx, obj in enumerate(entry['examples']):
            
            # obj[0] -> (sent-start, sent-end, ner_label(int))
            # obj[1] -> factuality-label for given obj to sub-trg pair
            # obj[2] -> org. obj element (doc-start/end, label)
            
            m2 = obj[0]
            label = obj[1] # 0 or 1

            mention_pos.append((m2[0], m2[1]))
            mention_2.append(obj[2])

            w1 = x_idx  
            w2 = w1 + num_pair

            w1 += self.max_seq_length
            w2 += self.max_seq_length
            
            position_ids[w1] = m2[0]
            position_ids[w2] = m2[1]

            for xx in [w1, w2]:
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1

            labels.append(label)
            ner_labels.append(m2[2])

            if self.use_typemarker:
                l_m = '[unused%d]' % (4 + m2[2] + len(self.ner_label_list)*2)
                r_m = '[unused%d]' % (4 + m2[2] + len(self.ner_label_list)*3)                
                l_m = self.tokenizer.encode(l_m, add_special_tokens=False)[0]
                r_m = self.tokenizer.encode(r_m, add_special_tokens=False)[0]
                                
                # print(m2[2], l_m, r_m)                
                input_ids[w1] = l_m
                input_ids[w2] = r_m

        pair_L = len(entry['examples'])
        if self.args.att_left:
            attention_mask[
                self.max_seq_length : self.max_seq_length+pair_L, 
                self.max_seq_length : self.max_seq_length+pair_L
            ] = 1
        if self.args.att_right:
            attention_mask[
                self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L, 
                self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L    
            ] = 1

        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))
        labels += [-1] * (num_pair - len(labels))
        ner_labels += [-1] * (num_pair - len(ner_labels))
        
        # print(entry['sentence'])
        # print(input_ids)
        # assert 1==0

        item = [
            torch.tensor(input_ids),
            attention_mask,
            torch.tensor(position_ids),
            torch.tensor(sub_position),
            torch.tensor(mention_pos),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(ner_labels, dtype=torch.int64),
            torch.tensor(sub_label, dtype=torch.int64),
            torch.tensor(trg_position),
            torch.tensor(trg_label, dtype=torch.int64),
        ]

        if self.evaluate:
            item.append(entry['index'])
            item.append(sub)
            item.append(mention_2)
            item.append(trg)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]
        num_metadata_fields = 4
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
        return stacked_fields

    def log_debug_info(self, pmid, sub, trg, sub_span, trg_span, sub_tokens, objects):
        logger.info(f"=== Debugging Information for {pmid} ===")
        logger.info(f"Subject: {sub}")
        logger.info(f"Trigger: {trg}")
        logger.info(f"Marked Tokens: {sub_tokens}")
        logger.info(f"Subject Markers: {sub_tokens[sub_span[0]], sub_tokens[sub_span[1]]}")
        logger.info(f"Trigger Markers: {sub_tokens[trg_span[0]], sub_tokens[trg_span[1]]}")        

        # Log object information
        logger.info("Objects:")
        for obj in objects:
            # (left, right, ner_label_map[obj_label]),
            # label,
            # obj            
            sent_obj, rel_label, obj_inst = obj
            logger.info(f"  Object: Start={sent_obj[0]}, End={sent_obj[1]}, Obj Label={sent_obj[2]}, Rel Label={rel_label}")
            logger.info(f"  Object Tokens: {sub_tokens[sent_obj[0]:sent_obj[1]+1]}")  # Include markers

        logger.info("="*89)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(
            "logs/"\
                + args.data_dir[max(args.data_dir.rfind('/'),0):] \
                    +"_re_logs/"+args.output_dir[args.output_dir.rfind('/'):]
        )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = ACEDataset(
        tokenizer=tokenizer, 
        args=args, 
        max_pair_length=args.max_pair_length
    )
    logger.info(f"Length of TRAIN set >> {len(train_dataset)}")
    
    if not args.no_test:
        logger.info("Including DEV set in training dataset.")
        dev_dataset = ACEDataset(
            tokenizer=tokenizer, 
            args=args, 
            max_pair_length=args.max_pair_length, 
            do_test=True
        )
        train_dataset.data.extend(dev_dataset.data)  # Combine dev data into train data
        logger.info(f"Length of TRAIN + DEV set >> {len(train_dataset)}")    

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size, 
        num_workers=min(2, os.cpu_count())
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)    
            ], 
            'weight_decay': args.weight_decay    
        },
        {
            'params': [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ], 
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    if args.warmup_steps==-1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # tr_ner_loss, logging_ner_loss = 0.0, 0.0
    # tr_re_loss, logging_re_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), 
        desc="Epoch", 
        disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1

    patience_counter = 0  # Initialize patience counter
    max_patience = 5  # Set maximum patience   

    for _ in train_iterator:
        if args.shuffle and _ > 0:  # not default
            train_dataset.initialize()
            
        epoch_iterator = tqdm(
            train_dataloader, 
            desc="Iteration", 
            disable=args.local_rank not in [-1, 0]
        )
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'position_ids':   batch[2],
                'labels':         batch[5],
                'ner_labels':     batch[6],
            }

            inputs['sub_positions'] = batch[3]
            inputs['trg_positions'] = batch[8]
            
            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.model_type.endswith('bertonedropoutnersub'):
                inputs['sub_ner_labels'] = batch[7]
                inputs['trg_ner_labels'] = batch[9]

            outputs = model(**inputs)
            
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # re_loss = outputs[1]
            # ner_loss = outputs[2]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                # re_loss = re_loss / args.gradient_accumulation_steps
                # ner_loss = ner_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            # if re_loss > 0:
            #     tr_re_loss += re_loss.item()
            # if ner_loss > 0:
            #     tr_ner_loss += ner_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.model_type.endswith('rel') :
                #     ori_model.bert.encoder.layer[args.add_coref_layer].attention.self.relative_attention_bias.weight.data[0].zero_() # 可以手动乘个mask

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                    # tb_writer.add_scalar('RE_loss', (tr_re_loss - logging_re_loss)/args.logging_steps, global_step)
                    # logging_re_loss = tr_re_loss

                    # tb_writer.add_scalar('NER_loss', (tr_ner_loss - logging_ner_loss)/args.logging_steps, global_step)
                    # logging_ner_loss = tr_ner_loss


                if args.local_rank in [-1, 0] \
                    and args.save_steps > 0 \
                        and global_step % args.save_steps == 0: # valid for bert/spanbert
                            
                    update = True
                    # Save model checkpoint
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:  
                        results = evaluate(
                            args, 
                            model, 
                            tokenizer,  
                            do_test=not args.no_test,
                            use_gold=True
                        )
                        f1 = results['f1_gold_rel']
                        tb_writer.add_scalar('f1_gold_rel', f1, global_step)

                        if f1 > best_f1:
                            best_f1 = f1
                            print ('Best F1', best_f1)
                            patience_counter = 0  # Reset patience counter on improvement
                        else:
                            update = False
                            patience_counter += 1  # Increment patience counter if no improvement

                    if args.no_test and update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(
                            args.output_dir, 
                            '{}-{}'.format(checkpoint_prefix, global_step)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
        if patience_counter >= max_patience:  # Check again after the epoch
            break  # Exit the training loop if patience exceeded  
                    
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, prefix="", do_test=False, use_gold=False):

    eval_dataset = ACEDataset(
        tokenizer=tokenizer, 
        args=args, 
        evaluate=True, 
        do_test=do_test, 
        max_pair_length=args.max_pair_length,
        use_gold=use_gold
    )
    
    golden_labels = set(eval_dataset.golden_labels)
    golden_labels_withner = set(eval_dataset.golden_labels_withner)
    
    label_list = list(eval_dataset.label_list)
    sym_labels = list(eval_dataset.sym_labels)
    tot_recall = eval_dataset.tot_recall

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    scores = defaultdict(dict)
    # ner_pred = not args.model_type.endswith('noner')
    example_subs = set([])
    example_sub_trg = set([])
    num_label = len(label_list)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=args.eval_batch_size, 
        collate_fn=ACEDataset.collate_fn, 
        num_workers=min(4, os.cpu_count())
    )

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))

    start_time = timeit.default_timer() 

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-4]  # (l_idx, n) * bsz
        subs = batch[-3]  # (sub, (sub_s, sub_e), sub_label)
        batch_m2s = batch[-2]  # list of object elements
        trgs = batch[-1]  # (trg, (trg_s, trg_e), trg_label)
        labels = batch[5]
        ner_labels = batch[6]

        batch = tuple(t.to(args.device) for t in batch[:-4])

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'position_ids':   batch[2],
                #   'labels':         batch[4],
                #   'ner_labels':     batch[5],
            }
            

            inputs['sub_positions'] = batch[3]
            inputs['trg_positions'] = batch[8]
            
            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]

            outputs = model(**inputs)

            logits = outputs[0]
            if args.eval_logsoftmax:  # perform a bit better
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
            elif args.eval_softmax:
                logits = torch.nn.functional.softmax(logits, dim=-1)

            if args.use_ner_results or args.model_type.endswith('nonersub'):                 
                ner_preds = ner_labels
            else:
                ner_preds = torch.argmax(outputs[1], dim=-1)
                
            logits = logits.cpu().numpy()
            ner_preds = ner_preds.cpu().numpy()
            labels = labels.cpu().numpy()
            
            for i in range(len(indexs)):
                index = indexs[i]
                sub = subs[i]
                m2s = batch_m2s[i]
                trg = trgs[i]
                
                example_subs.add(
                    ((index[0], index[1]), (sub[0], sub[1]))
                )
                example_sub_trg.add(
                    (
                        (index[0], index[1]), 
                        (sub[0], sub[1]),
                        (trg[0], trg[1])
                    )
                )                
                for j in range(len(m2s)):
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i, j]]
                    scores[(index[0], index[1])][((sub[0], sub[1]), (obj[0], obj[1]), (trg[0], trg[1]))] = (
                        logits[i, j].tolist(), 
                        labels[i, j],
                        ner_label, 
                        sub[-1], 
                        trg[-1]
                    )
            
    cor = 0 
    tot_pred = 0
    cor_with_ner = 0
    global_predicted_ners = eval_dataset.global_predicted_ners
    ner_golden_labels = eval_dataset.ner_golden_labels
    ner_cor = 0 
    ner_tot_pred = 0
    ner_ori_cor = 0
    tot_output_results = defaultdict(list)
    predict_relations = defaultdict(list)
    
    if not args.eval_unidirect:     # eval_unidrect is for ablation study
        # print (len(scores))
        for example_index, pair_dict in sorted(scores.items(), key=lambda x: x[0]):  
            # example_index: (l_idx, n)
            # pair_dict: dict of {s-o-t: (scores, obj_label, trg_label)} from (l_idx, n)
            
            visited = set([])
            sentence_results = []
            for k1, (v1, gold, v2_ner_label, sub_label, trg_label) in pair_dict.items():
                
                # k1 = tuple of s-o-t indices
                # v1 = logits
                # v2_ner_label = object label
                
                if k1 in visited:
                    continue
                visited.add(k1)

                if v2_ner_label == 'NIL':
                    continue
                
                v1 = list(v1)
                m1 = k1[0]  # sub indices
                m2 = k1[1]  # obj indices
                m3 = k1[2]  # trg indices
                
                if m1 == m2:
                    continue

                v1_ner_label = sub_label
                if v1_ner_label == 'NIL':
                    continue

                pred_label = np.argmax(v1)
                if pred_label > 0:
                    # if pred_label >= num_label:
                    #     pred_label = pred_label - num_label + len(sym_labels)
                    #     m1, m2 = m2, m1
                    #     v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label

                    pred_score = v1[pred_label]
                    sentence_results.append(
                        (
                            pred_score, 
                            m1, 
                            m2,
                            pred_label, 
                            v1_ner_label, 
                            v2_ner_label,
                            m3,
                            trg_label,
                            gold
                        ) 
                    )

            sentence_results.sort(key=lambda x: -x[0])
            
            no_overlap = []
            def is_overlap(m1, m2):
                if m2[0]<=m1[0] and m1[0]<=m2[1]:
                    return True
                if m1[0]<=m2[0] and m2[0]<=m1[1]:
                    return True
                return False

            output_preds = []

            for item in sentence_results:
                # print(item)
                m1 = item[1]
                m2 = item[2]
                m3 = item[-3]
                
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    # Same relation type 
                    # & overlap subject & overlap object --> delete
                    if item[-2] == x[-2] \
                        and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break

                pred_label = label_list[item[3]]

                if not overlap:
                    no_overlap.append(item)

            pos2ner = {}
            
            for item in no_overlap:
                
                m1 = item[1]
                m2 = item[2]
                m3 = item[-3]
                
                pred_label = label_list[item[3]]
                rel_label = item[-2]
                
                if item[-1]:
                    tot_pred += 1
                
                # if pred_label in sym_labels:
                #     tot_pred += 1 # duplicate
                #     if (example_index, m1, m2, rel_label, pred_label) in golden_labels \
                #         or (example_index, m2, m1, rel_label, pred_label) in golden_labels:
                #         cor += 2
                # else:
                if (example_index, m1, m2, rel_label, pred_label) in golden_labels:
                    cor += 1        

                if m1 not in pos2ner:
                    pos2ner[m1] = item[4]
                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                output_preds.append((m1, m2, rel_label, pred_label))
                
                if pred_label in sym_labels:
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner  \
                            or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in golden_labels_withner:
                        cor_with_ner += 2
                else:  
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                        cor_with_ner += 1      

            if do_test:
                tot_output_results[example_index[0]].append(
                    (example_index[1], output_preds)
                )
            predict_relations[example_index] = [
                (p[0][0], p[0][1], p[1][0], p[1][1], p[2], p[3]) for p in output_preds
            ]          

            # Refine NER results
            ner_results = list(global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i] 
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1
        
    else:
        for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
            sentence_results = []
            for k1, (v1, v2_ner_label) in pair_dict.items():
                if v2_ner_label == 'NIL':
                    continue
                
                v1 = list(v1)
                m1 = k1[0]
                m2 = k1[1]
                if m1 == m2:
                    continue
              
                pred_label = np.argmax(v1)
                if pred_label > 0 and pred_label < num_label:

                    pred_score = v1[pred_label]

                    sentence_results.append( (pred_score, m1, m2, pred_label, None, v2_ner_label) )

            sentence_results.sort(key=lambda x: -x[0])
            no_overlap = []
            def is_overlap(m1, m2):
                if m2[0]<=m1[0] and m1[0]<=m2[1]:
                    return True
                if m1[0]<=m2[0] and m2[0]<=m1[1]:
                    return True
                return False

            output_preds = []

            for item in sentence_results:
                m1 = item[1]
                m2 = item[2]
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    if item[3] == x[3] \
                        and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break

                pred_label = label_list[item[3]]

                output_preds.append((m1, m2, pred_label))

                if not overlap:
                    no_overlap.append(item)

            pos2ner = {}
            predpos2ner = {}
            ner_results = list(global_predicted_ners[example_index])
            for start, end, label in ner_results:
                predpos2ner[(start, end)] = label

            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                pred_label = label_list[item[3]]
                tot_pred += 1

                if (example_index, m1, m2, pred_label) in golden_labels:
                    cor += 1        

                if m1 not in pos2ner:
                    pos2ner[m1] = predpos2ner[m1]#item[4]

                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                # if pred_label in sym_labels:
                #     if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner \
                #         or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in golden_labels_withner:
                #         cor_with_ner += 2
                # else:  
                if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                    cor_with_ner += 1      
            
            # refine NER results
            ner_results = list(global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i] 
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1


    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f example per second)", 
        evalTime,  
        len(global_predicted_ners) / evalTime
    )
    
    f = open(eval_dataset.file_path)
    
    if not use_gold:
        if do_test:
            if 'goldner' in eval_dataset.file_path:
                output_w = open(os.path.join(args.output_dir, 'pred_results_goldner.json'), 'w')
            else:
                output_w = open(os.path.join(args.output_dir, 'pred_results.json'), 'w')
            json.dump(tot_output_results, output_w)
        
        # f = open(eval_dataset.file_path)
        if do_test:
            if 'goldner' in eval_dataset.file_path:
                output_w = open(os.path.join(args.output_dir, 'rel_pred_test_goldner.json'), 'w')  
            else:
                output_w = open(os.path.join(args.output_dir, 'rel_pred_test.json'), 'w')  
        else:
            output_w = open(os.path.join(args.output_dir, 'rel_pred_dev.json'), 'w')  
    else:
        if do_test:
            output_w = open(os.path.join(args.output_dir, 'pred_results_goldrel.json'), 'w')
            json.dump(tot_output_results, output_w)
        
        # f = open(eval_dataset.file_path)
        if do_test:
            output_w = open(os.path.join(args.output_dir, 'rel_pred_test_goldrel.json'), 'w')  
        else:
            output_w = open(os.path.join(args.output_dir, 'rel_pred_dev_goldrel.json'), 'w')        
        
    for l_idx, line in enumerate(f):
        data = json.loads(line)
        num_sents = len(data['sentences'])
        predicted_relations = []
        for n in range(num_sents):
            item = predict_relations.get((l_idx, n), [])
            item.sort()
            predicted_relations.append(item)

        data['predicted_relations'] = predicted_relations
        output_w.write(json.dumps(data)+'\n')

    ner_p = ner_cor / ner_tot_pred if ner_tot_pred > 0 else 0 
    ner_r = ner_cor / len(ner_golden_labels) 
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0

    p = cor / tot_pred if tot_pred > 0 else 0 
    r = cor / tot_recall 
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    # assert(tot_recall==len(golden_labels)) # Due to cross-sentence cases

    p_with_ner = cor_with_ner / tot_pred if tot_pred > 0 else 0 
    r_with_ner = cor_with_ner / tot_recall
    # assert(tot_recall==len(golden_labels_withner)) # Due to cross-sentence cases
    f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if cor_with_ner > 0 else 0.0

    if not use_gold:
        results = {'f1': f1, 'prec': p, 'rec': r, 'f1_with_ner': f1_with_ner, 'ner_f1': ner_f1}
    else:
        results = {'f1_gold_rel': f1, 'prec_gold_rel': p, 'rec_gold_rel': r, 'f1_with_gold_ner': f1_with_ner, 'gold_ner_f1': ner_f1}
        
    logger.info("Result: %s", json.dumps(results))
    
    return results



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='../data/DiMB-RE/ner_reduced_v6.1_trg_abs_result', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file",  default="train.json", type=str)
    parser.add_argument("--dev_file",  default="dev.json", type=str)
    parser.add_argument("--test_file",  default="test.json", type=str)
    parser.add_argument('--max_pair_length', type=int, default=64,  help="")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--eval_logsoftmax', action='store_true')
    parser.add_argument('--eval_softmax', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--no_sym', action='store_true')
    parser.add_argument('--att_left', action='store_true')
    parser.add_argument('--att_right', action='store_true')
    parser.add_argument('--use_ner_results', action='store_true')
    parser.add_argument('--use_typemarker', action='store_true')
    parser.add_argument('--eval_unidirect', action='store_true')
    
    parser.add_argument('--use_trigger', action='store_true')
    # parser.add_argument('--binary_cls', action='store_true')
    parser.add_argument('--debug', action='store_true', help="Enable detailed debugging output")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) \
        and os.listdir(args.output_dir) \
            and args.do_train \
                and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test')==-1:
        create_exp_dir(args.output_dir, scripts_to_save=['run_re.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_albert.py'])

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.data_dir.find('ace')!=-1:
        num_ner_labels = 8
        if args.no_sym:
            num_labels = 7 + 7 - 1
        else:
            num_labels = 7 + 7 - 2            
    elif args.data_dir.find('scierc')!=-1:
        num_ner_labels = 7
        if args.no_sym:
            num_labels = 8 + 8 - 1
        else:
            num_labels = 8 + 8 - 3            
    elif args.data_dir.find('DiMB-RE')!=-1:
        num_ner_labels = 13 # (12 + NIL)
        num_rel_labels = 13
        num_labels = 4 # (3 + None)
    else:
        assert False

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # args.model_name_or_path = "/jet/home/ghong1/ocean_cis230030p/ghong1/BiomedBERT-fulltext"
    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, 
        num_labels=num_labels
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = num_ner_labels

    model = model_class.from_pretrained(
        args.model_name_or_path, 
        from_tf=bool('.ckpt' in args.model_name_or_path), 
        config=config
    )

    # if args.model_type.startswith('albert'):
    if args.use_typemarker:
        special_tokens_dict = {
            'additional_special_tokens': [
                '[unused' + str(x) + ']' for x in range(num_ner_labels*4+4)
            ] + \
            [
                '[unused' + str(100+x) + ']' for x in range(num_rel_labels*2+2)
            ]
        }
    else:
        special_tokens_dict = {
            'additional_special_tokens': [
                '[unused' + str(x) + ']' for x in range(4)
            ] + \
                [
                '[unused' + str(100+x) + ']' for x in range(2)
            ]
        }
    tokenizer.add_special_tokens(special_tokens_dict)
    logger.info('Added tokens: %s', tokenizer.additional_special_tokens)
    logger.info('Added ids: %s', tokenizer.additional_special_tokens_ids)
    logger.info('Update vocab len: %s', len(tokenizer))
    # model.albert.resize_token_embeddings(len(tokenizer))
    model.bert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        subject_id = tokenizer.encode('subject', add_special_tokens=False)
        assert(len(subject_id)==1)
        subject_id = subject_id[0]
        object_id = tokenizer.encode('object', add_special_tokens=False)
        assert(len(object_id)==1)
        object_id = object_id[0]
        
        trigger_id = tokenizer.encode('trigger', add_special_tokens=False)
        assert(len(trigger_id)==1)
        trigger_id = trigger_id[0]

        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
        assert(len(mask_id)==1)
        mask_id = mask_id[0]

        logger.info(
            " subject_id = %s, object_id = %s, trigger_id = %s, mask_id = %s", 
            subject_id, object_id, trigger_id, mask_id
        )

        if args.lminit: 
            if args.model_type.startswith('albert'):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
                trgs = 30004
                trge = 30005
            # else:
            #     word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            #     subs = 1
            #     sube = 2
            #     objs = 3
            #     obje = 4
            #     trgs = 5
            #     trge = 6
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                # Only for untyped markers???
                subs = tokenizer.encode('[unused0]', add_special_tokens=False)[0]
                sube = tokenizer.encode('[unused1]', add_special_tokens=False)[0]
                objs = tokenizer.encode('[unused2]', add_special_tokens=False)[0]
                obje = tokenizer.encode('[unused3]', add_special_tokens=False)[0]
                trgs = tokenizer.encode('[unused100]', add_special_tokens=False)[0]
                trge = tokenizer.encode('[unused101]', add_special_tokens=False)[0]

            word_embeddings[subs].copy_(word_embeddings[mask_id])     
            word_embeddings[sube].copy_(word_embeddings[subject_id])   
            word_embeddings[objs].copy_(word_embeddings[mask_id])      
            word_embeddings[obje].copy_(word_embeddings[object_id])
            word_embeddings[trgs].copy_(word_embeddings[mask_id])
            word_embeddings[trge].copy_(word_embeddings[trigger_id])

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and \
        (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            
        # Evaluate after training is done
        # Save the final model if needed
        update = True
        if args.no_test and args.evaluate_during_training:
            results = evaluate(
                args, 
                model, 
                tokenizer,  
                do_test=not args.no_test,
                use_gold=True
            )
            f1 = results['f1_gold_rel']
            if f1 > best_f1:
                best_f1 = f1
                print('Best F1 >>', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            
            tokenizer.save_pretrained(output_dir)
            torch.save(
                args, os.path.join(output_dir, 'training_args.bin')
            )
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        # tokenizer.save_pretrained(args.output_dir)
        # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {'dev_best_f1': best_f1}
    if (args.do_eval or not args.no_test) and args.local_rank in [-1, 0]:
        # Initialize checkpoints to include only inner checkpoint folders
        checkpoints = sorted(
            glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))
        )

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            # checkpoints = list(
            #     os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True))
            # )
            
            # Filter to include only directories that contain the weights
            checkpoints = [
                c for c in checkpoints if os.path.exists(
                    os.path.join(c, WEIGHTS_NAME)
                )
            ]        

        logger.info("Evaluate on test (Dev or Test) set")
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 0 else ""

            model = model_class.from_pretrained(
                checkpoint,
                config=config
            )       
            model.to(args.device)

            if 'goldner' not in args.test_file:
                # 1. Final eval with GOLD REL
                logger.info("EVAL with GOLD Relations")
                result_goldrel = evaluate(
                    args, 
                    model, 
                    tokenizer, 
                    prefix=global_step, 
                    do_test=not args.no_test,
                    use_gold=True
                )
                result_goldrel = dict(
                    (k + '_{}'.format(global_step), v) for k, v in result_goldrel.items()
                )
                results.update(result_goldrel) 
            
            # 2. Final eval with Predicted REL   
            logger.info("EVAL with Predicted Relations")             
            result = evaluate(
                args, 
                model, 
                tokenizer, 
                prefix=global_step, 
                do_test=not args.no_test
            )
            result = dict(
                (k + '_{}'.format(global_step), v) for k, v in result.items()
            )
            results.update(result)
                        
        logger.info(results)

        if 'goldner' not in args.test_file:
            output_eval_file = os.path.join(args.output_dir, "results.json")
            json.dump(results, open(output_eval_file, "w"))


if __name__ == "__main__":
    main()

