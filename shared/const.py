task_ner_labels = {
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'bc5cdr': ['Disease', 'Chemical'],
    'bb2019': ['Habitat', 'Microorganism', 'Geographical', 'Phenotype'],
    'pn': ['Food', 'Nutrient', 'DietPattern', 'Microorganism', 'DiversityMetric', 'Metabolite', 'Physiology', 'Disease', 'Methodology', 'Population', 'Biospecimen', 'Enzyme', 'Gene', 'Measurement', 'Chemical'],
    'pn_reduced': ['Food', 'Nutrient', 'DietPattern', 'Microorganism', 'DiversityMetric', 'Metabolite', 'Physiology', 'Disease', 'Enzyme', 'Gene', 'Measurement', 'Chemical'],
    'pn_reduced_trg': ['Food', 'Nutrient', 'DietPattern', 'Microorganism', 'DiversityMetric', 'Metabolite', 'Physiology', 'Disease', 'Enzyme', 'Gene', 'Measurement', 'Chemical',\
                       'INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 'PREDISPOSES'],
    'pn_reduced_trg_dummy': ['Food', 'Nutrient', 'DietPattern', 'Microorganism', 'DiversityMetric', 'Metabolite', 'Physiology', 'Disease', 'Enzyme', 'Gene', 'Measurement', 'Chemical', 'TRIGGER'],
    'biocreative': ['GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature', 'SequenceVariant', 'ChemicalEntity', 'OrganismTaxon', 'CellLine']
}

task_rel_labels = {
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'bc5cdr': ['CID'],
    'pn': ['INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 'PREDISPOSES'],
    'pn_reduced': ['INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 'PREDISPOSES'],
    'pn_reduced_trg': ['INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 'PREDISPOSES'],
    'pn_reduced_trg_dummy': ['INCREASES', 'DECREASES', 'HAS_COMPONENT', 'POS_ASSOCIATED_WITH', 'AFFECTS', 'PREVENTS', 'IMPROVES', 'ASSOCIATED_WITH', 'NEG_ASSOCIATED_WITH', 'CAUSES', 'WORSENS', 'INTERACTS_WITH', 'PREDISPOSES'],
    'biocreative': []
}

task_certainty_labels = {
    'pn_reduced_trg': ['Factual', 'Negated', 'Unknown'],
    'pn_reduced_trg_dummy': ['Factual', 'Negated', 'Unknown'],
    'biocreative': []
}

def get_labelmap(label_list):
    entities = [label for label in label_list if not label.isupper()]
    triggers = [label for label in label_list if label.isupper()]

    entities.sort()
    triggers.sort()
    label_list = entities + triggers

    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label, len(entities), len(triggers)

def get_ner_labelmap(label_list):
    label2id_entity = {}
    id2label_entity = {}
    label2id_trigger = {}
    id2label_trigger = {}
    for i, label in enumerate(label_list):
        if label.isupper():
            label2id_trigger[label] = i + 1
            id2label_trigger[i + 1] = label
        else:
            label2id_entity[label] = i + 1
            id2label_entity[i + 1] = label
    return label2id_entity, id2label_entity, label2id_trigger, id2label_trigger
