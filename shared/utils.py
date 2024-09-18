import os
import random
import torch
import pandas as pd
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
from typing import Optional
from shared.checks import ConfigurationError

# create directory
def make_output_dir(output_dir, task, pipeline_task):

    assert task in ["ner", "triplet", "relation", "certainty"]

    # Create the main output folder for the dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Numbering the version of outputs
    existing_folders = []
    for d in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("EXP_"):
            existing_folders.append(int(d.split("_")[1]))
    num_version = 0
    if existing_folders:
        existing_folders = sorted(existing_folders, reverse=True)
        num_version = existing_folders[0] + 1

    model_output_dir = None
    entity_output_dir = None
    triplet_output_dir = None
    relation_output_dir = None
    factuality_output_dir = None

    if task == "ner":
        # Need to make a new folder to start a new version of task
        model_output_dir = os.path.join(output_dir, f"EXP_{num_version}") 
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)        
            print(f"## {model_output_dir} is created for {task.upper()} task ##")
        entity_output_dir = os.path.join(model_output_dir, "entity")
        print(f"## {entity_output_dir} is created for {task.upper()} task ##")
        return entity_output_dir
    
    elif task == "triplet":
        if pipeline_task.startswith("triplet"):  # use predicted entities in different folder
            # Setup new folder for triplet prediction
            model_output_dir = os.path.join(output_dir, f"EXP_{num_version}") 
            if not os.path.exists(model_output_dir):
                os.mkdir(model_output_dir)
                print(f"## {model_output_dir} is created for {task.upper()} task ##")
        else:  # entity path in the same folder (pipelined)
            model_output_dir = os.path.join(output_dir, str(existing_folders[0]))
            assert os.path.exists(model_output_dir)
            # entity_output_dir = os.path.join(model_output_dir, "entity")
        triplet_output_dir = os.path.join(model_output_dir, "triplet")
        print(f"## {triplet_output_dir} is created for {task.upper()} task ##")
        return triplet_output_dir
            
    elif task == "relation":
        if pipeline_task.startswith("relation"):  # Setup new folder for relation prediction
            model_output_dir = os.path.join(output_dir, f"EXP_{num_version}") 
            if not os.path.exists(model_output_dir):
                os.mkdir(model_output_dir)
                print(f"## {model_output_dir} is created for {task.upper()} task ##")
        else:
            model_output_dir = os.path.join(output_dir, str(existing_folders[0]))
            assert os.path.exists(model_output_dir)
            # # Create directories based on the order of task
            # if pipeline_task.startswith("triplet"):
            #     triplet_output_dir = os.path.join(model_output_dir, "triplet")
            # elif pipeline_task.startswith("entity"):
            #     entity_output_dir = os.path.join(model_output_dir, "entity")
            #     if "triplet" in pipeline_task:
            #         triplet_output_dir = os.path.join(model_output_dir, "triplet")
        relation_output_dir = os.path.join(model_output_dir, "relation")
        print(f"## {relation_output_dir} is created for {task.upper()} task ##")
        return relation_output_dir
    
    else:
        if pipeline_task == "certainty":  # Setup new folder for relation prediction
            model_output_dir = os.path.join(output_dir, f"EXP_{num_version}") 
            if not os.path.exists(model_output_dir):
                os.mkdir(model_output_dir)
                print(f"## {model_output_dir} is created for {task.upper()} task ##")
        else:
            model_output_dir = os.path.join(output_dir, str(existing_folders[0]))
            assert os.path.exists(model_output_dir)
            # # Create directories based on the order of task
            # if pipeline_task.startswith("triplet"):
            #     triplet_output_dir = os.path.join(model_output_dir, "triplet")
            # elif pipeline_task.startswith("entity"):
            #     entity_output_dir = os.path.join(model_output_dir, "entity")
            #     if "triplet" in pipeline_task:
            #         triplet_output_dir = os.path.join(model_output_dir, "triplet")
        certainty_output_dir = os.path.join(model_output_dir, "certainty")
        print(f"## {certainty_output_dir} is created for {task.upper()} task ##")
        return certainty_output_dir

    #     # In case of trigger or relation task, use the existing prediction result
    #     assert existing_folders, print("!!! You SHOULD have at least one entity outputs to run TRIPLET or RELATION !!!")

    #     for folder_num in existing_folders:
    #         pipeline_folder = {"curr": "", "entity": "", "triplet": ""}
    #         curr_folder = os.path.join(output_dir, f"EXP_{folder_num}")
    #         for d in os.listdir(curr_folder):
    #             if d == "entity" and not pipeline_folder["entity"]:
    #                 pipeline_folder["entity"] = d
    #             elif d == "triplet" and not pipeline_folder["triplet"]:
    #                 pipeline_folder["triplet"] = d
    #             if pipeline_folder["entity"] and pipeline_folder["triplet"]:
    #                 break
        
    #     if mode == "triplet":
    #         print(f"Entity predictions in the {pipeline_folder['entity']}")
    #         # entity_output_dir = os.path.join(curr_folder, "entity")
    #         if len(existing_tasks) == 1 and not model_output_dir:
    #             model_output_dir = curr_folder
    #             triplet_output_dir = os.path.join(model_output_dir, "triplet")
    #         else:
    #             if model_output_dir:
    #                 continue
    #             # Make a new version
    #             model_output_dir = os.path.join(output_dir, f"EXP_{num_version}")
    #             triplet_output_dir = os.path.join(model_output_dir, "triplet")
    #             if not os.path.exists(model_output_dir):
    #                 os.mkdir(model_output_dir)        
    #             print(f"## {model_output_dir} is created for {mode.upper()} task ##")
    #         if entity_output_dir and triplet_output_dir:
    #             return triplet_output_dir, entity_output_dir
                
    #         elif mode == "relation":
    #             if "relation" in existing_tasks:

    # latest_model_output_dir = os.path.join(output_dir, f"EXP_{str(max(existing_folders))}")
    # if mode == "triplet":
    #     if 'entity' in os.listdir(latest_model_output_dir):

    #         print(f'## This {mode.upper()} task would be stored in {latest_model_output_dir}')
    #         return latest_model_output_dir
    # if mode == "relation":
    #     if 'entity' in os.listdir(latest_model_output_dir) and 'triplet' in os.listdir(latest_model_output_dir):
    #         print(f'## This {mode.upper()} task would be stored in {latest_model_output_dir}')
    #         return latest_model_output_dir            

    # # Need to make a new folder to start a new version of task
    # model_output_dir = os.path.join(output_dir, f"EXP_{num_version}") 
    # if not os.path.exists(model_output_dir):
    #     os.mkdir(model_output_dir)        
    # print(f"## {model_output_dir} is created for Experiment outputs ##")
    # return model_output_dir

def generate_analysis_csv(pred_data, output_csv):
    csv_results = [] 
    for doc in pred_data.documents:
        for sid, sent in enumerate(doc.sentences):
            entry = {}
            entry['pmid'] = doc._doc_key
            entry['sent_id'] = sid
            entry['sent'] = ' '.join(sent.text)
            entry['gold_entities'] = " | ".join([f"{' '.join(ent.span.text)}: {ent.label}" for ent in sent.ner])
            entry['predicted_entities'] = " | ".join([f"{' '.join(ent.span.text)}: {ent.label}" for ent in sent.predicted_ner])
            entry['gold_triggers'] = " | ".join([f"{' '.join(trg.span.text)}: {trg.label}" for trg in sent.triggers])
            entry['predicted_triggers'] = " | ".join([f"{' '.join(trg.span.text)}: {trg.label}" for trg in sent.predicted_triggers])
            # entry['gold_relations'] = [(rel.pair[0].text, rel.pair[1].text, rel.label) for rel in doc['relations'][sid]]
            entry['gold_relations'] = " | ".join([f"{' '.join(rel.pair[0].text)}-{rel.label}-{' '.join(rel.pair[1].text)}-{rel.certainty}" for rel in sent.relations])
            entry['predicted_relations'] = " | ".join([f"{' '.join(rel.pair[0].text)}-{rel.label}-{' '.join(rel.pair[1].text)}-{rel.certainty}" for rel in sent.predicted_relations])

            csv_results.append(entry)
    
    pd.DataFrame(csv_results).to_csv(output_csv)

def set_seed(seed):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True)

def save_model(model, step, task, args):
    """
    Save the model to the output directory
    """
    if task == "ner":
        model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
        # model_to_save.save_pretrained(os.path.join(args.output_dir, new_dir))
        # model.tokenizer.save_pretrained(os.path.join(args.output_dir, new_dir))
        model_to_save.save_pretrained(args.entity_output_dir)
        model.tokenizer.save_pretrained(args.entity_output_dir)
    # elif task == "re":
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, new_dir))
    #     model_to_save.config.to_json_file(os.path.join(args.output_dir, new_dir))
    #     args.tokenizer.save_vocabulary(os.path.join(args.output_dir, new_dir))

def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    
    device = indices.get_device() if indices.is_cuda else -1
    offsets = get_range_vector(indices.size(0), device) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.
    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.
    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """

    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
        
    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets