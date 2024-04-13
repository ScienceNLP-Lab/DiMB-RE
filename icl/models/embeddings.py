import torch    


def get_embeddings(target, query, tokenizer, model, device, batch_size):

    idx = 0
    all_target_embeddings = []
    all_query_embeddings = []

    # Get the embeddings
    with torch.no_grad():
        samples = target
        while samples:
            to_take = min(batch_size, len(samples))
            batch = samples[idx:(idx+to_take)]
            samples = samples[(idx+to_take):]
            # for b in batch:
            #     b_input = tokenizer(b, padding=True, truncation=True, return_tensors="pt")
            #     target_embeddings = model(**b_input.to(device), output_hidden_states=True, return_dict=True).pooler_output
            #     all_target_embeddings.extend(target_embeddings.detach().cpu())
            b_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            target_embeddings = model(**b_input.to(device), output_hidden_states=True, return_dict=True).pooler_output
            all_target_embeddings.extend(target_embeddings.detach().cpu())

        samples = query
        while samples:
            to_take = min(batch_size, len(samples))
            batch = samples[idx:(idx+to_take)]
            samples = samples[(idx+to_take):]
            # for b in batch:
            #     b_input = tokenizer(b, padding=True, truncation=True, return_tensors="pt")
            #     query_embeddings = model(**b_input.to(device), output_hidden_states=True, return_dict=True).pooler_output
            #     all_query_embeddings.extend(query_embeddings.detach().cpu())
            # for b in batch:
            b_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            query_embeddings = model(**b_input.to(device), output_hidden_states=True, return_dict=True).pooler_output
            all_query_embeddings.extend(query_embeddings.detach().cpu())

    return all_query_embeddings, all_target_embeddings

def get_embeddings_st(target, query, model, device, batch_size):

    idx = 0
    all_target_embeddings = []
    all_query_embeddings = []

    # Get the embeddings
    samples = target
    while samples:
        to_take = min(batch_size, len(samples))
        batch = samples[idx:(idx+to_take)]
        samples = samples[(idx+to_take):]
        target_embeddings = model.encode(batch)
        all_target_embeddings.extend(target_embeddings)

    samples = query
    while samples:
        to_take = min(batch_size, len(samples))
        batch = samples[idx:(idx+to_take)]
        samples = samples[(idx+to_take):]
        query_embeddings = model.encode(batch)
        all_query_embeddings.extend(query_embeddings)

    return all_query_embeddings, all_target_embeddings
