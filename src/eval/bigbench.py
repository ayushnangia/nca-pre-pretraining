import os
import sys
import argparse
import json
import math

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
from datasets import load_dataset

from utils.util import (
    set_seed,
    setup_logger,
    load_model,
    wandb_log,
    write_jsonl,
    read_jsonl
)

from utils.models import (
    create_llama_model,
    DownstreamLlamaLM,
    create_attention_mask
)

from utils.dataset_utils import (
    BigBenchDataset,
    get_bigbench_dataset,
    pass_at_k
)

from utils.training_args import (
    BigBenchEvalArgs,
    create_bigbench_eval_parser,
    bigbench_eval_args_to_dataclass
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

### === CONSTANTS === ###
log = setup_logger('bigbench_eval')

### === DECODING FUNCTIONS === ###
def generate_response(args, model, seq):
    model.eval()
    MAX_LEN = min(args.max_len, args.seq_len - seq.shape[0])

    # Move to device and ensure correct dtype
    seq = torch.tile(seq, (args.passes, 1))
    seq = seq.to(args.device)
    # Clone to avoid any potential issues with in-place operations
    current_seq = seq.clone()

    with torch.no_grad():
        response = []
        
        for i in range(MAX_LEN):
            # Create attention mask for current sequence
            seq_len = current_seq.size(1)
            base_mask = create_attention_mask(seq_len).to(args.device)
            mask = base_mask.repeat(current_seq.size(0), 1, 1, 1).to(args.device)
            
            # Forward pass
            logits = model(current_seq, mask)
            
            # Get next token prediction (last position)
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature and sampling
            if args.temperature == 0.0:
                # Greedy decoding (argmax)
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature sampling with top-p (nucleus sampling)
                probs = F.softmax(next_token_logits / args.temperature, dim=-1)
                
                # Sort probabilities for top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Get indices to remove (top-p filtering)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                # Sample from filtered distribution
                if torch.any(torch.isnan(sorted_probs)) or torch.any(sorted_probs < 0):
                    # Fallback to greedy if probabilities are invalid
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token_sorted_idx)
            
            # Append to current sequence for next iteration
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
            # Store generated token for response
            response.append(next_token.detach().cpu())
    
    response = torch.cat(response, dim=1)
    return response

def decode_response(args, response, subset, example, idx):
    responses = []
    for i in range(response.shape[0]):
        decoded_response = args.tokenizer.decode(response[i].tolist()).split(args.eos_string)[0].strip()

        responses.append({
            "question": example["inputs"],
            "solution": decoded_response,
            "answer": example["targets"][0],
            "subset": subset,
            "idx": idx
        })
    return responses

def evaluate_pass_at_k(args, dataset,responses):
    metrics = {
        "sum": {str(k): 0 for k in args.eval_passes},
        "count": {str(k): 0 for k in args.eval_passes},
    }

    for i in range(len(dataset)):
        c = 0
        n = 0
        for response in responses:
            if response["idx"] == i:
                if response["solution"].lower() == response["answer"].lower().strip():
                    c+=1
                n+=1
        
        for k in args.eval_passes:
            metrics["sum"][str(k)] += pass_at_k(n, c, k)
            metrics["count"][str(k)] += 1

    metrics["mean"] = {str(k): metrics["sum"][str(k)] / metrics["count"][str(k)] for k in args.eval_passes}
    return metrics

### === UTILITY FUNCTIONS === ###
def init_args(args: BigBenchEvalArgs):
    assert max(args.eval_passes) <= args.passes, "Evaluation passes must be less than or equal to the number of passes"

    # set tokenizer 
    args.tokenizer = tiktoken.get_encoding("gpt2")
    args.eos_tk = args.tokenizer.eot_token
    args.eos_string = args.tokenizer.decode([args.eos_tk])
    args.solution_path = os.path.join(args.save_path, f"solutions_{args.start_idx}_{args.end_idx}.jsonl")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    return args

def build_model(args: BigBenchEvalArgs):
    pretrain_model = create_llama_model(
        vocab_size=args.vocab_size,
        seq_length=args.seq_len,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.n_embed,
    )

    model = DownstreamLlamaLM(
        model=pretrain_model,
        vocab_size=args.vocab_size,
        frozen_modules=args.freeze_modules,
        reinit_modules=args.reinit_modules+['embed'],
        weight_tying=args.weight_tying == 1,
    )
    return model

### === MAIN FUNCTIONS === ###

def main(args):
    set_seed(args.seed)
    args = init_args(args)

    # load dataset
    dataset = get_bigbench_dataset(split='validation', max_samples=args.max_per_task)
    dataset = BigBenchDataset(
        dataset=dataset,
        seq_len=args.seq_len,
        seed=args.seed,
        shot=args.n_shot,
        eval=True
    )

    # build and load model
    model = build_model(args)
    model = load_model(model, args.model_path, args.model_file, strict=True)
    model.to(args.device)
    model.eval()

    # output model info
    log.info(f"Model loaded from {args.model_path}")
    log.info(f"Model architecture: {model}")
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {params:,}")

    start_idx = 0
    end_idx = len(dataset) if args.end_idx is None else args.end_idx

    if args.resume and os.path.exists(args.solution_path):
        responses = read_jsonl(args.solution_path)
        start_idx += len(responses) // args.passes
        log.info(f"Resuming from {start_idx}")
    else:
        responses = []

    log.info(f"Evaluating from {start_idx} to {end_idx}")

    for i in range(start_idx, end_idx):
        seq, tar = dataset[i]
        log.info(f"Sequence: {seq.shape}")
        generation = generate_response(args, model, seq)
        subset, example = dataset.enumerated_examples[i]

        solution = decode_response(args, generation, subset, example, i)
        log.info(f"Solution: {generation.shape}")

        responses.extend(solution)
        write_jsonl(args.solution_path, responses)

    ### Evaluate Pass@K ###
    pass_at_k = evaluate_pass_at_k(args, dataset, responses)
    log.info(f"Pass@K: {pass_at_k}")

    fpath = os.path.join(args.save_path, "pass_at_k.json")
    with open(fpath, "w") as f:
        json.dump(pass_at_k, f)

if __name__ == "__main__":
    parser = create_bigbench_eval_parser()
    args = parser.parse_args()
    training_args = bigbench_eval_args_to_dataclass(args)
    main(training_args)
