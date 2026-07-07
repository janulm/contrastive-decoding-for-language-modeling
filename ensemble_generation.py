import torch
from torch.nn import functional as F
from transformers import LlamaForCausalLM
from typing import Optional





def contrastive_generate(
    good_model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    bad_model: Optional[LlamaForCausalLM] = None,
    max_new_tokens: int = 20,
    num_return_sequences: int = 1,
    logits_combiner = None,
    do_sample: bool = True,
    **generation_kwargs
) -> torch.LongTensor:
    """
    Efficiently generate tokens using both a good and optionally a bad model.
    
    Args:
        good_model: The primary model for generation
        input_ids: Input token ids
        attention_mask: Optional attention mask
        bad_model: Optional secondary model for contrastive generation
        max_new_tokens: Maximum number of new tokens to generate
        num_return_sequences: Number of output sequences to generate per input sequence
        logits_combiner: Function that takes (input_ids, good_logits, bad_logits) and returns combined logits
        do_sample: Whether to sample from the distribution (True) or use greedy decoding (False)
        **generation_kwargs: Additional arguments for generation
        
    Returns:
        Generated token ids with shape [batch_size * num_return_sequences, seq_length]
    """
    import torch
    import torch.nn.functional as F
    
    # Get original batch size
    batch_size = input_ids.shape[0]
    
    # Expand input_ids and attention_mask if num_return_sequences > 1
    if num_return_sequences > 1:
        # Repeat each sequence num_return_sequences times
        # [B, L] -> [B * num_return_sequences, L]
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

    # Get pad token for EOS handling
    #pad_token_id = generation_kwargs.get(
    #    "pad_token_id", 
    #    good_model.config.pad_token_id if hasattr(good_model.config, "pad_token_id") else 0
    #)
    
    # Prepare model kwargs for both models
    good_model_kwargs = {
        "attention_mask": attention_mask,
        "use_cache": True,
        "past_key_values": None
    }
    
    bad_model_kwargs = {
        "attention_mask": attention_mask,
        "use_cache": True,
        "past_key_values": None
    } if bad_model is not None else None
    
    # Create unfinished_sequences tensor for expanded batch
    expanded_batch_size = batch_size * num_return_sequences
    # unfinished_sequences = input_ids.new_ones(expanded_batch_size).bool()    
    # Start generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass for good model
            if good_model_kwargs["past_key_values"] is None:
                # First iteration: process the full input
                good_outputs = good_model(
                    input_ids=input_ids, 
                    **{k: v for k, v in good_model_kwargs.items() if k != "past_key_values"}
                )
            else:
                # Subsequent iterations: only process the last token with cached KV
                good_outputs = good_model(
                    input_ids=input_ids[:, -1].unsqueeze(-1),
                    **good_model_kwargs
                )
            
            # Update KV cache for good model
            good_model_kwargs["past_key_values"] = good_outputs.past_key_values
            
            # Get good model logits for the last token
            good_logits = good_outputs.logits[:, -1, :].float()
            
            # Process bad model if provided
            bad_logits = None
            if bad_model is not None:
                # Forward pass for bad model
                if bad_model_kwargs["past_key_values"] is None:
                    # First iteration: process the full input
                    bad_outputs = bad_model(
                        input_ids=input_ids,
                        **{k: v for k, v in bad_model_kwargs.items() if k != "past_key_values"}
                    )
                else:
                    # Subsequent iterations: only process the last token with cached KV
                    bad_outputs = bad_model(
                        input_ids=input_ids[:, -1].unsqueeze(-1),
                        **bad_model_kwargs
                    )
                
                # Update KV cache for bad model
                bad_model_kwargs["past_key_values"] = bad_outputs.past_key_values
                
                # Get bad model logits for the last token
                bad_logits = bad_outputs.logits[:, -1, :].float()
            
            # Logits combiners/ processors. 
            if bad_model is not None and logits_combiner is not None:
                next_token_logits = logits_combiner(input_ids, good_logits, bad_logits)
            elif logits_combiner is not None:
                next_token_logits = logits_combiner(input_ids, good_logits)
            else:
                next_token_logits = good_logits
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Apply EOS mask for finished sequences
            #next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~unfinished_sequences)  
            
            # Append tokens to input
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention_mask for both models if needed
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
                good_model_kwargs["attention_mask"] = attention_mask
                if bad_model is not None:
                    bad_model_kwargs["attention_mask"] = attention_mask
    
            # Update unfinished sequences
            #eos_token_id = generation_kwargs.get("eos_token_id", good_model.config.eos_token_id)
            #if eos_token_id is not None:
            #    unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)
            
            # Stop if all sequences are finished
            #if not unfinished_sequences.any():
            #    print("All sequences are finished, early stopping")
            #    break

    # Return the generated sequences with shape [batch_size * num_return_sequences, seq_length]
    return input_ids