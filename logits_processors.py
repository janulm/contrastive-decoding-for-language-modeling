import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor
import time

class TopKLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k: int, temperature: float = 1.0, penalty: float = -1000.0):
        assert isinstance(top_k, int) and top_k > 0, f"top_k must be a positive integer, got {top_k}"
        assert isinstance(temperature, float) and temperature > 0, f"temperature must be a positive float, got {temperature}"
        self.top_k, self.temperature, self.penalty = top_k, temperature, penalty

    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert isinstance(scores, torch.FloatTensor) or isinstance(scores, torch.Tensor), f"scores must be a Tensor, got {type(scores)}"
        assert self.top_k <= scores.size(-1), f"top_k must be less than or equal to the vocabulary size, got {self.top_k} and vocab size {scores.size(-1)}"
        
        # Apply temperature first
        scores = scores / self.temperature
        
        # Use the vectorized implementation by default
        return self._vectorized_implementation(scores)
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        # temperature
        scores = scores / self.temperature

        # ensure last-dim check works for both 2D/3D
        assert self.top_k <= scores.size(-1), (
            f"top_k must be <= vocab size, got {self.top_k} and vocab size {scores.size(-1)}"
        )

        # accept both [B,V] and [B,T,V]
        squeeze_back = False
        if scores.dim() == 2:
            scores = scores.unsqueeze(1)  # [B,1,V]
            squeeze_back = True
        elif scores.dim() != 3:
            raise ValueError(f"TopK expects scores dim 2 or 3, got shape {tuple(scores.shape)}")

        out = self._vectorized_implementation(scores)  # expects [B,T,V]

        if squeeze_back:
            out = out[:, -1, :]  # [B,V]
        return out

    def _vectorized_implementation(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Vectorized implementation of top-k filtering
        """
        if not hasattr(self, '_compiled_vectorized_implementation'):
            self._compiled_vectorized_implementation = torch.compile(
                self._vectorized_implementation_impl,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
                dynamic=True
            )
        return self._compiled_vectorized_implementation(scores)
    
    def _vectorized_implementation_impl(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Implementation of vectorized top-k filtering that gets compiled
        """
        batch_size, seq_len, vocab_size = scores.shape
        
        # Create a copy of scores for output
        filtered_scores = scores.clone()
        
        # Find values and indices of top-k elements along the vocab dimension
        top_k_values, _ = torch.topk(scores, k=min(self.top_k, vocab_size), dim=-1)
        
        # Get the k-th largest value for each batch and sequence position
        # This will be our threshold for filtering
        thresholds = top_k_values[:, :, -1].unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        
        # Create mask for values below threshold
        mask = scores < thresholds
        
        # Apply penalty to values below threshold
        filtered_scores = torch.where(mask, torch.full_like(scores, self.penalty), filtered_scores)
        
        return filtered_scores
    


class TopPLogitsProcessor(LogitsProcessor):
    def __init__(self, top_p: float, temperature: float = 1.0, penalty: float = -1000.0):
        assert isinstance(top_p, float) and 0.0 < top_p <= 1.0, f"top_p must be a float between 0 and 1, got {top_p}"
        assert isinstance(temperature, float) and temperature > 0, f"temperature must be a positive float, got {temperature}"
        self.top_p, self.temperature, self.penalty = top_p, temperature, penalty
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
            
        # temperature
        scores = scores / self.temperature

        # accept both [B,V] and [B,T,V]
        squeeze_back = False
        if scores.dim() == 2:
            scores = scores.unsqueeze(1)  # [B,1,V]
            squeeze_back = True
        elif scores.dim() != 3:
            raise ValueError(f"TopP expects scores dim 2 or 3, got shape {tuple(scores.shape)}")

        out = self._better_vectorized_implementation(scores)  # expects [B,T,V]

        if squeeze_back:
            out = out[:, -1, :]  # [B,V]
        return out
    
    
    def _better_vectorized_implementation(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        More optimized vectorized implementation using batch operations where possible
        """
        if not hasattr(self, '_compiled_better_vectorized_implementation'):
            self._compiled_better_vectorized_implementation = torch.compile(
                self._better_vectorized_implementation_impl,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
                dynamic=True
            )
        return self._compiled_better_vectorized_implementation(scores)
    
    def _better_vectorized_implementation_impl(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Implementation of better vectorized nucleus sampling that gets compiled
        """
        batch_size, seq_len, vocab_size = scores.shape
        new_scores = scores.clone()
        
        # Process each batch (but vectorize over sequence dimension where possible)
        for b in range(batch_size):
            # Convert to probabilities for this batch (all sequence positions)
            probs = F.softmax(scores[b], dim=-1)  # shape: [seq_len, vocab_size]
            
            # For each sequence position, we need to sort and get cumulative sum
            # Create tensors to store results
            keep_masks = torch.zeros(seq_len, vocab_size, dtype=torch.bool, device=scores.device)
            
            # This part still requires iteration over sequence positions due to sorting
            for s in range(seq_len):
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs[s], descending=True)
                
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create mask for tokens to keep (cumulative prob <= p)
                mask = cumulative_probs <= self.top_p
                
                # Always include at least one token
                if not mask.any():
                    mask[0] = True
                else:
                    # Include the first token that exceeds p as well
                    idx = torch.where(cumulative_probs > self.top_p)[0]
                    if len(idx) > 0:
                        mask[idx[0]] = True
                
                # Get indices of tokens to keep
                filtered_indices = sorted_indices[mask]
                
                # Update the keep mask for this sequence position
                keep_masks[s, filtered_indices] = True
            
            # Apply all masks for this batch at once (vectorized over sequence dimension)
            penalty_tensor = torch.full_like(scores[b], self.penalty)
            new_scores[b] = torch.where(keep_masks, scores[b], penalty_tensor)
        
        return new_scores
    

class DefaultLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        assert isinstance(temperature, float) and temperature > 0, f"temperature must be a positive float, got {temperature}"
        self.temperature = temperature
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert isinstance(scores, torch.FloatTensor) or isinstance(scores, torch.Tensor), f"scores must be a Tensor, got {type(scores)}"
        print("Calling DefaultLogitsProcessor, input_ids shape:", input_ids.shape)
        return scores / self.temperature



"""
NoContrastVHeadLogitsProcessor:
    This logits processor is used to penalize tokens that are not in the V_head of the good model.    
"""
class NoContrastVHeadLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature: float = 1.0, alpha: float = 0.1, penalty: float = -20.0):
        assert isinstance(temperature, float) and temperature > 0, f"temperature must be a positive float, got {temperature}"
        assert isinstance(alpha, float) and 0.0 < alpha < 1.0, f"alpha must be a float between 0 and 1, got {alpha}"
        self.temperature = temperature
        self.alpha = alpha
        self.penalty = penalty
    
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores / self.temperature
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1, keepdim=True)
        v_head_mask = probs > (self.alpha * max_probs)
        log_probs = torch.log(probs + 1e-10)
        penalty_tensor = torch.full_like(scores, self.penalty)
        v_head_logits = torch.where(v_head_mask, log_probs, penalty_tensor)
        return v_head_logits


class ContrastiveLogitsProcessor(LogitsProcessor):
    def __init__(self, contrast_strength: float = 0.1, alpha: float = 0.1, temperature: float = 1.0, penalty: float = -1000.0):
        """
        Contrastive decoding as per the original paper "Contrastive Decoding: Open-ended Text Generation as Optimization".
        
        The algorithm computes: log(p_good) - contrast_strength * log(p_bad) for tokens in V_head,
        where V_head is the set of tokens with p_good > alpha * max(p_good)
        
        Args:
            contrast_strength: Coefficient for amateur model contribution in log space
            alpha: Threshold coefficient for V_head vocabulary filtering (0 < alpha < 1)
            temperature: Temperature for softmax
            penalty: Penalty value for tokens not in V_head
        """
        assert isinstance(contrast_strength, float) and contrast_strength >= 0.0, \
            f"contrast_strength must be non-negative, got {contrast_strength}"
        assert isinstance(alpha, float) and 0.0 < alpha < 1.0, \
            f"alpha must be between 0 and 1, got {alpha}"
        assert isinstance(temperature, float) and temperature > 0, \
            f"temperature must be positive, got {temperature}"
        
        self.contrast_strength = contrast_strength
        self.alpha = alpha
        self.temperature = temperature
        self.penalty = penalty
    
    def __call__(self, input_ids: torch.LongTensor, good_logits: torch.FloatTensor, bad_logits: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply contrastive decoding: log(p_good) - contrast_strength * log(p_bad) for tokens in V_head
        
        Args:
            input_ids: Current token ids (not used but kept for API consistency)
            good_logits: Expert/good model logits
            bad_logits: Amateur/bad model logits
            
        Returns:
            Modified logits after contrastive decoding
        """
        # Use the vectorized implementation by default
        return self._vectorized_implementation(good_logits, bad_logits)
    
    def _vectorized_implementation(self, good_logits: torch.FloatTensor, bad_logits: torch.FloatTensor) -> torch.FloatTensor:
        """
        Vectorized implementation of contrastive decoding
        """
        # Apply temperature scaling
        good_logits = good_logits / self.temperature
        bad_logits = bad_logits / self.temperature
        
        # Convert to probability distributions (along last dimension)
        good_probs = F.softmax(good_logits, dim=-1)
        bad_probs = F.softmax(bad_logits, dim=-1)
        
        # Find max probabilities for V_head filtering
        max_good_probs, _ = good_probs.max(dim=-1, keepdim=True)
        
        # Create V_head mask: tokens with p_good > alpha * max(p_good)
        v_head_mask = good_probs > (self.alpha * max_good_probs)
        
        # Compute log probabilities
        log_good_probs = torch.log(good_probs + 1e-10)
        log_bad_probs = torch.log(bad_probs + 1e-10)
        
        # Apply contrastive decoding formula
        contrast_scores = log_good_probs - self.contrast_strength * log_bad_probs
        
        # Create tensor of penalties with same shape as logits
        penalty_tensor = torch.full_like(good_logits, self.penalty)
        
        # Apply V_head filtering using mask
        contrastive_logits = torch.where(v_head_mask, contrast_scores, penalty_tensor)
        #print("Contrastive logits shape:", contrastive_logits.shape)
        return contrastive_logits
    


class ContrastiveTailLogitsProcessor(LogitsProcessor):
    def __init__(self, contrast_strength: float = 0.1, alpha: float = 0.1, temperature: float = 1.0):
        """
        Contrastive decoding as per "Contrastive Decoding: Open-ended Text Generation as Optimization".

        Args:
            contrast_strength: Coefficient for amateur model contribution in log space
            alpha: Threshold coefficient for V_head vocabulary filtering (0 < alpha < 1)
            temperature: Temperature for softmax
        """
        assert isinstance(contrast_strength, float) and contrast_strength >= 0.0, \
            f"contrast_strength must be non-negative, got {contrast_strength}"
        assert isinstance(alpha, float) and 0.0 < alpha < 1.0, \
            f"alpha must be between 0 and 1, got {alpha}"
        assert isinstance(temperature, float) and temperature > 0, \
            f"temperature must be positive, got {temperature}"

        self.contrast_strength = contrast_strength
        self.alpha = alpha
        self.temperature = temperature
        print(f"ContrastiveTailLogitsProcessor initialized with contrast_strength={contrast_strength}, alpha={alpha}, temperature={temperature}")

    def __call__(self, input_ids: torch.LongTensor, good_logits: torch.FloatTensor, bad_logits: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply contrastive decoding: log(p_good) - contrast_strength * log(p_bad) for tokens in V_head,
        and expert logits for tail, shifted down to match min(head).

        Args:
            input_ids: Current token ids (not used but kept for API consistency)
            good_logits: Expert/good model logits
            bad_logits: Amateur/bad model logits

        Returns:
            Modified logits after contrastive decoding
        """
        return self._vectorized_implementation(good_logits, bad_logits)

    def _vectorized_implementation(self, good_logits, bad_logits):
        # Temperature scaling
        good_logits = good_logits / self.temperature
        bad_logits = bad_logits / self.temperature

        # Compute probabilities and logprobs
        good_probs = F.softmax(good_logits, dim=-1)
        bad_probs = F.softmax(bad_logits, dim=-1)
        
        log_good_probs = torch.log(good_probs + 1e-10)
        log_bad_probs = torch.log(bad_probs + 1e-10)

        # V_head mask
        max_good_probs, _ = good_probs.max(dim=-1, keepdim=True)
        v_head_mask = good_probs > (self.alpha * max_good_probs)
        v_tail_mask = ~v_head_mask

        # Head: contrastive logprobs
        contrastive_logits = log_good_probs - self.contrast_strength * log_bad_probs

        # Tail: just the expert's logprobs
        tail_logits = log_good_probs

        # Alignment for smooth boundary
        contrastive_logits_head_only = torch.where(v_head_mask, contrastive_logits, torch.full_like(contrastive_logits, 1e9))
        min_head, _ = contrastive_logits_head_only.min(dim=-1, keepdim=True)
        tail_logits_tail_only = torch.where(v_tail_mask, tail_logits, torch.full_like(tail_logits, -1e9))
        max_tail, _ = tail_logits_tail_only.max(dim=-1, keepdim=True)

        tail_empty = (v_tail_mask.sum(dim=-1, keepdim=True) == 0)
        head_empty = (v_head_mask.sum(dim=-1, keepdim=True) == 0)

        shift = min_head - max_tail
        shift = torch.where(tail_empty, torch.full_like(shift, -1e9), shift)
        shift = torch.where(head_empty, torch.zeros_like(shift), shift)

        shifted_tail_logits = tail_logits + shift

        output_logits = torch.where(v_head_mask, contrastive_logits, shifted_tail_logits)
        return output_logits





########################################################################################
# Contrastive + no-repeat-n-gram wrapper.
# Usage:
#   base = ContrastiveLogitsProcessor(...your params...)
#   proc = ContrastiveWithNoRepeatNGram(base, no_repeat_ngram_size=3)
#   logits = proc(input_ids, good_logits, bad_logits)

from typing import Optional
import torch
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor as HFNoRepeatNGram



class ContrastiveWithNoRepeatNGram:
    """
    Wraps an existing contrastive-decoding logits processor and bans tokens that would
    recreate any previously seen n-gram of length `no_repeat_ngram_size`.

    Expected calling convention (matches your contrastive processor):
        __call__(input_ids: LongTensor[B, T], good_logits: Tensor, bad_logits: Tensor) -> Tensor
    Returns logits of shape [B, V] or [B, T, V] (same as your inner processor).
    """

    def __init__(self, inner_processor, no_repeat_ngram_size: int):
        if no_repeat_ngram_size is None or no_repeat_ngram_size < 1:
            raise ValueError("no_repeat_ngram_size must be a positive integer.")
        self.inner = inner_processor
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)

        if HFNoRepeatNGram is not None:
            self._hf = HFNoRepeatNGram(self.no_repeat_ngram_size)
        else:
            self._hf = None  # Fallback will be used

    def __repr__(self) -> str:
        name = getattr(self.inner, "__class__", type(self.inner)).__name__
        return f"ContrastiveWithNoRepeatNGram(inner={name}, n={self.no_repeat_ngram_size})"

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, good_logits: torch.Tensor, bad_logits: torch.Tensor) -> torch.Tensor:
        # 1) Get contrastive-combined logits from your existing processor
        scores = self.inner(input_ids, good_logits, bad_logits)  # [B,V] or [B,T,V]

        if scores.dim() == 3:
            # Only the last step's distribution is relevant for the next token
            last_scores = scores[:, -1, :]  # [B, V]
            masked = self._apply_no_repeat(input_ids, last_scores)
            # Write back
            out = scores.clone()
            out[:, -1, :] = masked
            return out
        elif scores.dim() == 2:
            # Already a step-wise distribution
            return self._apply_no_repeat(input_ids, scores)
        else:
            raise ValueError(f"Unexpected logits shape {tuple(scores.shape)}; expected [B,V] or [B,T,V].")

    def _apply_no_repeat(self, input_ids: torch.LongTensor, step_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply no-repeat-n-gram blocking to `step_scores` (shape [B, V]) based on `input_ids` (shape [B, T]).
        Uses HF's processor if available; otherwise falls back to a simple Python implementation.
        """
        # Ensure tensor types and device alignment
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=step_scores.device)
        else:
            input_ids = input_ids.to(device=step_scores.device, dtype=torch.long)
     
        return self._hf(input_ids, step_scores)



class NoContrastWithNoRepeatNGram:
    """
    Wraps an existing contrastive-decoding logits processor and bans tokens that would
    recreate any previously seen n-gram of length `no_repeat_ngram_size`.

    Expected calling convention (matches your contrastive processor):
        __call__(input_ids: LongTensor[B, T], good_logits: Tensor) -> Tensor
    Returns logits of shape [B, V] or [B, T, V] (same as your inner processor).
    """

    def __init__(self, no_repeat_ngram_size: int):
        if no_repeat_ngram_size is None or no_repeat_ngram_size < 1:
            raise ValueError("no_repeat_ngram_size must be a positive integer.")
        
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)

        if HFNoRepeatNGram is not None:
            self._hf = HFNoRepeatNGram(self.no_repeat_ngram_size)
        else:
            self._hf = None  # Fallback will be used

    def __repr__(self) -> str:
        return f"NoContrastWithNoRepeatNGram(n={self.no_repeat_ngram_size})"

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, good_logits: torch.Tensor) -> torch.Tensor:
        # 1) Get contrastive-combined logits from your existing processor
        scores = good_logits

        if scores.dim() == 3:
            # Only the last step's distribution is relevant for the next token
            last_scores = scores[:, -1, :]  # [B, V]
            masked = self._apply_no_repeat(input_ids, last_scores)
            # Write back
            out = scores.clone()
            out[:, -1, :] = masked
            return out
        elif scores.dim() == 2:
            # Already a step-wise distribution
            return self._apply_no_repeat(input_ids, scores)
        else:
            raise ValueError(f"Unexpected logits shape {tuple(scores.shape)}; expected [B,V] or [B,T,V].")

    def _apply_no_repeat(self, input_ids: torch.LongTensor, step_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply no-repeat-n-gram blocking to `step_scores` (shape [B, V]) based on `input_ids` (shape [B, T]).
        Uses HF's processor if available; otherwise falls back to a simple Python implementation.
        """
        # Ensure tensor types and device alignment
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=step_scores.device)
        else:
            input_ids = input_ids.to(device=step_scores.device, dtype=torch.long)
     
        return self._hf(input_ids, step_scores)
       


class ContrastiveWithTopP:
    def __init__(self, top_p: float = 0.9, contrast_strength: float = 0.1, alpha: float = 0.1, temperature: float = 1.0, penalty: float = -1000.0):
        self.top_p = top_p
        self.contrast_strength = contrast_strength
        self.alpha = alpha
        self.temperature = temperature
        self.penalty = penalty
        self.contrastive_processor = ContrastiveLogitsProcessor(contrast_strength=contrast_strength, alpha=alpha, temperature=temperature, penalty=penalty)
        self.top_p_processor = TopPLogitsProcessor(top_p=top_p, temperature=temperature, penalty=penalty)

    def __call__(self, input_ids: torch.LongTensor, good_logits: torch.Tensor, bad_logits: torch.Tensor) -> torch.Tensor:
        
        contrastive_logits = self.contrastive_processor(input_ids, good_logits, bad_logits)
        top_p_logits = self.top_p_processor(input_ids, contrastive_logits)
        return top_p_logits


class ContrastiveWithTopK:
    def __init__(self, top_k: int = 50, contrast_strength: float = 0.1, alpha: float = 0.1, temperature: float = 1.0, penalty: float = -1000.0):
        self.top_k = top_k
        self.contrast_strength = contrast_strength
        self.alpha = alpha
        self.temperature = temperature
        self.penalty = penalty
        self.contrastive_processor = ContrastiveLogitsProcessor(contrast_strength=contrast_strength, alpha=alpha, temperature=temperature, penalty=penalty)
        self.top_k_processor = TopKLogitsProcessor(top_k=top_k, temperature=temperature, penalty=penalty)

    def __call__(self, input_ids: torch.LongTensor, good_logits: torch.Tensor, bad_logits: torch.Tensor) -> torch.Tensor:
        contrastive_logits = self.contrastive_processor(input_ids, good_logits, bad_logits)
        top_k_logits = self.top_k_processor(input_ids, contrastive_logits)
        return top_k_logits


########################################################################################
#



def main():
    import time
    
    # Create test inputs with larger vocabulary and more varied scores
    batch_size, seq_len, vocab_size = 20, 20, 1000  # Increased vocab_size for more realistic testing
    
    # Create more varied logits with a power-law distribution (more realistic)
    # This will create scores where a few tokens have much higher prob than others
    base_scores = torch.rand(batch_size, seq_len, vocab_size)
    # Apply power to create more variance
    scores = (base_scores ** 3) * 10  # Cubic function creates more separation
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create a larger test set for timing comparisons
    large_batch_size, large_seq_len = 64, 64  # More typical for real-world scenarios
    large_base_scores = torch.rand(large_batch_size, large_seq_len, vocab_size)
    large_scores = (large_base_scores ** 3) * 10
    large_input_ids = torch.randint(0, vocab_size, (large_batch_size, large_seq_len))
    
    print("=== Testing TopKLogitsProcessor ===")
    
    # Create processor
    k = 50  # More realistic top-k value
    processor = TopKLogitsProcessor(top_k=k, temperature=1.0)
    
    # Verify that vectorized implementation works correctly
    result_vectorized = processor(input_ids, scores.clone())
    
    # Use the loop implementation for comparison
    result_loop = processor._loop_implementation(scores.clone() / processor.temperature)
    
    # Check if both implementations produce the same result
    is_same = torch.allclose(result_vectorized, result_loop)
    print(f"TopK vectorized implementation matches loop implementation: {is_same}")
    
    if not is_same:
        # Print differences to help debug
        diff = torch.abs(result_vectorized - result_loop).max().item()
        print(f"Maximum difference: {diff}")
        
        # Check if the non-matching elements are just differences in the penalty value
        penalty_mask_vec = result_vectorized <= -900
        penalty_mask_loop = result_loop <= -900
        penalty_masks_match = (penalty_mask_vec == penalty_mask_loop).all().item()
        print(f"Penalty masks match: {penalty_masks_match}")
    
    # Performance benchmark
    print("\nTopK Performance Benchmark:")
    # Warm-up runs
    for _ in range(3):
        processor(large_input_ids, large_scores.clone())
        processor._loop_implementation(large_scores.clone() / processor.temperature)
    
    # Timing for vectorized implementation (default)
    start_time = time.time()
    for _ in range(10):
        processor(large_input_ids, large_scores.clone())
    vectorized_time = (time.time() - start_time) / 10
    
    # Timing for loop implementation
    start_time = time.time()
    for _ in range(10):
        processor._loop_implementation(large_scores.clone() / processor.temperature)
    loop_time = (time.time() - start_time) / 10
    
    print(f"  Vectorized implementation: {vectorized_time:.6f} seconds per call")
    print(f"  Loop implementation: {loop_time:.6f} seconds per call")
    print(f"  Speedup: {loop_time / vectorized_time:.2f}x")
    
    print("\n=== Testing TopPLogitsProcessor ===")
    
    # Create processor
    p = 0.9
    processor = TopPLogitsProcessor(top_p=p, temperature=1.0)
    
    # Test and compare implementations
    # Default is better_vectorized_implementation
    result_better = processor(input_ids, scores.clone())
    
    # Test the original vectorized implementation
    # Save the current method
    original_method = processor._better_vectorized_implementation
    # Temporarily replace with original vectorized method
    processor._better_vectorized_implementation = processor._vectorized_implementation
    result_vectorized = processor(input_ids, scores.clone())
    # Restore the better method
    processor._better_vectorized_implementation = original_method
    
    # Test loop implementation
    result_loop = processor._loop_implementation(scores.clone() / processor.temperature)
    
    # Check if implementations produce the same results
    is_same_vec = torch.allclose(result_vectorized, result_loop)
    is_same_better = torch.allclose(result_better, result_loop)
    print(f"Original vectorized implementation matches loop: {is_same_vec}")
    print(f"Better vectorized implementation matches loop: {is_same_better}")
    
    # Performance benchmark
    print("\nTopP Performance Benchmark:")
    # Warm-up runs
    for _ in range(3):
        processor(large_input_ids, large_scores.clone())  # Uses better vectorized implementation
        processor._vectorized_implementation(large_scores.clone() / processor.temperature)
        processor._loop_implementation(large_scores.clone() / processor.temperature)
    
    # Timing for better vectorized implementation (default)
    start_time = time.time()
    for _ in range(10):
        processor(large_input_ids, large_scores.clone())
    better_time = (time.time() - start_time) / 10
    
    # Timing for original vectorized implementation
    start_time = time.time()
    for _ in range(10):
        processor._vectorized_implementation(large_scores.clone() / processor.temperature)
    vectorized_time = (time.time() - start_time) / 10
    
    # Timing for loop implementation
    start_time = time.time()
    for _ in range(10):
        processor._loop_implementation(large_scores.clone() / processor.temperature)
    loop_time = (time.time() - start_time) / 10
    
    print(f"  Better vectorized implementation (default): {better_time:.6f} seconds per call")
    print(f"  Original vectorized implementation: {vectorized_time:.6f} seconds per call")
    print(f"  Loop implementation: {loop_time:.6f} seconds per call")
    print(f"  Speedup (better vs original): {vectorized_time / better_time:.2f}x")
    print(f"  Speedup (better vs loop): {loop_time / better_time:.2f}x")
    
    print("\n=== Testing DefaultLogitsProcessor ===")
    
    # Test default implementation
    temp = 0.8
    processor = DefaultLogitsProcessor(temperature=temp)
    
    # Performance benchmark
    print("\nDefaultLogitsProcessor Performance Benchmark:")
    # Timing for implementation
    start_time = time.time()
    for _ in range(100):  # More iterations since this is very fast
        processor(large_input_ids, large_scores.clone())
    default_time = (time.time() - start_time) / 100
    
    print(f"  Implementation: {default_time:.6f} seconds per call")
    
    print("\n=== Testing ContrastiveLogitsProcessor ===")
    
    # Create good and bad model logits with different distributions
    good_logits = scores.clone()
    # Create bad model logits with less certainty (flatter distribution)
    bad_logits = scores.clone() * 0.5 + torch.rand_like(scores) * 2
    
    large_good_logits = large_scores.clone()
    large_bad_logits = large_scores.clone() * 0.5 + torch.rand_like(large_scores) * 2
    
    # Create processor
    contrast_strength = 0.1  # As recommended in the paper
    alpha = 0.1  # For V_head filtering
    
    processor = ContrastiveLogitsProcessor(contrast_strength=contrast_strength, alpha=alpha, temperature=1.0)
    
    # Test and compare implementations
    result_vectorized = processor(input_ids, good_logits.clone(), bad_logits.clone())
    result_loop = processor._loop_implementation(good_logits.clone(), bad_logits.clone())
    
    # Check if implementations produce the same results
    is_same = torch.allclose(result_vectorized, result_loop)
    print(f"Vectorized implementation matches loop implementation: {is_same}")
    
    if not is_same:
        # Print differences to help debug
        diff = torch.abs(result_vectorized - result_loop).max().item()
        print(f"Maximum difference: {diff}")
        
        # Check if the non-matching elements are just differences in the penalty value
        penalty_mask_vec = result_vectorized <= -900
        penalty_mask_loop = result_loop <= -900
        penalty_masks_match = (penalty_mask_vec == penalty_mask_loop).all().item()
        print(f"Penalty masks match: {penalty_masks_match}")
    
    # Performance benchmark
    print("\nContrastive Performance Benchmark:")
    # Warm-up runs
    for _ in range(3):
        processor(large_input_ids, large_good_logits.clone(), large_bad_logits.clone())
        processor._loop_implementation(large_good_logits.clone(), large_bad_logits.clone())
    
    # Timing for vectorized implementation (default)
    start_time = time.time()
    for _ in range(10):
        processor(large_input_ids, large_good_logits.clone(), large_bad_logits.clone())
    vectorized_time = (time.time() - start_time) / 10
    
    # Timing for loop implementation
    start_time = time.time()
    for _ in range(10):
        processor._loop_implementation(large_good_logits.clone(), large_bad_logits.clone())
    loop_time = (time.time() - start_time) / 10
    
    print(f"  Vectorized implementation (default): {vectorized_time:.6f} seconds per call")
    print(f"  Loop implementation: {loop_time:.6f} seconds per call")
    print(f"  Speedup: {loop_time / vectorized_time:.2f}x")
    
    print("\nAll tests completed successfully!")
    
if __name__ == "__main__":
    main() 