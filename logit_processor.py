from collections.abc import Callable
from typing import Any

import torch
from transformers.generation import LogitsProcessor


class ConstrainedLogitsProcessor(LogitsProcessor):
    """based on trie to restrict next token candidates."""

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, list[int]], list[int]],
        num_beams: int,
        prefix_index: int = 3,
        prefix_ids: list[int] = None,
        eos_token_id: int = None,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.prefix_index = prefix_index
        self.eos_token_id = eos_token_id
        self.prefix_ids = prefix_ids
        self.count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        # support beam search: (batch*beams, seq) -> (batch, beams, seq)
        assert input_ids.dim() == 2, "input_ids must be a 2D tensor"
        input_ids.shape[0]
        seq_len = input_ids.shape[1]
        beam_sents = input_ids.view(-1, self._num_beams, seq_len)
        for batch_id, beam_sent in enumerate(beam_sents):
            for beam_id, sent in enumerate[Any](beam_sent):
                if sent[-self.prefix_index :].tolist() == self.prefix_ids:
                    self.count = 0
                prefix_ids = sent[-self.prefix_index - self.count :].tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(prefix_ids)
                if len(prefix_allowed_tokens) == 0:
                    assert len(prefix_allowed_tokens) > 0, "No valid tokens for prefix_ids"
                idx = batch_id * self._num_beams + beam_id
                mask[idx, prefix_allowed_tokens] = 0

        self.count += 1
        return scores + mask
