import math
from typing import List

import editdistance
import numpy as np


class FuzzyMatcher:

    @staticmethod
    def fuzzy_match_freeform(
        gt_words: List[str], text_words: List[str], match_similarity_threshold: float
    ) -> np.ndarray:
        tagged_segments = np.zeros(len(text_words), dtype=np.int)
        num_gt_words = len(gt_words)
        num_text_words = len(text_words)
        num_allowed_char_errors = math.floor(
            (1.0 - match_similarity_threshold) * sum(len(gt_word) for gt_word in gt_words)
        )
        total_gt_window_errors = np.zeros(num_text_words - num_gt_words + 1, dtype=np.int)
        for gt_word_idx, gt_word in enumerate(gt_words):
            gt_word_errors = np.array(
                [
                    editdistance.eval(gt_word, text_word)
                    for text_word in text_words[
                        gt_word_idx: (gt_word_idx + num_text_words - num_gt_words + 1)
                    ]
                ]
            )
            total_gt_window_errors += gt_word_errors

        prev_start_word_idx = None
        for candidate_start_word_idx in (
            total_gt_window_errors <= num_allowed_char_errors
        ).nonzero()[0]:
            best_start_word_idx_in_window = total_gt_window_errors[
                candidate_start_word_idx: candidate_start_word_idx + num_gt_words
            ].argmin()
            start_word_idx = candidate_start_word_idx + best_start_word_idx_in_window
            if prev_start_word_idx is None or start_word_idx >= prev_start_word_idx + num_gt_words:
                tagged_segments[start_word_idx: start_word_idx + num_gt_words] = 1
                prev_start_word_idx = start_word_idx

        return tagged_segments
