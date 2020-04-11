from typing import Optional


class FinCausalExample:

    def __init__(
            self,
            example_id: str,
            context_text: str,
            cause_text: str,
            effect_text: str,
            cause_start_position_character: Optional[int],
            cause_end_position_character: Optional[int],
            effect_start_position_character: Optional[int],
            effect_end_position_character: Optional[int]
    ):

        self.example_id = example_id
        self.context_text = context_text
        self.cause_text = cause_text
        self.effect_text = effect_text

        self.start_cause_position, self.end_cause_position = 0, 0
        self.start_effect_position, self.end_effect_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if cause_start_position_character is not None:
            assert(cause_start_position_character + len(cause_text) == cause_end_position_character)
            self.start_cause_position = char_to_word_offset[cause_start_position_character]
            self.end_cause_position = char_to_word_offset[
                min(cause_start_position_character + len(cause_text) - 1, len(char_to_word_offset) - 1)
            ]
        if effect_start_position_character is not None:
            self.start_effect_position = char_to_word_offset[effect_start_position_character]
            assert(effect_start_position_character + len(effect_text) == effect_end_position_character)
            self.end_effect_position = char_to_word_offset[
                min(effect_start_position_character + len(effect_text) - 1, len(char_to_word_offset) - 1)
            ]


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
