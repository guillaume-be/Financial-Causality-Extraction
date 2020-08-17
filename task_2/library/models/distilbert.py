# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Guillaume Becquin.
# MODIFIED FOR CAUSE EFFECT EXTRACTION
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

from torch import nn
from transformers import DistilBertPreTrainedModel, DistilBertModel


class DistilBertForCauseEffect(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.cause_outputs = nn.Linear(config.dim, config.num_labels)
        self.effect_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            start_cause_positions=None,
            end_cause_positions=None,
            start_effect_positions=None,
            end_effect_positions=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        cause_logits = self.cause_outputs(hidden_states)  # (bs, max_query_len, 2)
        effect_logits = self.effect_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_cause_logits, end_cause_logits = cause_logits.split(1, dim=-1)
        start_effect_logits, end_effect_logits = effect_logits.split(1, dim=-1)

        start_cause_logits = start_cause_logits.squeeze(-1)  # (bs, max_query_len)
        end_cause_logits = end_cause_logits.squeeze(-1)  # (bs, max_query_len)
        start_effect_logits = start_effect_logits.squeeze(-1)  # (bs, max_query_len)
        end_effect_logits = end_effect_logits.squeeze(-1)  # (bs, max_query_len)

        outputs = (start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits,) \
                  + distilbert_output[1:]
        if start_cause_positions is not None \
                and end_cause_positions is not None \
                and start_effect_positions is not None \
                and end_effect_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_cause_logits.size(1)
            start_cause_positions.clamp_(0, ignored_index)
            end_cause_positions.clamp_(0, ignored_index)
            start_effect_positions.clamp_(0, ignored_index)
            end_effect_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_cause_loss = loss_fct(start_cause_logits, start_cause_positions)
            end_cause_loss = loss_fct(end_cause_logits, end_cause_positions)
            start_effect_loss = loss_fct(start_effect_logits, start_effect_positions)
            end_effect_loss = loss_fct(end_effect_logits, end_effect_positions)
            total_loss = (start_cause_loss + end_cause_loss + start_effect_loss + end_effect_loss) / 4
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits(hidden_states), (attentions)
