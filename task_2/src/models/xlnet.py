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
from torch.nn import CrossEntropyLoss
from transformers import XLNetPreTrainedModel, XLNetModel


class XLNetForCauseEffect(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.cause_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.effect_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=True,
            start_cause_positions=None,
            end_cause_positions=None,
            start_effect_positions=None,
            end_effect_positions=None,
    ):

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        sequence_output = outputs[0]

        cause_logits = self.cause_outputs(sequence_output)
        start_cause_logits, end_cause_logits = cause_logits.split(1, dim=-1)
        start_cause_logits = start_cause_logits.squeeze(-1)
        end_cause_logits = end_cause_logits.squeeze(-1)

        effect_logits = self.effect_outputs(sequence_output)
        start_effect_logits, end_effect_logits = effect_logits.split(1, dim=-1)
        start_effect_logits = start_effect_logits.squeeze(-1)
        end_effect_logits = end_effect_logits.squeeze(-1)

        outputs = (start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits,) + outputs[2:]
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

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_cause_loss = loss_fct(start_cause_logits, start_cause_positions)
            end_cause_loss = loss_fct(end_cause_logits, end_cause_positions)
            start_effect_loss = loss_fct(start_effect_logits, start_effect_positions)
            end_effect_loss = loss_fct(end_effect_logits, end_effect_positions)
            total_loss = (start_cause_loss + end_cause_loss + start_effect_loss + end_effect_loss) / 4
            outputs = (total_loss,) + outputs

        return outputs
