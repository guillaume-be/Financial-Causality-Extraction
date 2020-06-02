from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class BertForCauseEffect(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForCauseEffect, self).__init__(config)

        self.bert = BertModel(config)
        self.cause_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.effect_outputs = nn.Linear(config.hidden_size, config.num_labels)
        assert config.num_labels == 2
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_cause_positions=None,
            end_cause_positions=None,
            start_effect_positions=None,
            end_effect_positions=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        hidden_states = bert_output[0]  # (bs, max_query_len, dim)
        cause_logits = self.cause_outputs(hidden_states)  # (bs, max_query_len, 2)
        effect_logits = self.effect_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_cause_logits, end_cause_logits = cause_logits.split(1, dim=-1)
        start_effect_logits, end_effect_logits = effect_logits.split(1, dim=-1)

        start_cause_logits = start_cause_logits.squeeze(-1)  # (bs, max_query_len)
        end_cause_logits = end_cause_logits.squeeze(-1)  # (bs, max_query_len)
        start_effect_logits = start_effect_logits.squeeze(-1)  # (bs, max_query_len)
        end_effect_logits = end_effect_logits.squeeze(-1)  # (bs, max_query_len)

        outputs = (start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits,) \
                  + bert_output[2:]
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


class BertForCauseEffectClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cause_effect_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = 2
        assert config.num_labels == 2
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        hidden_states = bert_output[1]  # (bs, max_query_len, dim)
        logits = self.cause_effect_classifier(hidden_states)  # (bs, max_query_len, 2)

        outputs = (logits,) + bert_output[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
