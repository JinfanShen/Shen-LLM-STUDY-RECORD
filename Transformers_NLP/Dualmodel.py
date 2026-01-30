from transformers import  BertPreTrainedModel, BertModel
from transformers.configuration_utils import PretrainedConfig
from typing import Optional
import torch
from torch.nn import CosineSimilarity, CosineEmbeddingLoss

class DualModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Step1 分别获取sentenceA和sentenceB的输入
        senA_input_ids,  senB_input_ids = input_ids.chunk(2, dim=1)  # [batch, 1, seq_len]
        senA_attention_mask, senB_attention_mask = attention_mask.chunk(2, dim=1)
        senA_token_type_ids, senB_token_type_ids = token_type_ids.chunk(2, dim=1)
        # Step2 获得两个句子的向量表示
        senA_outputs = self.bert(
            senA_input_ids.squeeze(1), #chunk会保留为1的维度
            attention_mask=senA_attention_mask.squeeze(1),
            token_type_ids=senA_token_type_ids.squeeze(1),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
        senA_pooled_output = senA_outputs[1]  #[batch, hidden]

        SenB_outputs = self.bert(
            senB_input_ids.squeeze(1),
            attention_mask=senB_attention_mask.squeeze(1),
            token_type_ids=senB_token_type_ids.squeeze(1),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        senB_pooled_output = SenB_outputs[1]  #[batch, hidden]

        # Step3 计算相似度
        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)

        # Step4 计算loss
        loss = None
        if labels is not None:
            loss_fct = CosineEmbeddingLoss(0.3)
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels)

        output = (cos,)
        return ((loss,) + output) if loss is not None else output