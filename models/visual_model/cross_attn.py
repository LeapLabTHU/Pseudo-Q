# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.cross_hidden_size % config.cross_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.cross_hidden_size, config.cross_num_attention_heads))
        self.cross_num_attention_heads = config.cross_num_attention_heads
        self.cross_attention_head_size = int(config.cross_hidden_size / config.cross_num_attention_heads)
        self.cross_all_head_size = self.cross_num_attention_heads * self.cross_attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.cross_hidden_size
        self.query = nn.Linear(config.cross_hidden_size, self.cross_all_head_size)
        self.key = nn.Linear(ctx_dim, self.cross_all_head_size)
        self.value = nn.Linear(ctx_dim, self.cross_all_head_size)

        self.dropout = nn.Dropout(config.cross_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AttOutput(nn.Module):
    def __init__(self, config):
        super(AttOutput, self).__init__()
        self.dense = nn.Linear(config.cross_hidden_size, config.cross_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.cross_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.cross_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class cross_module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = Attention(config)
        self.output = AttOutput(config)

    def act(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

    def forward(self, visn_input, lang_input, visn_attention_mask, lang_attention_mask):
        lang_att_output = self.act(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.act(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output
