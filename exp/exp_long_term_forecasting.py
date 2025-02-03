import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
)

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, get_lr_scheduler, visual
from layers.CrossModal import CrossModal
from layers.RevIN import RevIN
from layers.Embed import DataEmbedding
from exp.exp_basic import Exp_Basic


def norm(input_emb):
    input_emb = input_emb - input_emb.mean(1, keepdim=True).detach()
    input_emb = input_emb / torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5
    )

    return input_emb


def select_eos_token(prompt, eos_token_id):
    token_pos = torch.argmax((prompt == eos_token_id).type(torch.float), dim=1)
    token_pos = torch.where(token_pos == 0, prompt.size()[1], token_pos)
    return token_pos


def prepare_prompt(batch_text, batch_x, text_ts=False, text_avg=False):
    vx = batch_x.detach().clone().squeeze(2)
    tx = []
    for txt, vals in zip(batch_text, vx):
        values = vals.tolist()
        prompt = txt
        if text_ts:
            values_str = ", ".join([f"{value:.3f}" for value in values])
            prompt = f"The values are {values_str}. {prompt}".strip()
        if text_avg:
            values_avg = np.mean(values)
            prompt = f"The average value is {values_avg:.3f}. {prompt}".strip()
        tx.append(prompt)
    return tx


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        for i in range(len(layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class LLMTSBase(nn.Module):
    def _build_llm_model(self, config):
        if config.llm_model == "LLAMA2":
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = config.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            self.llm_model = LlamaModel.from_pretrained(
                "huggyllama/llama-7b",
                trust_remote_code=True,
                local_files_only=False,
                config=self.llama_config,
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(
                "huggyllama/llama-7b",
                trust_remote_code=True,
                local_files_only=False,
            )
        elif config.llm_model == "LLAMA3":
            llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            cache_path = "/localscratch/hliu763/"

            # Load the configuration with custom adjustments
            self.llama_config = LlamaConfig.from_pretrained(
                llama3_path, token="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", cache_dir=cache_path
            )

            self.llama_config.num_hidden_layers = config.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            self.llm_model = LlamaModel.from_pretrained(
                llama3_path,
                config=self.llama_config,
                token="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",
                cache_dir=cache_path,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                llama3_path,
                use_auth_token="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",
                cache_dir=cache_path,
            )
        elif config.llm_model == "GPT2":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")

            self.gpt2_config.num_hidden_layers = config.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            if config.llm_from_stratch:
                self.llm_model = GPT2Model(self.gpt2_config)
            else:
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2",
                local_files_only=False,
            )
        elif config.llm_model == "GPT2M":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2-medium")

            self.gpt2_config.num_hidden_layers = config.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2-medium",
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2-medium", trust_remote_code=True, local_files_only=False
            )
        elif config.llm_model == "GPT2L":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2-large")

            self.gpt2_config.num_hidden_layers = config.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2-large",
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2-large",
                trust_remote_code=True,
                local_files_only=False,
            )
        elif config.llm_model == "GPT2XL":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2-xl")

            self.gpt2_config.num_hidden_layers = config.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2-xl",
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2-xl",
                trust_remote_code=True,
                local_files_only=False,
            )
        elif config.llm_model == "BERT":
            self.bert_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")

            self.bert_config.num_hidden_layers = config.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True

            self.llm_model = BertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
                config=self.bert_config,
            )

            self.tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )
        else:
            raise Exception("LLM model is not defined")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token


class LLMTSLateFusion(LLMTSBase):
    def __init__(self, config, ts_encoder_model, ts_decoder_model):
        super().__init__()

        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.rev_in = config.rev_in

        self.d_llm = config.llm_dim
        self.text_embedding_dim = config.text_emb
        self.text_random = config.text_random
        self.text_random_order = config.text_random_order
        self.use_fullmodel = config.use_fullmodel
        self.pool_type = config.pool_type

        self.prompt_weight = config.prompt_weight
        self.fusion_method = config.fusion_method
        self.history_weight = config.history_weight

        self.output_attention = config.output_attention

        self._build_llm_model(config)

        # freeze llm model weights
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if self.fusion_method.startswith("query"):
            self.cross_modal = CrossModal(
                config.d_model,
                config.n_heads,
                d_ff=config.d_ff,
                scale=config.factor,
                attn_dropout=config.dropout,
                dropout=config.dropout,
                activation=config.activation,
            )

        mlp_sizes = [self.d_llm, self.d_llm // 8, self.text_embedding_dim]
        self.llm_proj = MLP(mlp_sizes, dropout_rate=config.connector_dropout)

        if self.rev_in:
            self.revin_layer = RevIN(config.enc_in)

        self.ts_encoder_model = ts_encoder_model
        self.ts_decoder_model = ts_decoder_model

        if self.ts_decoder_model is None:
            self.ts_proj = nn.Linear(self.seq_len, self.pred_len)
            if self.fusion_method == "concat" and self.prompt_weight > 0:
                self.out_proj = nn.Linear(config.d_model * 2, config.c_out)
            else:
                self.out_proj = nn.Linear(config.d_model, config.c_out)

    def forward(self, x, y, x_mark, y_mark, text_x, prior_y):
        if self.prompt_weight > 0:
            prompt = [f"{text_info}" for text_info in text_x]

            prompt = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).input_ids

            if self.text_random:
                min_token = torch.min(prompt).item()
                max_token = torch.max(prompt).item()
                prompt = torch.randint(low=min_token, high=max_token, size=prompt.size())

            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x.device))
            # B prompt_token

            if self.use_fullmodel:
                prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
            else:
                prompt_emb = prompt_embeddings

            prompt_emb = self.llm_proj(prompt_emb)
            # B prompt_token llm_text_emb_dim->text_emb_dim(ts_emb_dim)

            if self.pool_type == "avg":
                pooled_prompt_emb = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            elif self.pool_type == "max":
                pooled_prompt_emb = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            elif self.pool_type == "min":
                pooled_prompt_emb = F.adaptive_max_pool1d(
                    -1.0 * prompt_emb.transpose(1, 2), 1
                ).squeeze(2)
            else:
                pooled_prompt_emb = prompt_emb[:, -1, :].squeeze(1)

            prompt_y = norm(prompt_emb)
            pooled_prompt_y = norm(pooled_prompt_emb)

        # encoder
        if self.rev_in:
            x = self.revin_layer(x, "norm")

        if self.output_attention:
            ts_embedding = self.ts_encoder_model(x, x_mark, None, None)[0]
        else:
            ts_embedding = self.ts_encoder_model(x, x_mark, None, None)
        ts_embedding = ts_embedding[:, -self.seq_len :, :]

        # fusion
        if self.prompt_weight > 0:
            if self.fusion_method == "add":
                fused_embedding = (1 - self.prompt_weight) * ts_embedding
                if self.prompt_weight > 0:
                    fused_embedding += self.prompt_weight * pooled_prompt_y.unsqueeze(1)
            elif self.fusion_method == "query":
                fused_embedding = self.cross_modal(ts_embedding, prompt_y)
            elif self.fusion_method == "queryone":
                fused_embedding = self.cross_modal(ts_embedding, pooled_prompt_y.unsqueeze(1))
            elif self.fusion_method == "concat":
                fused_embedding = torch.cat(
                    [ts_embedding, pooled_prompt_y.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1
                )
            else:
                fused_embedding = ts_embedding
        else:
            fused_embedding = ts_embedding

        if self.ts_decoder_model is None:
            outputs = self.out_proj(self.ts_proj(fused_embedding.transpose(1, 2)).transpose(1, 2))
        else:
            # decoder input
            dec_inp = torch.zeros_like(y[:, -self.pred_len :, :]).float()
            dec_inp = (
                torch.cat([y[:, : self.label_len, :], dec_inp], dim=1)
                .float()
                .to(fused_embedding.device)
            )

            # decoder
            outputs = self.ts_decoder_model(fused_embedding, None, dec_inp, y_mark)

        if self.rev_in:
            outputs = self.revin_layer(outputs, "denorm")

        if self.history_weight > 0:
            outputs += self.history_weight * prior_y

        return outputs


class LLMTSEarlyFusion(LLMTSBase):
    def __init__(self, config, ts_encoder_model, ts_decoder_model):
        super().__init__()

        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.rev_in = config.rev_in

        self.d_llm = config.llm_dim
        self.text_embedding_dim = config.text_emb
        self.text_random = config.text_random
        self.text_random_order = config.text_random_order
        self.use_fullmodel = config.use_fullmodel
        self.pool_type = config.pool_type

        self.prompt_weight = config.prompt_weight
        self.fusion_method = config.fusion_method
        self.history_weight = config.history_weight

        self.output_attention = config.output_attention

        self._build_llm_model(config)

        # freeze llm model weights
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if self.fusion_method.startswith("query"):
            self.cross_modal = CrossModal(
                config.d_model,
                config.n_heads,
                d_ff=config.d_ff,
                scale=config.factor,
                attn_dropout=config.dropout,
                dropout=config.dropout,
                activation=config.activation,
            )

        mlp_sizes = [self.d_llm, self.d_llm // 8, self.text_embedding_dim]
        self.llm_proj = MLP(mlp_sizes, dropout_rate=config.connector_dropout)

        if self.rev_in:
            self.revin_layer = RevIN(config.enc_in)

        self.ts_encoder_model = ts_encoder_model
        self.ts_decoder_model = ts_decoder_model

        self.ts_embedding = DataEmbedding(
            config.enc_in, self.text_embedding_dim, config.embed, config.freq, config.dropout
        )

        if self.ts_decoder_model is None:
            self.ts_proj = nn.Linear(self.seq_len, self.pred_len)
            if self.fusion_method == "concat" and self.prompt_weight > 0:
                self.out_proj = nn.Linear(config.d_model * 2, config.c_out)
            else:
                self.out_proj = nn.Linear(config.d_model, config.c_out)

    def forward(self, x, y, x_mark, y_mark, text_x, prior_y):
        if self.prompt_weight > 0:
            prompt = [f"{text_info}" for text_info in text_x]

            prompt = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).input_ids

            if self.text_random:
                min_token = torch.min(prompt).item()
                max_token = torch.max(prompt).item() + 1
                prompt = torch.randint(low=min_token, high=max_token, size=prompt.size())

            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x.device))
            # B prompt_token

            if self.use_fullmodel:
                prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
            else:
                prompt_emb = prompt_embeddings

            prompt_emb = self.llm_proj(prompt_emb)
            # B prompt_token llm_text_emb_dim->text_emb_dim(ts_emb_dim)

            if self.pool_type == "avg":
                pooled_prompt_emb = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            elif self.pool_type == "max":
                pooled_prompt_emb = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            elif self.pool_type == "min":
                pooled_prompt_emb = F.adaptive_max_pool1d(
                    -1.0 * prompt_emb.transpose(1, 2), 1
                ).squeeze(2)
            else:
                pooled_prompt_emb = prompt_emb[:, -1, :].squeeze(1)

            prompt_y = norm(prompt_emb)
            pooled_prompt_y = norm(pooled_prompt_emb)

        # encoder
        if self.rev_in:
            x = self.revin_layer(x, "norm")

        ts_embedding = self.ts_embedding(x, x_mark)
        ts_embedding = ts_embedding[:, -self.seq_len :, :]

        # fusion
        if self.prompt_weight > 0:
            if self.fusion_method == "add":
                fused_embedding = (1 - self.prompt_weight) * ts_embedding
                if self.prompt_weight > 0:
                    fused_embedding += self.prompt_weight * pooled_prompt_y.unsqueeze(1)
            elif self.fusion_method == "query":
                fused_embedding = self.cross_modal(ts_embedding, prompt_y)
            elif self.fusion_method == "queryone":
                fused_embedding = self.cross_modal(ts_embedding, pooled_prompt_y.unsqueeze(1))
            elif self.fusion_method == "concat":
                fused_embedding = torch.cat(
                    [ts_embedding, pooled_prompt_y.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1
                )
            else:
                fused_embedding = ts_embedding
        else:
            fused_embedding = ts_embedding

        if self.output_attention:
            outputs = self.ts_encoder_model(fused_embedding, x_mark, None, None)[0]
        else:
            outputs = self.ts_encoder_model(fused_embedding, x_mark, None, None)
        outputs = outputs[:, -self.seq_len :, :]

        if self.ts_decoder_model is None:
            outputs = self.out_proj(self.ts_proj(fused_embedding.transpose(1, 2)).transpose(1, 2))
        else:
            # decoder input
            dec_inp = torch.zeros_like(y[:, -self.pred_len :, :]).float()
            dec_inp = (
                torch.cat([y[:, : self.label_len, :], dec_inp], dim=1)
                .float()
                .to(fused_embedding.device)
            )

            # decoder
            outputs = self.ts_decoder_model(fused_embedding, None, dec_inp, y_mark)

        if self.rev_in:
            outputs = self.revin_layer(outputs, "denorm")

        if self.history_weight > 0:
            outputs += self.history_weight * prior_y

        return outputs


class LLMTSLLMAsPredictor(LLMTSBase):
    def __init__(self, config, ts_encoder_model, ts_decoder_model):
        super().__init__()

        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.rev_in = config.rev_in

        self.d_llm = config.llm_dim
        self.text_embedding_dim = config.text_emb
        self.text_random = config.text_random
        self.text_random_order = config.text_random_order
        self.use_fullmodel = config.use_fullmodel
        self.pool_type = config.pool_type

        self.prompt_weight = config.prompt_weight
        self.fusion_method = config.fusion_method
        self.history_weight = config.history_weight

        self.output_attention = config.output_attention

        self._build_llm_model(config)

        # freeze llm model weights
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if self.fusion_method.startswith("query"):
            self.cross_modal = CrossModal(
                self.d_llm,
                config.n_heads,
                d_ff=config.d_ff,
                scale=config.factor,
                attn_dropout=config.dropout,
                dropout=config.dropout,
                activation=config.activation,
            )

        if self.rev_in:
            self.revin_layer = RevIN(config.enc_in)

        self.ts_encoder_model = ts_encoder_model
        self.ts_decoder_model = ts_decoder_model

        self.ts_embedding = DataEmbedding(
            config.enc_in, self.text_embedding_dim, config.embed, config.freq, config.dropout
        )

        self.ts_proj = nn.Linear(1, self.pred_len)
        if self.fusion_method == "concat" and self.prompt_weight > 0:
            self.out_proj = nn.Linear(config.d_model * 2, config.c_out)
        else:
            self.out_proj = nn.Linear(self.text_embedding_dim, config.c_out)

    def forward(self, x, y, x_mark, y_mark, text_x, prior_y):
        if self.prompt_weight > 0:
            prompt = [f"{text_info}" for text_info in text_x]

            prompt = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).input_ids

            if self.text_random:
                min_token = torch.min(prompt).item()
                max_token = torch.max(prompt).item() + 1
                prompt = torch.randint(low=min_token, high=max_token, size=prompt.size())

            prompt_emb = self.llm_model.get_input_embeddings()(prompt.to(x.device))
            # B prompt_token

            # eos_token_pos = select_eos_token(prompt, self.tokenizer.eos_token_id).to(x.device)

            if self.pool_type == "avg":
                pooled_prompt_emb = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            elif self.pool_type == "max":
                pooled_prompt_emb = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            elif self.pool_type == "min":
                pooled_prompt_emb = F.adaptive_max_pool1d(
                    -1.0 * prompt_emb.transpose(1, 2), 1
                ).squeeze(2)
            else:
                pooled_prompt_emb = prompt_emb[:, -1, :].squeeze(1)

            prompt_y = norm(prompt_emb)
            pooled_prompt_y = norm(pooled_prompt_emb)

        # encoder
        if self.rev_in:
            x = self.revin_layer(x, "norm")

        ts_emb = self.ts_embedding(x, x_mark)
        ts_emb = ts_emb[:, -self.seq_len :, :]

        # fusion
        if self.prompt_weight > 0:
            if self.fusion_method == "add":
                fused_embedding = (1 - self.prompt_weight) * ts_emb
                if self.prompt_weight > 0:
                    fused_embedding += self.prompt_weight * pooled_prompt_y.unsqueeze(1)
            elif self.fusion_method == "query":
                fused_embedding = self.cross_modal(prompt_y, ts_emb)
            elif self.fusion_method == "queryone":
                fused_embedding = self.cross_modal(ts_emb, pooled_prompt_y.unsqueeze(1))
            elif self.fusion_method == "concat":
                fused_embedding = torch.cat(
                    [ts_emb, pooled_prompt_y.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1
                )
            else:
                fused_embedding = ts_emb
        else:
            fused_embedding = ts_emb

        llm_out = self.llm_model(inputs_embeds=fused_embedding).last_hidden_state

        # B prompt_token llm_text_emb_dim->text_emb_dim(ts_emb_dim)

        llm_out = llm_out[:, -1, :].unsqueeze(1)
        outputs = self.out_proj(self.ts_proj(llm_out.transpose(1, 2)).transpose(1, 2))

        if self.rev_in:
            outputs = self.revin_layer(outputs, "denorm")

        if self.history_weight > 0:
            outputs += self.history_weight * prior_y

        return outputs


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        config = args
        self.prompt_weight = config.prompt_weight
        self.history_weight = config.history_weight
        self.d_llm = config.llm_dim
        self.pred_len = config.pred_len
        self.text_random = config.text_random
        self.text_random_order = config.text_random_order
        self.text_embedding_dim = config.text_emb
        self.pool_type = config.pool_type
        self.use_fullmodel = config.use_fullmodel

    def _build_model(self):
        self.args.task_name = "embedding"
        ts_encoder_model = self.model_dict[self.args.model].Model(self.args, mode="encoder").float()
        if self.args.d_layers > 0:
            ts_decoder_model = (
                self.model_dict[self.args.model].Model(self.args, mode="decoder").float()
            )
        else:
            ts_decoder_model = None

        model = LLMTSLateFusion(self.args, ts_encoder_model, ts_decoder_model).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        lr_scheduler = get_lr_scheduler(
            model_optim,
            self.args.train_epochs,
        )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            print(f"Learning rate {model_optim.param_groups[0]['lr']}")

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1

                batch_text = train_data.get_text(index)

                if self.args.text_ts:
                    batch_text = prepare_prompt(batch_text, batch_x, text_ts=self.args.text_ts)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.text_random_order:
                    np.random.shuffle(batch_text)

                model_optim.zero_grad()

                outputs = self.model(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    batch_text,
                    train_data.get_prior_y(index) if self.history_weight > 0 else None,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr_scheduler.step()

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_text = vali_data.get_text(index)

                if self.args.text_ts:
                    batch_text = prepare_prompt(batch_text, batch_x, text_ts=self.args.text_ts)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    batch_text,
                    vali_data.get_prior_y(index) if self.history_weight > 0 else None,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true) * len(batch_x)

                total_loss.append(loss.item())

        total_samples = len(vali_loader.dataset)
        if vali_loader.drop_last:
            total_samples -= len(vali_loader.dataset) % vali_loader.batch_size
        total_loss = np.sum(total_loss) / total_samples

        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_text = test_data.get_text(index)

                if self.args.text_ts:
                    batch_text = prepare_prompt(batch_text, batch_x, text_ts=self.args.text_ts)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    batch_text,
                    test_data.get_prior_y(index) if self.history_weight > 0 else None,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        print("test shape:", preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}")
        f = open(self.args.save_name, "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(mse, mae, rmse, mape, mspe))
        f.write("\n")
        f.write("\n")
        f.close()

        # result save
        # folder_path = "./results/" + setting + "/"
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + "pred.npy", preds)
        # np.save(folder_path + "true.npy", trues)

        return mse
