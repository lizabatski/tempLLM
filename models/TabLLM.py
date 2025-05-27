import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

class TabLLM(nn.Module):
    def __init__(self, configs, prompt="Classify using the following tabular features"):
        super(TabLLM, self).__init__()

        # save
        self.num_classes = configs.num_classes
        self.n_vars = configs.n_vars
        self.prompt = prompt
        self.llm_layers = configs.llm_layers


        self.dropout = nn.Dropout(configs.dropout)
        self.task_name = configs.task_name

        # using gpt
        self.gpt2_config = GPT2Config.from_pretrained("gpt2")
        self.gpt2_config.output_attentions = False
        self.gpt2_config.output_hidden_states = False
        self.gpt2_config.n_layer = self.llm_layers  

        self.llm = GPT2Model.from_pretrained("gpt2", config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        if configs.freeze_llm:
            print("Freezing all LLM parameters.")
            for param in self.llm.parameters():
                param.requires_grad = False
        else:
            print("Fine tuning")

       
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding_dim = self.gpt2_config.n_embd  

        # project tabular features to LLM embedding size
        self.feature_proj = nn.Linear(self.n_vars, self.embedding_dim)

       
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.embedding_dim, self.num_classes)
        )

    def forward(self, x):
        """
        x: (batch_size, num_features)
        """
        B, N = x.shape
        assert N == self.n_vars, f"Expected {self.n_vars} features but got {N}"

        # encode prompt
        prompts = [self.prompt] * B
        prompt_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(x.device)
        prompt_embeddings = self.llm.get_input_embeddings()(prompt_tokens.input_ids)

        # Project tabular features to embedding space
        feature_embeddings = self.feature_proj(x).unsqueeze(1)  # (B, 1, emb_dim)

        # concat prompt and feature
        inputs_embeds = torch.cat([prompt_embeddings, feature_embeddings], dim=1)

 
        llm_out = self.llm(inputs_embeds=inputs_embeds).last_hidden_state

  
        feature_token = llm_out[:, -1, :]  # (B, emb_dim)

        # classification
        logits = self.classifier(feature_token)
        return logits
