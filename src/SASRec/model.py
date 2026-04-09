import torch
import torch.nn as nn
import numpy as np
import os

def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(nn.Module):
    def __init__(self, item_num, config):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, config['hidden_units'], padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)

        for _ in range(config['num_blocks']):
            self.attention_layernorms.append(nn.LayerNorm(config['hidden_units'], eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(config['hidden_units'], config['num_heads'], config['dropout_rate']))
            self.forward_layernorms.append(nn.LayerNorm(config['hidden_units'], eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(config['hidden_units'], config['dropout_rate']))

        fix_torch_seed(config['manual_seed'])
        self.initialize()

    def initialize(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs):
        log_feats = self.log2feats(log_seqs)
        logits = torch.matmul(log_feats, self.item_emb.weight.t())
        return logits

    def score(self, seq):
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
    

def save_sasrec_model(model, config, data_description, data_index, filepath):
    """Сохраняет веса модели, конфиг и метаинформацию."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'data_description': data_description,
        'data_index': data_index,          # для декодирования при инференсе
        'pad_token': model.pad_token,
        'item_num': model.item_num
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_sasrec_model(filepath, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model = SASRec(checkpoint['item_num'], checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['config'], checkpoint['data_description'], checkpoint['data_index']
# def load_sasrec_model(filepath, device='cpu'):
#     checkpoint = torch.load(filepath, map_location=device)
#     model = SASRec(checkpoint['item_num'], checkpoint['config'])
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     return model, checkpoint['config'], checkpoint['data_description'], checkpoint['data_index']

def get_model_path(filename):
    """Возвращает полный путь к файлу модели внутри папки saved_models."""
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)

def generate_model_name(config, suffix='best'):
    """Генерирует имя файла на основе гиперпараметров."""
    # Ключевые параметры, влияющие на архитектуру/обучение
    parts = [
        f"e{config['num_epochs']}",
        f"ml{config['maxlen']}",
        f"hid{config['hidden_units']}",
        f"b{config['batch_size']}",
        f"lr{config['learning_rate']}",
        f"{suffix}"
    ]
    return "sasrec_" + "_".join(parts) + ".pt"

def get_latest_model_path():
    """Возвращает путь к самому свежему .pt файлу в папке saved_models."""
    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory '{model_dir}' does not exist.")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No .pt files found in '{model_dir}'.")
    latest = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    return os.path.join(model_dir, latest)