import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertForSequenceClassification
import abc
from src.utils import load_labels
import numpy as np
from src.data import get_label_map


class BertWrapper(nn.Module):
    def __init__(self, d_model=256, d_out=128, ext_fts_dim=0):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.d_out = self.bert.pooler.dense.out_features

    def forward(self, src, norm_output=True):
        with torch.no_grad():
            mask = self.build_mask(src)
            reps = self.bert(src, attention_mask=mask)[1]
        return reps

    def build_mask(self, src, mask_value=0):
        # Hugging face experts masked values = 0 and non masked values = 1
        mask = (src != mask_value) * 1
        return mask


class BaseEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder.eval()

    def base_encode(self, src):
        src = self.text_encoder(src)
        return src

    @abc.abstractmethod
    def forward(self, *args):
        return


class InputEncoder(BaseEncoder):
    def __init__(self, text_encoder, d_head, d_out, n_head_layers=2, norm_output=False):
        super().__init__(text_encoder)
        self.head = FeedForward(self.text_encoder.d_out, d_out, d_head, n_head_layers)
        self.norm_output = norm_output

    def forward(self, text_src):
        src = self.base_encode(text_src)
        src = self.head(src)
        if self.norm_output:
            src = src / src.norm(dim=1, keepdim=True)
        return src


class LabelEncoder(BaseEncoder):
    def __init__(
        self,
        text_encoder,
        tokenizer,
        d_head,
        d_out,
        n_head_layers=2,
        norm_output=False,
        labels_path="src/config/labels.txt",
    ):
        super().__init__(text_encoder)
        self.labels = load_labels(labels_path)
        self.label_map = get_label_map(self.labels)
        self.label_tokens = []
        self.num_negative_samples = 0
        self.norm_output = norm_output
        self.head = FeedForward(self.text_encoder.d_out, d_out, d_head, n_head_layers)

        for label in self.labels:
            self.label_tokens.append(tokenizer(label))
        self.label_tokens = torch.nn.utils.rnn.pad_sequence(self.label_tokens).T

        assert self.label_tokens.shape[0] == len(self.labels)

        _, self.embedding_dim = self.encode_tokens(self.label_tokens[0].unsqueeze(0)).shape
        self.negative_idxs = torch.empty((32, self.num_negative_samples))
        self.reset_cache()

    def reset_cache(self):
        self._cache = torch.full((len(self.labels), self.embedding_dim), float("nan"))
        self._cache_built_for_inference = False

    def populate_cache_from_batch(self, embeddings, idxs):
        for i, ix in enumerate(idxs):
            if self._cache[ix].isnan().sum() != 0:
                self._cache[ix] = embeddings[i]

    def populate_cache_from_labels(self):
        label_encodings = self.encode_tokens(self.label_tokens).detach()
        self._cache = label_encodings
        self._cache_built_for_inference = True

    def forward(self, idxs, num_negative_samples=2):

        tokens = F.embedding(idxs, self.label_tokens)

        if self.training:
            src = self.encode_tokens(tokens)
            self.populate_cache_from_batch(src, idxs)
            # negative_samples
            negative_samples = self.sample_negatives(num_negative_samples, idxs)
            self.reset_cache()
            return src, negative_samples
        else:
            return self.get_all_label_embeddings()[idxs]

    def get_all_label_embeddings(self):
        if not self._cache_built_for_inference:
            self.populate_cache_from_labels()
        return self._cache

    def encode_tokens(self, src):
        src = self.base_encode(src)
        src = self.head(src)
        if self.norm_output:
            src = src / src.norm(dim=1, keepdim=True)
        return src

    def sample_negatives(self, n, idxs):
        choices = set(idxs.numpy())
        bsz = idxs.shape[0]
        if n != self.num_negative_samples or bsz != self.negative_idxs.shape[0]:
            self.negative_idxs = torch.empty((bsz, n), dtype=int)
        for i, ix in enumerate(idxs):
            filtered_choices = list(filter(lambda x: x != ix, choices))
            self.negative_idxs[i, :] = torch.tensor(
                np.random.choice(filtered_choices, size=n, replace=True)
            )

        negative_samples = F.embedding(self.negative_idxs, self._cache)

        return negative_samples

    def idx_to_text(self, idx):
        return self.labels[idx]

    def text_to_idx(self, text):
        return self.label_map[text.lower().strip()]


class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers):
        super().__init__()
        if n_layers == 1:
            self.layers = [nn.Linear(d_in, d_out)]
        else:
            self.layers = [nn.Linear(d_in, d_hidden)]
            self.layers.extend([nn.Linear(d_hidden, d_hidden) for i in range(n_layers - 2)])
            self.layers.append(nn.Linear(d_hidden, d_out))

    def forward(self, src):
        for layer in self.layers[:-1]:
            src = layer(src)
            src = F.relu(src)
        src = self.layers[-1](src)
        return src


class MarginRankingLoss(nn.Module):
    """
    MarginRankingLoss, see StarSpace Algorithm for more details
    """

    def __init__(self, margin=0.1, sim_measure="inner_product"):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9
        self.sim_measure = "inner_product"

        if self.sim_measure == "inner_product":
            self.sim = self.inner_product_sim
        else:
            print(f"simalirty measure {sim_measure} not supported")

    def inner_product_sim(self, A, B, batch=True):
        """
        Inner prodcut similarity measure for two tensors A & B
        if batch == True then A & B shapes must be bsz, n_samples, feat_dim
        else
        """
        assert len(A.shape) == len(B.shape) == 3, "shapes must be bsz, n_samples, feat_dim"
        assert A.shape[0] == B.shape[0]
        return torch.bmm(A, B.transpose(2, 1))  # bsz x n_samples_a x n_samples_b

    def normalized_inner_product_sim(self, A, B, batch=True):
        return self.inner_product_sim(A / A.norm(keepdims=True), B / B.norm(keepdims=True))

    def forward(self, anchor, positive, negatives):
        bsz, n_neg, ft_dim = negatives.shape
        assert len(positive.shape) == 2  # Only one postive example

        positive_sim = self.sim(anchor.unsqueeze(1), positive.unsqueeze(1)).squeeze()
        negative_sim = self.sim(anchor.unsqueeze(1), negatives).sum(dim=-1).squeeze()
        batch_loss = torch.clamp(self.margin - positive_sim + negative_sim, min=0)
        return batch_loss.mean()
