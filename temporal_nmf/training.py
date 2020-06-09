import pathlib
import pickle

import numpy as np
import pandas as pd
import sparse
import torch
import tqdm.auto as tqdm
import zstandard
from torch import nn

from temporal_nmf.model import TemporalNMF


class DataWrapper:
    def __init__(self, base_path, dataset_name, from_day_incl, dense=True):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.from_day_incl = from_day_incl
        self.dataset_name = dataset_name

        self.imls_log_daily = sparse.load_npz(base_path / f"interactions_{dataset_name}_sparse.npz")

        self.dense = dense
        if self.dense:
            self.imls_log_daily = self.imls_log_daily.todense()

        self.num_days = self.imls_log_daily.shape[0]
        self.num_entities = self.imls_log_daily.shape[1]
        self.num_matrices = self.imls_log_daily.shape[-1]

        self.alive_df = pd.read_csv(
            base_path / f"alive_{dataset_name}.csv", parse_dates=["date_emerged"]
        )
        self.num_classes = len(self.alive_df.date_emerged.unique())

        indices_df = pd.read_csv(base_path / f"indices_{dataset_name}.csv")
        self.id_to_idx = dict(indices_df.values)
        self.idx_to_id = dict(indices_df.values[:, ::-1])

        self.bee_ages, self.valid_ages = self.parse_bee_ages(
            self.alive_df, from_day_incl, self.num_days, self.num_entities
        )

        self.labels = self.parse_labels(self.alive_df)

    @staticmethod
    def parse_bee_ages(alive_df, from_date_incl, num_days, num_entities):
        bee_ages = np.ones((num_days, num_entities), dtype=np.float) * -1
        valid_ages = np.zeros((num_days, num_entities), dtype=np.bool)

        for day in range(0, num_days):
            bee_ages[day] = (from_date_incl - alive_df.sort_index().date_emerged).apply(
                lambda td: int(td.days) + day
            )

            valid_ages[day] = bee_ages[day] >= 0
            has_died = np.where(bee_ages[day] > alive_df.sort_index().days_alive)
            valid_ages[np.ix_([day], has_died[0])] = False

        bee_ages = np.clip(bee_ages, 0, np.inf)

        return bee_ages.astype(np.float32), valid_ages

    @staticmethod
    def parse_labels(alive_df):
        date_to_label = {v: k for k, v in enumerate(alive_df.date_emerged.unique())}
        labels = np.array([date_to_label[d] for d in alive_df.date_emerged.values])

        return labels

    def save(self, path):
        with open(path, "wb") as fh:
            cctx = zstandard.ZstdCompressor()
            with cctx.stream_writer(fh) as compressor:
                self.imls_log_daily = sparse.from_numpy(self.imls_log_daily)
                compressor.write(pickle.dumps(self, protocol=4))

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as decompressor:
                wrapper = pickle.loads(decompressor.read())
                if wrapper.dense:
                    wrapper.imls_log_daily = wrapper.imls_log_daily.todense()
                return wrapper


class TrainingWrapper:
    def __init__(
        self,
        data,
        num_hidden=32,
        num_embeddings=32,
        num_factors=8,
        device=None,
        age_embedder=None,
        offsetter=None,
        discriminator=None,
    ):
        self.data = data

        self.num_hidden = num_hidden
        self.num_embeddings = num_embeddings
        self.num_factors = num_factors

        self.device = device

        self.tnmf = TemporalNMF(
            self.data.num_entities,
            self.num_embeddings,
            self.num_factors,
            self.num_hidden,
            self.data.num_days,
            self.data.num_matrices,
            self.data.num_classes,
            self.data.bee_ages,
            age_embedder=age_embedder,
            discriminator=discriminator,
            offsetter=offsetter,
        ).to(device)

        self.loss = nn.SmoothL1Loss(reduction="none").to(self.device)
        self.disc_loss = nn.CrossEntropyLoss(reduction="none").to(self.device)

    def train_batch(self, batch_size, label_offset=0, use_valid_ages=False):
        batch_idxs = torch.randint(0, self.data.num_entities, (batch_size,))
        batch_targets = torch.from_numpy(
            self.data.imls_log_daily[
                np.ix_(list(range(self.data.num_days)), list(batch_idxs), list(batch_idxs),)
            ]
        )
        batch_targets = batch_targets.pin_memory().to(self.device, non_blocking=True).float()
        disc_targets = torch.from_numpy(self.data.labels[batch_idxs] + label_offset)
        disc_targets = disc_targets.pin_memory().to(self.device, non_blocking=True)

        loss_mask = 1 - torch.eye(batch_size)[None, :, :, None].repeat(
            self.data.num_days, 1, 1, self.data.num_matrices
        )
        loss_mask = loss_mask.pin_memory().to(self.device, non_blocking=True)

        if use_valid_ages:
            valid_ages = self.data.valid_ages[:, batch_idxs]
            is_valid = valid_ages[:, None, :] * valid_ages[:, :, None]
            is_valid = torch.from_numpy(is_valid).pin_memory().to(self.device, non_blocking=True)
            loss_mask *= is_valid[:, :, :, None]
        else:
            is_valid = batch_targets.sum(dim=(2, 3)) > 0
            is_valid = (is_valid[:, :, None] * is_valid[:, None, :])[:, :, :, None].float()
            loss_mask *= is_valid

        (rec_by_age, rec_by_emb, factors_by_age, factors_by_emb, factor_offsets, embs,) = self.tnmf(
            batch_idxs
        )
        disc_logits = self.tnmf.get_discriminator_output(embs)
        batch_disc_loss = self.disc_loss(disc_logits, disc_targets)

        batch_losses = {
            "reconstruction_by_age": (self.loss(batch_targets, rec_by_age) * loss_mask).sum()
            / loss_mask.sum(),
            "reconstruction_by_emb": (self.loss(batch_targets, rec_by_emb) * loss_mask).sum()
            / loss_mask.sum(),
            "factor_l1": (self.tnmf.factor_l1_loss(factors_by_age)),
            "factor_nonnegativity": self.tnmf.nonnegativity_loss(factors_by_age, factors_by_emb),
            "adversarial": -batch_disc_loss.mean(),
            "basis_function_l1": self.tnmf.basis_function_l1_loss(),
            "embedding_sparsity": self.tnmf.embedding_sparsity_loss()
            * self.tnmf.embedding_l1_loss(),
        }

        return batch_losses, batch_disc_loss, batch_idxs, embs, disc_targets

    def save(self, path):
        with open(path, "wb") as fh:
            cctx = zstandard.ZstdCompressor()
            with cctx.stream_writer(fh) as compressor:
                compressor.write(pickle.dumps(self, protocol=4))

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as decompressor:
                return pickle.loads(decompressor.read())
