import pickle

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
import zstandard
from torch import nn

from temporal_nmf.model import TemporalNMF


def get_daily_interactions(interaction_path, max_days=np.inf):
    imls_log = pickle.load(open(interaction_path, "rb"))

    max_days = min(len(imls_log), max_days)

    imls_log_daily = np.zeros(
        (max_days, imls_log.shape[-1], imls_log.shape[-1], 1), dtype=np.float16,
    )

    for day in tqdm.trange(max_days):
        imls_log_daily[day, :, :, 0] = np.log1p(imls_log[day].sum(axis=0))

    return imls_log_daily


def parse_alive_data(alive_path):
    alive_df = pd.read_parquet(alive_path)
    alive_bees = sorted(alive_df.bee_id.values)

    id_to_idx = {bee_id: idx for (idx, bee_id) in enumerate(alive_bees)}
    idx_to_id = {idx: bee_id for (idx, bee_id) in enumerate(alive_bees)}

    alive_df.set_index("bee_id", inplace=True)

    return alive_df, id_to_idx, idx_to_id


def parse_bee_ages(alive_df, from_date_incl, num_days, num_entities):
    bee_ages = np.ones((num_days, num_entities), dtype=np.float) * -1

    for day in range(0, num_days):
        bee_ages[day] = (from_date_incl - alive_df.sort_index().date_emerged).apply(
            lambda td: int(td.days) + day
        )
        has_died = np.where(bee_ages[day] > alive_df.sort_index().days_alive)
        bee_ages[np.ix_([day], has_died[0])] = -1

    valid_ages = bee_ages >= 0
    bee_ages = np.clip(bee_ages, 0, np.inf)

    return bee_ages.astype(np.float32), valid_ages


def parse_labels(alive_df):
    doy_to_label = {v: k for k, v in enumerate(alive_df.doy_emerged.unique())}
    labels = alive_df.doy_emerged.apply(lambda doy: doy_to_label[doy]).values

    return labels


class DataWrapper:
    def __init__(
        self, interaction_path, alive_path, from_day_incl,
    ):
        self.parse_interaction_data(interaction_path, alive_path)
        self.bee_ages, self.valid_ages = parse_bee_ages(
            self.alive_df, from_day_incl, self.num_days, self.num_entities
        )
        self.labels = parse_labels(self.alive_df)

        self.imls_log_daily = (self.valid_ages[:, :, None] * self.valid_ages[:, None, :])[
            :, :, :, None
        ].astype(np.float16) * self.imls_log_daily

    def parse_interaction_data(self, interaction_path, alive_path, max_days=np.inf):
        self.imls_log_daily = get_daily_interactions(interaction_path, max_days=max_days)
        self.num_days = self.imls_log_daily.shape[0]
        self.num_entities = self.imls_log_daily.shape[1]
        self.num_matrices = self.imls_log_daily.shape[-1]

        alive_df, id_to_idx, idx_to_id = parse_alive_data(alive_path)
        self.alive_df = alive_df
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id

        self.num_classes = len(alive_df.doy_emerged.unique())

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

        factor_corr_loss, _, _ = self.tnmf.factor_correlation_loss(self.device)

        batch_losses = {
            "reconstruction_by_age": (self.loss(batch_targets, rec_by_age) * loss_mask).sum()
            / loss_mask.sum(),
            "reconstruction_by_emb": (self.loss(batch_targets, rec_by_emb) * loss_mask).sum()
            / loss_mask.sum(),
            "factor_offsets_l2": self.tnmf.offset_l2_loss(factor_offsets),
            "factor_l1": (self.tnmf.factor_l1_loss(factors_by_age)),
            "embedding_orthogonality": self.tnmf.ortho_loss(),
            "factor_nonnegativity": self.tnmf.nonnegativity_loss(factors_by_age, factors_by_emb),
            "adversarial": -batch_disc_loss.mean(),
            "factor_correlation": factor_corr_loss,
            "basis_function_l1": self.tnmf.basis_function_l1_loss(),
            "embedding_l2": self.tnmf.embedding_l2_loss(),
            "embedding_l1": self.tnmf.embedding_l1_loss(),
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
