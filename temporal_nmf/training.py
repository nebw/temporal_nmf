import pathlib
import pickle

import numpy as np
import pandas as pd
import sparse
import torch
import tqdm.auto as tqdm
import zstandard
from torch import nn

import temporal_nmf.model
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
                self.imls_log_daily = sparse.COO.from_numpy(self.imls_log_daily)
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
        datasets,
        device,
        lambdas,
        num_hidden=32,
        num_embeddings=16,
        num_factors=8,
    ):
        self.datasets = datasets
        self.device = device
        self.lambdas = lambdas
        self.num_hidden = num_hidden
        self.num_embeddings = num_embeddings
        self.num_factors = num_factors
        self.num_classes = sum([d.num_classes for d in datasets])

        self.age_embedder = temporal_nmf.model._default_age_embedder(num_hidden, num_factors).to(
            device
        )
        self.discriminator = temporal_nmf.model._default_discriminator(
            num_embeddings, num_hidden, self.num_classes
        ).to(device)

        self.offsetter = temporal_nmf.model._default_offsetter(
            num_embeddings, num_hidden, num_factors
        ).to(device)

        self.loss = nn.SmoothL1Loss(reduction="none").to(self.device)
        self.disc_loss = nn.CrossEntropyLoss(reduction="none").to(self.device)

        self.models = []
        for data in self.datasets:
            self.models.append(
                TemporalNMF(
                    data.num_entities,
                    self.num_embeddings,
                    self.num_factors,
                    self.num_hidden,
                    data.num_days,
                    data.num_matrices,
                    data.num_classes,
                    data.bee_ages,
                    age_embedder=self.age_embedder,
                    discriminator=self.discriminator,
                    offsetter=self.offsetter,
                    nonnegative=lambdas["factor_nonnegativity"] > 0,
                ).to(device)
            )

        combined_ages = torch.cat(
            [m.ages[d.valid_ages] for d, m in zip(self.datasets, self.models)]
        )
        self.mean_age = combined_ages.mean()
        self.std_age = combined_ages.std()

        for model in self.models:
            model.mean_age = self.mean_age.clone()
            model.std_age = self.std_age.clone()

        self.optim = torch.optim.Adam(
            set().union(*tuple(map(lambda m: m.get_model_parameters(), self.models))),
            amsgrad=True,
        )
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), amsgrad=True)

        self.loss_hist = []

    def subbatch(self, data, model, batch_size, label_offset, use_valid_ages=False):
        batch_idxs = torch.randint(0, data.num_entities, (batch_size,))

        batch_targets = torch.from_numpy(
            data.imls_log_daily[
                np.ix_(
                    list(range(data.num_days)),
                    list(batch_idxs),
                    list(batch_idxs),
                )
            ]
        )
        batch_targets = batch_targets.pin_memory().to(self.device, non_blocking=True).float()

        disc_targets = torch.from_numpy(data.labels[batch_idxs] + label_offset)
        disc_targets = disc_targets.pin_memory().to(self.device, non_blocking=True)

        loss_mask = 1 - torch.eye(batch_size)[None, :, :, None].repeat(
            data.num_days, 1, 1, data.num_matrices
        )
        loss_mask = loss_mask.pin_memory().to(self.device, non_blocking=True)

        if use_valid_ages:
            valid_ages = data.valid_ages[:, batch_idxs]
            is_valid = valid_ages[:, None, :] * valid_ages[:, :, None]
            is_valid = torch.from_numpy(is_valid).pin_memory().to(self.device, non_blocking=True)
            loss_mask *= is_valid[:, :, :, None]
        else:
            is_valid = batch_targets.sum(dim=(2, 3)) > 0
            is_valid = (is_valid[:, :, None] * is_valid[:, None, :])[:, :, :, None].float()
            loss_mask *= is_valid

        (
            rec_by_age,
            rec_by_emb,
            factors_by_age,
            factors_by_emb,
            factor_offsets,
            embs,
        ) = model(batch_idxs)

        batch_losses = {
            "reconstruction_by_age": (self.loss(batch_targets, rec_by_age) * loss_mask).sum()
            / loss_mask.sum(),
            "reconstruction_by_emb": (self.loss(batch_targets, rec_by_emb) * loss_mask).sum()
            / loss_mask.sum(),
            "factor_l1": (model.factor_l1_loss(factors_by_age)),
            "factor_nonnegativity": model.nonnegativity_loss(factors_by_age, factors_by_emb),
            "basis_function_l1": model.basis_function_l1_loss(),
            "embedding_sparsity": model.embedding_l1_loss(),
        }

        batch_losses = {f"{k}_{data.dataset_name}": v for k, v in batch_losses.items()}

        return batch_losses, embs, disc_targets

    def batch(self, batch_size):
        self.optim.zero_grad()
        self.disc_optim.zero_grad()

        batch_losses = dict()
        label_offset = 0
        batch_embs = []
        batch_disc_targets = []
        for (data, model) in zip(self.datasets, self.models):
            subbatch_losses, embs, disc_targets = self.subbatch(
                data, model, batch_size, label_offset
            )
            batch_losses.update(subbatch_losses)
            label_offset += data.num_classes
            batch_embs.append(embs)
            batch_disc_targets.append(disc_targets)

        batch_embs = torch.cat(batch_embs)
        batch_disc_targets = torch.cat(batch_disc_targets)

        disc_logits = self.discriminator(batch_embs)
        disc_loss = self.disc_loss(disc_logits, batch_disc_targets)

        batch_idx_from = 0
        for data in self.datasets:
            batch_losses[f"adversarial_{data.dataset_name}"] = -disc_loss[
                batch_idx_from : batch_idx_from + batch_size
            ].mean()
            batch_idx_from += batch_size

        batch_losses_scaled = []
        for loss_name, loss in batch_losses.items():
            loss_base_name = "_".join(loss_name.split("_")[:-1])

            if loss_base_name in self.lambdas:
                loss_scaled = loss * self.lambdas[loss_base_name]
            else:
                loss_scaled = loss

            batch_losses_scaled.append((loss_name, loss_scaled))
        batch_losses_scaled = dict(batch_losses_scaled)

        combined_loss = sum(batch_losses_scaled.values())

        combined_loss.backward(retain_graph=True)
        self.optim.step()

        disc_loss.mean().backward()
        self.disc_optim.step()

        self.loss_hist.append({k: v.detach().cpu().item() for k, v in batch_losses_scaled.items()})

        return combined_loss.data.cpu().item()

    def save(self, path):
        datasets = self.datasets
        self.datasets = None
        with open(path, "wb") as fh:
            torch.save(self, fh, pickle_protocol=4)
        self.datasets = datasets

    @classmethod
    def load(cls, path, datasets, *args, **kwargs):
        with open(path, "rb") as fh:
            wrapper = torch.load(fh, *args, **kwargs)
        wrapper.datasets = datasets

        return wrapper
