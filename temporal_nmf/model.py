import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import trange


def _default_age_embedder(num_hidden, num_factors):
    return nn.Sequential(
        nn.Linear(1, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_factors),
    )


def _default_discriminator(num_embeddings, num_hidden, num_classes):
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(num_embeddings, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.utils.spectral_norm(nn.Linear(num_hidden, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.utils.spectral_norm(nn.Linear(num_hidden, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.utils.spectral_norm(nn.Linear(num_hidden, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_classes),
    )


def _default_offsetter(num_embeddings, num_hidden, num_factors):
    return nn.Sequential(
        nn.Linear(1 + num_embeddings, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_factors),
    )


def _default_time_embedder(num_factors, num_hidden):
    return nn.Sequential(
        nn.Linear(num_factors + 2, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_factors),
    )


class TemporalNMF(nn.Module):
    def __init__(
        self,
        num_entities,
        num_embeddings,
        num_factors,
        num_hidden,
        num_days,
        num_matrices,
        num_classes,
        ages,
        symmetric=True,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_hidden = num_hidden
        self.num_factors = num_factors

        self.symmetric = symmetric
        if not symmetric:
            self.num_factors *= 2

        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.num_days = num_days

        self.age_embedder = _default_age_embedder(num_hidden, num_factors)
        self.discriminator = _default_discriminator(num_embeddings, num_hidden, self.num_classes)
        self.offsetter = _default_offsetter(num_embeddings, num_hidden, num_factors)

        self.embeddings = nn.Embedding(num_entities, num_embeddings, max_norm=1)
        nn.init.orthogonal_(self.embeddings.weight.data)

        # TODO: data-based init
        self.daily_bias = nn.Parameter(torch.zeros(num_days, num_matrices))

        if symmetric:
            self.output_map = nn.Parameter(torch.ones(self.num_factors, num_matrices))
        else:
            self.output_map = nn.Parameter(torch.ones(self.num_factors // 2, num_matrices))

        self.ages = torch.from_numpy(ages)
        self.std_age = torch.std(self.ages)

        self.modules = nn.ModuleList((self.age_embedder, self.discriminator, self.offsetter))

    def get_device(self):
        return next(self.parameters()).device

    def get_model_parameters(self):
        return [params for name, params in self.named_parameters() if "discriminator" not in name]

    def get_discriminator_parameters(self):
        return [params for name, params in self.named_parameters() if "discriminator" in name]

    def get_age_factors(self, idxs):
        batch_size = len(idxs)
        ages = self.ages[:, idxs] / self.std_age

        factors_by_age = self.age_embedder(
            torch.clamp_min(ages.view(-1, 1), 0)
            .pin_memory()
            .to(self.get_device(), non_blocking=True)
        )
        factors_by_age = factors_by_age.view(self.num_days, batch_size, self.num_factors)

        return factors_by_age

    def get_embedding_factor_offsets(self, idxs):
        device = self.get_device()

        batch_size = len(idxs)
        ages = self.ages[:, idxs] / self.std_age

        embs = self.embeddings(idxs.pin_memory().to(device, non_blocking=True))
        offsets = self.offsetter(
            torch.cat(
                (
                    embs[None, :, :].repeat(self.num_days, 1, 1),
                    torch.clamp_min(ages[:, :, None], 0).pin_memory().to(device, non_blocking=True),
                ),
                dim=-1,
            ).view(-1, self.num_embeddings + 1)
        ).view(self.num_days, batch_size, self.num_factors)

        return embs, offsets

    def get_discriminator_output(self, embs):
        logits = self.discriminator(embs)

        return logits

    def reconstruction(self, factors):
        output_map_reparam = nn.functional.softplus(self.output_map)
        output_map_reparam = output_map_reparam / output_map_reparam.sum(dim=0) * self.num_factors

        factors = nn.functional.relu(factors)

        if self.symmetric:
            rec_unmapped = factors[:, :, None] * factors[:, None, :]
        else:
            rec_unmapped = (
                factors[:, : self.num_factors // 2, None]
                * factors[:, None, self.num_factors // 2 :]
            )

        rec_mapped = (
            rec_unmapped[:, :, :, None, :]
            @ output_map_reparam[None, None, None, :, :].to(rec_unmapped.device)
        )[:, :, :, 0, :]

        return rec_mapped

    def reconstruct_inputs(self, with_offsets=True):
        with torch.no_grad():
            daily_bias_reshaped = self.daily_bias[:, None, None, :].repeat(
                1, self.num_entities, self.num_entities, 1
            )

            all_idxs = torch.LongTensor(list(range(self.num_entities)))

            factors_by_age = self.get_age_factors(all_idxs)

            if with_offsets:
                _, factor_offsets = self.get_embedding_factor_offsets(all_idxs)
                factors_by_emb = factors_by_age + factor_offsets
            else:
                factors_by_emb = factors_by_age

            recs = []
            for day in trange(self.num_days):
                recs.append(
                    (
                        self.reconstruction(factors_by_emb[day][None, :, :])
                        + daily_bias_reshaped[day]
                    ).cpu()
                )

            recs = torch.cat(recs, dim=0)

        return recs

    def ortho_loss(self):
        return torch.sqrt(
            torch.mean(
                torch.pow(
                    (self.embeddings.weight.transpose(0, 1) @ self.embeddings.weight)
                    - (torch.eye(self.num_embeddings, dtype=torch.float32)).to(
                        self.get_device(), non_blocking=True
                    ),
                    2,
                )
            )
        )

    @staticmethod
    def nonnegativity_loss(factors_by_age, factors_by_emb):
        return (
            nn.functional.relu(-factors_by_emb).mean() + nn.functional.relu(-factors_by_age).mean()
        )

    @staticmethod
    def offset_l2_loss(factor_offsets):
        return torch.sqrt(torch.mean(torch.pow(factor_offsets.mean(dim=1), 2)))

    @staticmethod
    def factor_l1_loss(factors):
        return factors.abs().mean()

    def forward(self, idxs):
        batch_size = len(idxs)

        daily_bias_reshaped = self.daily_bias[:, None, None, :].repeat(1, batch_size, batch_size, 1)

        factors_by_age = self.get_age_factors(idxs)
        rec_by_age = self.reconstruction(factors_by_age) + daily_bias_reshaped

        embs, factor_offsets = self.get_embedding_factor_offsets(idxs)
        factors_by_emb = factors_by_age + factor_offsets
        rec_by_emb = self.reconstruction(factors_by_emb) + daily_bias_reshaped

        return rec_by_age, rec_by_emb, factors_by_age, factors_by_emb, factor_offsets, embs

    def get_factor_df(self, ids=None, embedding_dim=2, batch_size=128):
        if embedding_dim is not None:
            if embedding_dim == self.num_embeddings:
                embs_reduced = self.embeddings.weight.data.cpu().numpy()
            else:
                from umap import UMAP

                embs_reduced = UMAP(n_components=embedding_dim).fit_transform(
                    self.embeddings.weight.data.cpu().numpy()
                )

        with torch.no_grad():
            idx = 0
            dfs = []

            while idx < self.num_entities:
                batch_idxs = torch.arange(idx, min((idx + batch_size, self.num_entities)))
                (
                    rec_by_age,
                    rec_by_emb,
                    factors_by_age,
                    factors_by_emb,
                    factor_offsets,
                    embs,
                ) = self.forward(batch_idxs)

                idx += batch_size

                bee_ages_flat = self.ages[:, batch_idxs].numpy().flatten()
                factors_flat = factors_by_emb.data.cpu().numpy().reshape(-1, self.num_factors)
                day_flat = np.tile(
                    np.arange(self.num_days)[:, None], (1, len(batch_idxs))
                ).flatten()
                columns = ["age", "day"] + [f"f_{f}" for f in range(self.num_factors)]
                df_data = np.concatenate(
                    (bee_ages_flat[:, None], day_flat[:, None], factors_flat), axis=-1
                )

                if ids is not None:
                    columns = ["bee_id"] + columns
                    ids_flat = np.tile(ids[batch_idxs][None, :], (self.num_days, 1)).flatten()
                    df_data = np.concatenate((ids_flat[:, None], df_data), axis=-1)

                if embedding_dim is not None:
                    columns += [f"e_{f}" for f in range(embedding_dim)]
                    embs_flat = np.tile(
                        embs_reduced[batch_idxs][None, :], (self.num_days, 1)
                    ).reshape(-1, embedding_dim)
                    df_data = np.concatenate((df_data, embs_flat), axis=-1)

                factor_df = pd.DataFrame(df_data, columns=columns)
                dfs.append(factor_df)

            factor_df = pd.concat(dfs)

        factor_df.reset_index(inplace=True, drop=True)
        factor_df = factor_df[factor_df.age >= 0]

        return factor_df


class TemporalNMFWithTime(nn.Module):
    def __init__(
        self,
        num_entities,
        num_embeddings,
        num_factors,
        num_hidden,
        num_days,
        num_timesteps,
        num_matrices,
        num_classes,
        ages,
        fix_output_map=False,
        symmetric=True,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_hidden = num_hidden

        self.symmetric = symmetric
        if not symmetric:
            num_factors *= 2

        self.num_factors = num_factors
        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.num_days = num_days
        self.num_timesteps = num_timesteps
        self.num_matrices = num_matrices

        self.age_embedder = _default_age_embedder(num_hidden, num_factors)
        self.discriminator = _default_discriminator(num_embeddings, num_hidden, self.num_classes)
        self.offsetter = _default_offsetter(num_embeddings, num_hidden, num_factors)
        self.time_embedder = _default_time_embedder(num_factors, num_hidden)

        self.embeddings = nn.Embedding(num_entities, num_embeddings, max_norm=1)
        nn.init.orthogonal_(self.embeddings.weight.data)

        # TODO: data-based init
        self.daily_bias = nn.Parameter(torch.zeros(num_days, num_matrices))

        num_output_factors = (
            self.num_factors
        )  # self.num_factors if symmetric else self.num_factors // 2

        if fix_output_map:
            self.output_map = torch.ones(num_matrices, num_output_factors)
        else:
            self.output_map = nn.Parameter(torch.ones(num_matrices, num_output_factors))

        self.ages = torch.from_numpy(ages)
        self.std_age = torch.std(self.ages)

        self.modules = nn.ModuleList((self.age_embedder, self.discriminator, self.offsetter))

    def get_device(self):
        return next(self.parameters()).device

    def get_model_parameters(self):
        return [params for name, params in self.named_parameters() if "discriminator" not in name]

    def get_discriminator_parameters(self):
        return [params for name, params in self.named_parameters() if "discriminator" in name]

    def get_age_factors(self, idxs):
        batch_size = len(idxs)
        ages = self.ages[:, idxs] / self.std_age

        factors_by_age = self.age_embedder(
            torch.clamp_min(ages.view(-1, 1), 0).to(self.get_device())
        )
        factors_by_age = factors_by_age.view(self.num_days, batch_size, self.num_factors)

        return factors_by_age

    def get_embedding_factor_offsets(self, idxs):
        device = self.get_device()

        batch_size = len(idxs)
        ages = self.ages[:, idxs] / self.std_age

        embs = self.embeddings(idxs.to(device))
        offsets = self.offsetter(
            torch.cat(
                (
                    embs[None, :, :].repeat(self.num_days, 1, 1),
                    torch.clamp_min(ages[:, :, None], 0).to(device),
                ),
                dim=-1,
            ).view(-1, self.num_embeddings + 1)
        ).view(self.num_days, batch_size, self.num_factors)

        return embs, offsets

    def get_discriminator_output(self, embs):
        logits = self.discriminator(embs)

        return logits

    def reconstruction(self, factors):
        output_map_reparam = nn.functional.softplus(self.output_map)
        output_map_reparam = output_map_reparam / output_map_reparam.sum(dim=0) * self.num_factors
        output_map_reparam = output_map_reparam.to(factors.device)

        factors = (
            nn.functional.relu(factors[:, :, :, None, :])
            * output_map_reparam[None, None, None, :, :]
        )
        factors = torch.stack(factors.split(self.num_matrices, dim=-1), dim=-1)[:, :, :, 0, :, :]

        if self.symmetric:
            rec_mapped = (factors[:, :, :, None] * factors[:, :, None]).sum(dim=-1)
        else:
            rec_mapped = (
                factors[:, :, :, None, :, self.num_factors // 2 :]
                * factors[:, :, None, :, :, self.num_factors // 2 :]
            ).sum(dim=-1)

        return rec_mapped

    def reconstruct_inputs(self, with_offsets=True):
        with torch.no_grad():
            daily_bias_reshaped = self.daily_bias[:, None, None, :].repeat(
                1, self.num_entities, self.num_entities, 1
            )

            all_idxs = torch.LongTensor(list(range(self.num_entities)))

            factors_by_age = self.get_age_factors(all_idxs)

            if with_offsets:
                _, factor_offsets = self.get_embedding_factor_offsets(all_idxs)
                factors_by_emb = factors_by_age + factor_offsets
            else:
                factors_by_emb = factors_by_age

            recs = []
            for day in trange(self.num_days):
                recs.append(
                    (
                        self.reconstruction(factors_by_emb[day][None, :, :])
                        + daily_bias_reshaped[day]
                    ).cpu()
                )

            recs = torch.cat(recs, dim=0)

        return recs

    def ortho_loss(self):
        return torch.sqrt(
            torch.mean(
                torch.pow(
                    (self.embeddings.weight.transpose(0, 1) @ self.embeddings.weight)
                    - (torch.eye(self.num_embeddings, dtype=torch.float32)).to(self.get_device()),
                    2,
                )
            )
        )

    @staticmethod
    def nonnegativity_loss(factors_by_age, factors_by_emb):
        return (
            nn.functional.relu(-factors_by_emb).mean() + nn.functional.relu(-factors_by_age).mean()
        )

    @staticmethod
    def offset_l2_loss(factor_offsets):
        return torch.sqrt(torch.mean(torch.pow(factor_offsets.mean(dim=1), 2)))

    @staticmethod
    def factor_l1_loss(factors):
        return factors.abs().mean()

    def timeembed_factors(self, factors):
        times = torch.arange(self.num_timesteps, dtype=torch.float32) / (self.num_timesteps - 1)
        times *= 2 * np.pi
        times = torch.stack((np.sin(times), np.cos(times)), dim=1).to(factors.device)

        factors_repeated = factors[:, None, :, :].repeat(1, self.num_timesteps, 1, 1)

        factors_with_time = torch.cat(
            (
                factors_repeated,
                times[None, :, None, :].repeat(self.num_days, 1, factors.shape[1], 1),
            ),
            dim=-1,
        )

        offsets = self.time_embedder(factors_with_time)
        factors_timeembedded = factors_repeated + offsets

        return factors_timeembedded, offsets

    def forward(self, idxs):
        batch_size = len(idxs)

        daily_bias_reshaped = self.daily_bias[:, None, None, None, :].repeat(
            1, self.num_timesteps, batch_size, batch_size, 1
        )

        factors_by_age = self.get_age_factors(idxs)
        factors_by_age_timeembedded, factors_by_age_timeembedded_offsets = self.timeembed_factors(
            factors_by_age
        )

        rec_by_age = self.reconstruction(factors_by_age_timeembedded) + daily_bias_reshaped

        embs, factor_offsets = self.get_embedding_factor_offsets(idxs)
        factors_by_emb = factors_by_age + factor_offsets
        factors_by_emb_timeembedded, factors_by_emb_timeembedded_offsets = self.timeembed_factors(
            factors_by_emb
        )

        rec_by_emb = self.reconstruction(factors_by_emb_timeembedded) + daily_bias_reshaped

        return (
            rec_by_age,
            rec_by_emb,
            factors_by_age,
            factors_by_emb,
            factors_by_age_timeembedded,
            factors_by_emb_timeembedded,
            factors_by_age_timeembedded_offsets,
            factors_by_emb_timeembedded_offsets,
            factor_offsets,
            embs,
        )

    def get_factor_df(
        self, ids=None, embedding_dim=2, batch_size=128, importance_weighted_embeddings=True
    ):
        if embedding_dim is not None:
            embs_reduced = self.embeddings.weight.data.cpu().numpy()
            embs_reduced = embs_reduced - embs_reduced.mean(axis=0)[None, :]

            if importance_weighted_embeddings:
                importances = self.get_embedding_importances()
                embs_reduced = embs_reduced * importances[None, :]

            if embedding_dim != self.num_embeddings:
                from umap import UMAP

                embs_reduced = UMAP(n_components=embedding_dim, min_dist=0).fit_transform(
                    embs_reduced
                )

        with torch.no_grad():
            idx = 0
            dfs = []

            while idx < self.num_entities:
                batch_idxs = torch.arange(idx, min((idx + batch_size, self.num_entities)))
                (
                    rec_by_age,
                    rec_by_emb,
                    factors_by_age,
                    factors_by_emb,
                    _,
                    _,
                    _,
                    _,
                    factor_offsets,
                    embs,
                ) = self.forward(batch_idxs)

                idx += batch_size

                bee_ages_flat = self.ages[:, batch_idxs].numpy().flatten()
                factors_flat = factors_by_emb.data.cpu().numpy().reshape(-1, self.num_factors)
                day_flat = np.tile(
                    np.arange(self.num_days)[:, None], (1, len(batch_idxs))
                ).flatten()
                columns = ["age", "day"] + [f"f_{f}" for f in range(self.num_factors)]
                df_data = np.concatenate(
                    (bee_ages_flat[:, None], day_flat[:, None], factors_flat), axis=-1
                )

                if ids is not None:
                    columns = ["bee_id"] + columns
                    ids_flat = np.tile(ids[batch_idxs][None, :], (self.num_days, 1)).flatten()
                    df_data = np.concatenate((ids_flat[:, None], df_data), axis=-1)

                if embedding_dim is not None:
                    columns += [f"e_{f}" for f in range(embedding_dim)]
                    embs_flat = np.tile(
                        embs_reduced[batch_idxs][None, :], (self.num_days, 1)
                    ).reshape(-1, embedding_dim)
                    df_data = np.concatenate((df_data, embs_flat), axis=-1)

                factor_df = pd.DataFrame(df_data, columns=columns)
                dfs.append(factor_df)

            factor_df = pd.concat(dfs)

        factor_df.reset_index(inplace=True, drop=True)
        factor_df = factor_df[factor_df.age >= 0]

        return factor_df

    def get_embedding_importances(self, num_distortions=64):
        importances = []

        with torch.no_grad():
            device = self.get_device()
            idxs = torch.arange(0, self.num_entities).to(device)

            ages = self.ages[:, idxs] / self.std_age

            embs_orig = self.embeddings(idxs.to(device))
            embs_repeated = embs_orig[:, None, :].repeat(1, num_distortions, 1)

            offsets_orig = self.offsetter(
                torch.cat(
                    (
                        embs_repeated[None, :, :].repeat(self.num_days, 1, 1, 1),
                        torch.clamp_min(ages[:, :, None, None], 0)
                        .repeat(1, 1, num_distortions, 1)
                        .to(device),
                    ),
                    dim=-1,
                ).view(-1, self.num_embeddings + 1)
            )

            for emb_dim in range(self.num_embeddings):
                distortions = torch.randn_like(embs_repeated) * embs_orig.std()

                embs_distorted = embs_repeated.clone()
                embs_distorted[:, :, emb_dim] = (
                    embs_distorted[:, :, emb_dim] + distortions[:, :, emb_dim]
                )

                offsets_distorted = self.offsetter(
                    torch.cat(
                        (
                            embs_distorted[None, :, :].repeat(self.num_days, 1, 1, 1),
                            torch.clamp_min(ages[:, :, None, None], 0)
                            .repeat(1, 1, num_distortions, 1)
                            .to(device),
                        ),
                        dim=-1,
                    ).view(-1, self.num_embeddings + 1)
                )

                distortion_effect = torch.sqrt(torch.mean((offsets_distorted - offsets_orig) ** 2))
                importances.append(distortion_effect.item())

        return np.array(importances)
