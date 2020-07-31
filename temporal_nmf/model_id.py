import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import trange


def _default_age_embedder(num_hidden, num_factors):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Linear(3, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.utils.weight_norm(nn.Linear(num_hidden, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.utils.weight_norm(nn.Linear(num_hidden, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.utils.weight_norm(nn.Linear(num_hidden, num_hidden)),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_factors),
        nn.Softplus(),
    )


def _default_discriminator(num_embeddings, num_hidden, num_classes):
    return nn.Sequential(
        nn.Linear(num_embeddings, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_hidden),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_classes),
    )


def _default_offsetter(num_embeddings, num_hidden, num_factors):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Linear(3, num_hidden, bias=False)),
        nn.LeakyReLU(0.3),
        nn.utils.weight_norm(nn.Linear(num_hidden, num_hidden, bias=False)),
        nn.LeakyReLU(0.3),
        nn.utils.weight_norm(nn.Linear(num_hidden, num_hidden, bias=False)),
        nn.LeakyReLU(0.3),
        nn.utils.weight_norm(nn.Linear(num_hidden, num_hidden, bias=False)),
        nn.LeakyReLU(0.3),
        nn.Linear(num_hidden, num_factors * num_embeddings, bias=False),
    )


class TemporalNMF(nn.Module):
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
        symmetric=True,
        age_embedder=None,
        discriminator=None,
        offsetter=None,
        max_norm_embedding=1,
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
        self.num_timesteps = num_timesteps

        self.age_embedder = (
            _default_age_embedder(num_hidden, num_factors) if age_embedder is None else age_embedder
        )
        self.discriminator = (
            _default_discriminator(num_embeddings, num_hidden, self.num_classes)
            if discriminator is None
            else discriminator
        )
        self.offsetter = (
            _default_offsetter(num_embeddings, num_hidden, num_factors)
            if offsetter is None
            else offsetter
        )

        self.embeddings = nn.Embedding(
            num_entities, num_embeddings, max_norm=max_norm_embedding, norm_type=1
        )
        nn.init.orthogonal_(self.embeddings.weight, gain=10)

        self.daily_bias = nn.Parameter(torch.zeros(num_days, num_matrices))

        """
        if symmetric:
            self.output_map = nn.Parameter(torch.ones(self.num_factors, num_matrices))
        else:
            self.output_map = nn.Parameter(torch.ones(self.num_factors // 2, num_matrices))
        """
        self.output_map = torch.ones(self.num_factors, num_matrices)

        self.ages = torch.from_numpy(ages)
        self.std_age = torch.std(self.ages[self.ages > 0])
        self.mean_age = torch.mean(self.ages[self.ages > 0])

        self.modules = nn.ModuleList((self.age_embedder, self.discriminator, self.offsetter))

    def get_device(self):
        return next(self.parameters()).device

    def get_model_parameters(self):
        return [params for name, params in self.named_parameters() if "discriminator" not in name]

    def get_discriminator_parameters(self):
        return [params for name, params in self.named_parameters() if "discriminator" in name]

    def get_age_factors(self, idxs):
        batch_size = len(idxs)
        ages = (self.ages[:, idxs] - self.mean_age) / self.std_age

        ages = ages[:, None, :, None].repeat(1, self.num_timesteps, 1, 1)
        timesteps = torch.linspace(0, 2 * np.pi, steps=self.num_timesteps + 1)[
            None, :-1, None, None
        ].repeat(self.num_days, 1, batch_size, 1)
        timeoffsets = torch.linspace(0, 1, steps=self.num_timesteps)[None, :, None, None].repeat(
            self.num_days, 1, batch_size, 1
        )
        ages = ages + timeoffsets / self.std_age

        age_emb_input = torch.cat((ages, torch.sin(timesteps), torch.cos(timesteps)), dim=-1)

        factors_by_age = self.age_embedder(
            age_emb_input.pin_memory().to(self.get_device(), non_blocking=True)
        )

        return factors_by_age

    def get_basis_functions(self, ages):
        num_days = ages.shape[0]
        batch_size = ages.shape[1]

        ages = ages[:, None, :, None].repeat(1, self.num_timesteps, 1, 1)
        timesteps = torch.linspace(0, 2 * np.pi, steps=self.num_timesteps + 1)[
            None, :-1, None, None
        ].repeat(num_days, 1, batch_size, 1)
        timesteps = timesteps.pin_memory().to(self.get_device(), non_blocking=True)
        timeoffsets = (
            torch.linspace(0, 1, steps=self.num_timesteps)[None, :, None, None]
            .repeat(num_days, 1, batch_size, 1)
            .pin_memory()
            .to(self.get_device(), non_blocking=True)
        )
        ages = ages + timeoffsets / self.std_age

        offsetter_input = torch.cat((ages, torch.sin(timesteps), torch.cos(timesteps)), dim=-1)

        basis_functions = self.offsetter(offsetter_input)
        basis_functions = basis_functions.view(
            num_days, self.num_timesteps, batch_size, self.num_embeddings, self.num_factors
        )

        return basis_functions

    def get_embedding_factor_offsets(self, idxs):
        device = self.get_device()

        ages = (self.ages[:, idxs] - self.mean_age) / self.std_age
        ages = ages.pin_memory().to(device, non_blocking=True)

        embs = self.embeddings(idxs.pin_memory().to(device, non_blocking=True)).abs()

        basis_functions = self.get_basis_functions(ages)
        offsets = basis_functions * embs[None, None, :, :, None].repeat(
            self.num_days, self.num_timesteps, 1, 1, 1
        )
        offsets = offsets.sum(dim=-2)

        return embs, offsets

    def get_discriminator_output(self, embs):
        logits = self.discriminator(embs)

        return logits

    def reconstruction(self, factors):
        output_map_reparam = nn.functional.softplus(self.output_map)
        output_map_reparam = output_map_reparam / output_map_reparam.sum(dim=0) * self.num_factors

        factors = nn.functional.relu(factors)

        if self.symmetric:
            rec_unmapped = factors[:, :, :, None] * factors[:, :, None, :]
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
                        + self.daily_bias[day, None, None, :].repeat(
                            self.num_entities, self.num_entities, 1
                        )
                    )
                    .cpu()
                    .half()
                )

            recs = torch.cat(recs, dim=0)

        return recs

    @staticmethod
    def nonnegativity_loss(factors_by_age, factors_by_emb):
        return (
            nn.functional.relu(-factors_by_emb).mean() + nn.functional.relu(-factors_by_age).mean()
        )

    @staticmethod
    def factor_l1_loss(factors):
        return factors.abs().sum(dim=-1).mean()

    def basis_function_l1_loss(self, min_age=0, max_age=60, num_age_samples=100):
        ages = (
            torch.linspace(min_age, max_age, steps=num_age_samples, device=self.get_device())
            - self.mean_age
        ) / self.std_age
        basis_functions = self.get_basis_functions(ages[:, None])

        return basis_functions.abs().sum(dim=-2).mean()

    def embedding_l1_loss(self):
        return torch.mean((self.embeddings.weight).abs().sum(dim=-1))

    def embedding_sparsity_loss(self):
        return (
            1
            - self.embeddings.weight.abs().max(dim=1).values
            / self.embeddings.weight.abs().sum(dim=1)
        ).mean()

    def forward(self, idxs):
        batch_size = len(idxs)

        daily_bias_reshaped = self.daily_bias[:, None, None, None, :].repeat(
            1, self.num_timesteps, batch_size, batch_size, 1
        )

        factors_by_age = self.get_age_factors(idxs)
        rec_by_age = self.reconstruction(factors_by_age) + daily_bias_reshaped

        embs, factor_offsets = self.get_embedding_factor_offsets(idxs)
        factors_by_emb = factors_by_age + factor_offsets
        rec_by_emb = self.reconstruction(factors_by_emb) + daily_bias_reshaped

        return rec_by_age, rec_by_emb, factors_by_age, factors_by_emb, factor_offsets, embs

    def get_factor_df(self, ids=None, embedding_dim=2, batch_size=128, valid_ages=None):
        if embedding_dim is not None:
            if embedding_dim == self.num_embeddings:
                embs_reduced = np.abs(self.embeddings.weight.data.cpu().numpy())
            else:
                from umap import UMAP

                embs_reduced = UMAP(n_components=embedding_dim).fit_transform(
                    np.abs(self.embeddings.weight.data.cpu().numpy())
                )

        with torch.no_grad():
            idx = 0
            dfs = []

            while idx < self.num_entities:
                batch_idxs = torch.arange(idx, min((idx + batch_size, self.num_entities)))
                (_, _, _, factors_by_emb, _, _,) = self.forward(batch_idxs)

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

                if valid_ages is not None:
                    columns = ["valid_age"] + columns
                    valid_flat = valid_ages[:, batch_idxs].flatten()
                    df_data = np.concatenate((valid_flat[:, None], df_data), axis=-1)

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
