#!/usr/bin/env python3

import uuid

import click
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import sklearn
import sklearn.metrics
import torch
import tqdm.auto as tqdm
from torch import nn

import temporal_nmf
from temporal_nmf.model import TemporalNMF


@click.command()
@click.argument("output_path")
@click.option("--batch_size", default=16)
@click.option("--num_batches", default=100000)
@click.option("--num_hidden", default=32)
@click.option("--data_num_factors", default=3)
@click.option("--data_smoothing_sigma", default=10)
@click.option("--data_num_groups", default=4)
@click.option("--data_factor_scaler", default=0.18)
@click.option("--data_offset_scaler", default=0.1)
@click.option("--data_num_individuals", default=2 ** 10)
@click.option("--data_life_expectancy_mean", default=30)
@click.option("--data_life_expectancy_std", default=10)
@click.option("--data_num_days", default=100)
@click.option("--device", default="cpu")
@click.option("--lambda_factor_l1", default=0.005)
@click.option("--lambda_basis_function_l1", default=0.01)
@click.option("--lambda_embedding_sparsity", default=0.1)
@click.option("--lambda_factor_nonnegativity", default=1)
@click.option("--lambda_adversarial", default=0.1)
@click.option("--min_noise", default=0)
@click.option("--max_noise", default=50)
@click.option("--num_noise_samples", default=50)
def synth(output_path, **kwargs):
    config = kwargs

    num_factors = config["data_num_factors"]
    smoothing_sigma = config["data_smoothing_sigma"]
    num_groups = config["data_num_groups"]
    factor_scaler = config["data_factor_scaler"]
    offset_scaler = config["data_offset_scaler"]

    num_individuals = config["data_num_individuals"]
    life_expectancy_mean = config["data_life_expectancy_mean"]
    life_expectancy_std = config["data_life_expectancy_std"]
    num_days = config["data_num_days"]

    batch_size = config["batch_size"]
    num_batches = config["num_batches"]
    device = config["device"]

    lambdas = {
        "factor_l1": config["lambda_factor_l1"],
        "basis_function_l1": config["lambda_basis_function_l1"],
        "embedding_sparsity": config["lambda_embedding_sparsity"],
        "factor_nonnegativity": config["lambda_factor_nonnegativity"],
        "adversarial": config["lambda_adversarial"],
    }

    min_noise = config["min_noise"]
    max_noise = config["max_noise"]
    num_noise_samples = config["num_noise_samples"]

    ages = np.linspace(0, 100, num=101)

    def generate_trajectories():
        mean_trajectory = scipy.ndimage.gaussian_filter1d(
            np.sqrt(
                np.cumsum(np.random.randn(len(ages), num_factors) * factor_scaler, axis=0) ** 2
            ),
            smoothing_sigma,
            axis=0,
        )
        offsets = scipy.ndimage.gaussian_filter1d(
            np.cumsum(np.random.randn(num_groups, len(ages), num_factors) * offset_scaler, axis=1),
            smoothing_sigma,
            axis=1,
        )
        offsets -= offsets.mean(axis=(1))[:, None, :]
        group_trajectories = np.clip(mean_trajectory[None, :, :] + offsets, 0, np.inf)

        return mean_trajectory, offsets, group_trajectories

    def generate_individuals():
        birthdays = np.clip(
            np.sort(np.random.randint(0, num_days, size=num_individuals)),
            0,
            num_days - life_expectancy_mean,
        )

        lifetimes = np.random.normal(
            loc=group_life_expectancy_mean[group_assigments],
            scale=life_expectancy_std,
            size=num_individuals,
        )
        lifetimes = np.clip(lifetimes, 1, np.inf).astype(np.int)

        deathdays = np.clip(birthdays + lifetimes, 0, num_days)

        individual_factors = np.zeros((num_days, num_individuals, num_factors))
        individual_ages = np.zeros((num_days, num_individuals)) * np.nan
        individual_ages[np.isnan(individual_ages)] = -1.0

        for individual in range(num_individuals):
            group = group_assigments[individual]
            group_trajectory = group_trajectories[group]
            birthday = birthdays[individual]
            deathday = deathdays[individual]
            individual_factors[birthday:deathday, individual] = group_trajectory[
                : np.min((deathday - birthday, len(group_trajectory) - birthday - 1))
            ]
            individual_ages[birthday:deathday, individual] = np.arange(
                np.min((deathday - birthday, len(group_trajectory) - birthday - 1))
            )

        return birthdays, deathdays, individual_factors, individual_ages

    def generate_interactions(noise_std):
        interactions = np.zeros((num_days, num_individuals, num_individuals, 1), dtype=np.float32)

        for day in range(num_days):
            factors = individual_factors[day]
            interactions[day] = (factors @ factors.T)[:, :, None]
            noise = np.random.normal(
                loc=0, scale=noise_std, size=(num_individuals, num_individuals, 1)
            )
            noise *= interactions[day] > 0
            interactions[day] = np.clip(interactions[day] + noise, 0, np.inf)

        return interactions

    def get_model(verbose=False):
        model = TemporalNMF(
            num_individuals,
            num_groups * 4,
            num_factors * 4,
            32,
            num_days,
            1,
            num_days,
            individual_ages.astype(np.float32),
        ).to(device)

        optim = torch.optim.Adam(model.get_model_parameters(), amsgrad=True)
        disc_optim = torch.optim.Adam(model.discriminator.parameters(), amsgrad=True)
        recon_loss = nn.MSELoss(reduction="none").to(device)
        disc_loss = nn.CrossEntropyLoss().to(device)

        loss_hist = []

        for i in range(num_batches):
            optim.zero_grad()
            disc_optim.zero_grad()

            batch_idxs = torch.randint(0, num_individuals, (batch_size,))
            temporal_idxs = torch.LongTensor(list(range(num_days))).to(device)

            batch_targets = torch.from_numpy(
                interactions[
                    np.ix_(
                        list(range(num_days)),
                        list(batch_idxs),
                        list(batch_idxs),
                    )
                ]
            )
            batch_targets = batch_targets.to(device, non_blocking=True).float()

            disc_targets = torch.from_numpy(birthdays[batch_idxs])
            disc_targets = disc_targets.to(device, non_blocking=True)

            loss_mask = 1 - torch.eye(batch_size)[None, :, :, None].repeat(num_days, 1, 1, 1)
            loss_mask = loss_mask.to(device, non_blocking=True)

            is_valid = torch.from_numpy(individual_ages[:, batch_idxs] >= 0).to(
                device, non_blocking=True
            )
            is_valid = (is_valid[:, :, None] * is_valid[:, None, :])[:, :, :, None].float()
            loss_mask *= is_valid

            (
                rec_by_age,
                rec_by_emb,
                factors_by_age,
                factors_by_emb,
                factor_offsets,
                embs,
            ) = model(temporal_idxs, batch_idxs)

            disc_logits = model.discriminator(embs)
            batch_disc_loss = disc_loss(disc_logits, disc_targets)

            batch_losses = {
                "reconstruction_by_age": (recon_loss(batch_targets, rec_by_age) * loss_mask).sum()
                / loss_mask.sum(),
                "reconstruction_by_emb": (recon_loss(batch_targets, rec_by_emb) * loss_mask).sum()
                / loss_mask.sum(),
                "factor_l1": (model.factor_l1_loss(factors_by_age)),
                "factor_nonnegativity": model.nonnegativity_loss(factors_by_age, factors_by_emb),
                "basis_function_l1": model.basis_function_l1_loss(),
                "embedding_sparsity": model.embedding_sparsity_loss(),
                "adversarial": -batch_disc_loss,
            }

            batch_losses_scaled = []
            for loss_name, loss in batch_losses.items():
                if loss_name in lambdas:
                    loss_scaled = loss * lambdas[loss_name]
                else:
                    loss_scaled = loss

                batch_losses_scaled.append((loss_name, loss_scaled))
            batch_losses_scaled = dict(batch_losses_scaled)

            combined_loss = sum(batch_losses_scaled.values())

            combined_loss.backward(retain_graph=True)
            optim.step()

            batch_disc_loss.backward()
            disc_optim.step()

            loss_hist.append({k: v.detach().cpu().item() for k, v in batch_losses_scaled.items()})

            if i % 1000 == 0:
                factor_df = model.get_factor_df(
                    ids=np.arange(num_individuals),
                    embedding_dim=model.num_embeddings,
                    valid_ages=individual_ages >= 0,
                )
                factor_df.age = factor_df.age.astype(int)
                factor_df.day = factor_df.day.astype(int)
                factor_df.bee_id = factor_df.bee_id.astype(int)

                truth_df = pd.DataFrame(
                    np.concatenate(
                        (np.arange(num_individuals)[:, None], group_assigments[:, None]), axis=1
                    ),
                    columns=("bee_id", "group"),
                )
                factor_df = factor_df.merge(truth_df)

                embeddings = model.embeddings.weight.abs().argmax(axis=1).cpu().data.numpy()
                factor_df = factor_df.merge(
                    pd.DataFrame(
                        np.stack((np.arange(num_individuals), embeddings)).T,
                        columns=(("bee_id", "group_assigned")),
                    )
                )
                assignment_df = factor_df.drop_duplicates("bee_id")
                ami = sklearn.metrics.adjusted_mutual_info_score(
                    assignment_df.group, assignment_df.group_assigned
                )

                print(ami)

        return model

    mean_trajectory, offsets, group_trajectories = generate_trajectories()
    group_assigments = np.random.choice(num_groups, num_individuals)
    group_life_expectancy_mean = np.random.normal(life_expectancy_mean, 10, size=num_groups)
    birthdays, deathdays, individual_factors, individual_ages = generate_individuals()

    results = []
    for noise_std in tqdm.tqdm(np.linspace(min_noise, max_noise, num_noise_samples)):
        interactions = generate_interactions(noise_std)
        model = get_model()

        factor_df = model.get_factor_df(
            ids=np.arange(num_individuals),
            embedding_dim=model.num_embeddings,
            valid_ages=individual_ages >= 0,
        )
        factor_df.age = factor_df.age.astype(int)
        factor_df.day = factor_df.day.astype(int)
        factor_df.bee_id = factor_df.bee_id.astype(int)

        truth_df = pd.DataFrame(
            np.concatenate(
                (np.arange(num_individuals)[:, None], group_assigments[:, None]), axis=1
            ),
            columns=("bee_id", "group"),
        )
        factor_df = factor_df.merge(truth_df)

        embeddings = model.embeddings.weight.abs().argmax(axis=1).cpu().data.numpy()
        factor_df = factor_df.merge(
            pd.DataFrame(
                np.stack((np.arange(num_individuals), embeddings)).T,
                columns=(("bee_id", "group_assigned")),
            )
        )
        assignment_df = factor_df.drop_duplicates("bee_id")
        ami = sklearn.metrics.adjusted_mutual_info_score(
            assignment_df.group, assignment_df.group_assigned
        )

        data = (
            mean_trajectory,
            offsets,
            group_trajectories,
            group_assigments,
            group_life_expectancy_mean,
            birthdays,
            deathdays,
            individual_factors,
            individual_ages,
        )
        results.append((noise_std, data, model, factor_df, ami))

    torch.save((data, results), f"{output_path}/{str(uuid.uuid4())}.pt")


if __name__ == "__main__":
    synth()
