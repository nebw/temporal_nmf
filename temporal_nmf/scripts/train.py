#!/usr/bin/env python3

import datetime
import sys

import click
import pandas as pd
import torch
import tqdm.auto as tqdm

from temporal_nmf.training import DataWrapper, TrainingWrapper


@click.command()
@click.argument("base_path", type=click.Path(exists=True))
@click.argument("output_path")
@click.option("--batch_size_timesteps", default=8)
@click.option("--batch_size_entities", default=256)
@click.option("--device", default="cuda:0")
@click.option("--num_batches", default=50000)
@click.option("--num_hidden", default=32)
@click.option("--num_embeddings", default=16)
@click.option("--num_factors", default=8)
@click.option("--lambda_factor_l1", default=0.005)
@click.option("--lambda_basis_function_l1", default=0.01)
@click.option("--lambda_embedding_sparsity", default=0.1)
@click.option("--lambda_factor_nonnegativity", default=1)
@click.option("--lambda_adversarial", default=0.1)
def train(
    base_path, output_path, batch_size_timesteps, batch_size_entities, device, num_batches, **kwargs
):
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    lambdas = {
        "factor_l1": kwargs["lambda_factor_l1"],
        "basis_function_l1": kwargs["lambda_basis_function_l1"],
        "embedding_sparsity": kwargs["lambda_embedding_sparsity"],
        "factor_nonnegativity": kwargs["lambda_factor_nonnegativity"],
        "adversarial": kwargs["lambda_adversarial"],
    }
    print(lambdas)

    data16 = DataWrapper(base_path, "bn16", datetime.datetime(2016, 7, 23))
    data19 = DataWrapper(base_path, "bn19", datetime.datetime(2019, 7, 25))

    trainer = TrainingWrapper(
        (data16, data19),
        device,
        lambdas,
        num_hidden=kwargs["num_hidden"],
        num_embeddings=kwargs["num_embeddings"],
        num_factors=kwargs["num_factors"],
    )

    for i in tqdm.trange(len(trainer.loss_hist), num_batches):
        trainer.batch(batch_size_timesteps, batch_size_entities)

    trainer.save(output_path)


if __name__ == "__main__":
    train()
