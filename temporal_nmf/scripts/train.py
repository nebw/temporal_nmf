import datetime
import sys

import click
import pandas as pd
import torch

from temporal_nmf.training import DataWrapper, TrainingWrapper


@click.command()
@click.argument("base_path", type=click.Path(exists=True))
@click.argument("output_path")
@click.option("--batch_size", default=64)
@click.option("--num_batches", default=200000)
@click.option("--num_hidden", default=32)
@click.option("--num_embeddings", default=16)
@click.option("--num_factors", default=8)
@click.option("--lambda_factor_l1", default=0.005)
@click.option("--lambda_basis_function_l1", default=0.01)
@click.option("--lambda_embedding_sparsity", default=0.1)
@click.option("--lambda_factor_nonnegativity", default=1)
@click.option("--lambda_adversarial", default=0.1)
def train(base_path, output_path, batch_size, num_batches, **kwargs):
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    lambdas = {
        "factor_l1": kwargs["lambda_factor_l1"],
        "basis_function_l1": kwargs["lambda_basis_function_l1"],
        "embedding_sparsity": kwargs["lambda_embedding_sparsity"],
        "factor_nonnegativity": kwargs["lambda_factor_nonnegativity"],
        "adversarial": kwargs["lambda_adversarial"],
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(lambdas)

    data16 = DataWrapper(base_path, "bn16", datetime.datetime(2016, 7, 23))
    data19 = DataWrapper(base_path, "bn19", datetime.datetime(2019, 7, 25))

    trainer = TrainingWrapper((data16, data19), device, lambdas)

    for i in range(len(trainer.loss_hist), num_batches):
        sys.stdout.write("\r{}/{} - {:.2f}".format(i, num_batches, trainer.batch(batch_size)))

        if (i % 1000) == 0:
            print("\n\n", pd.DataFrame(trainer.loss_hist[-1000:]).mean(axis=0))

    trainer.save(output_path)


if __name__ == "__main__":
    train()
