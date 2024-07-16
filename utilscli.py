import click
import requests
# from mlib.mlib import retrain


@click.group()
@click.version_option("1.0")
def cli():
    """Machine Learning Utility Belt"""


# @cli.command("retrain")
# @click.option("--tsize", default=0.1, help="Test Size")
# def retrain(tsize):
#     """Retrain Model

#     You may want to extend this with more options, such as setting model_name
#     """
#     click.echo(click.style("Retraining Model", bg="green", fg="white"))
#     accuracy, model_name = retrain(tsize=tsize)
#     click.echo(
#         click.style(f"Retrained Model Accuracy: {accuracy}", bg="blue", fg="white")
#     )
#     click.echo(click.style(f"Retrained Model Name: {model_name}", bg="red", fg="white"))


@cli.command("predict")
@click.option("--prices", prompt="7 lookback prices list", help="Pass in the list of 7 lookback prices to predict next price float value")
@click.option("--host", default="http://localhost:8080/predict_next_price", help="Host to query")
def mkrequest(prices, host):
    """Sends prediction to ML Endpoint"""

    try:
        prices_list = [float(x) for x in prices.split(',')]
        if len(prices_list) != 7:
            raise ValueError("Exactly 7 prices are required.")
    except ValueError as e:
        click.echo(click.style(f"Invalid input: {e}", bg="red", fg="white"))
        return

    click.echo(
        click.style(
            f"Querying host {host} with prices: {prices}", bg="yellow", fg="white"
        )
    )
    payload = {"prices": prices_list}
    try:
        result = requests.post(url=host, json=payload)
        result.raise_for_status()
        click.echo(click.style(f"Result: {result.json()}", bg="green", fg="white"))
    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", bg="red", fg="white"))


if __name__ == "__main__":
    cli()
