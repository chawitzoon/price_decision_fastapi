import click
from mlib.mlib import predict
import numpy as np

def parse_prices(_ctx, _param, value):
    try:
        # Split the string by commas and convert to a list of floats
        prices = [float(x) for x in value.split(',')]
        if len(prices) != 7:
            raise click.BadParameter("Exactly 7 prices are required.")
        return prices
    except ValueError as e:
        raise click.BadParameter("Prices should be a list of floats.") from e

@click.command()
@click.option(
    "--prices",
    prompt="7 lookback prices list (comma-separated)",
    callback=parse_prices,
    help="Pass in the list of 7 lookback prices to predict next price float value, separated by commas (e.g., 1.1,2.2,3.3,4.4,5.5,6.6,7.7)",
)
def predictcli(prices):
    """Predict next price float value based on list of 7 lookback prices"""
    input_prices = np.array([prices])
    next_price_predicted = predict(input_prices)
    next_price_predicted_float = round(float(next_price_predicted), 5)

    # for testing the color only
    if next_price_predicted > prices[0]:
        click.echo(click.style(f"price increases to {next_price_predicted_float}", bg="green", fg="white"))
    else:
        click.echo(click.style(f"price decreases to {next_price_predicted_float}", bg="red", fg="white"))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    predictcli()
