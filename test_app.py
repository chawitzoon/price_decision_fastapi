import pytest
from fastapi.testclient import TestClient
from app import app
from click.testing import CliRunner
from cli import predictcli
import utilscli


@pytest.fixture
def client():
    return TestClient(app)


def test_read_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hello root route"}


def test_clisearch():
    runner = CliRunner()
    result = runner.invoke(predictcli, ["--prices", "1.1,2.2,3.3,4.4,5.5,6.6,7.7"])
    assert result.exit_code == 0
    assert "price increases to 1.84918" in result.output


def test_retrain():
    runner = CliRunner()
    result = runner.invoke(utilscli.cli, ["--version"])
    assert result.exit_code == 0