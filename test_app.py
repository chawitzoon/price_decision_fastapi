# tbd
# for now test app.py directly

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_read_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hello root route"}


def test_add(client):
    response = client.get("/add/1/2")
    assert response.status_code == 200
    assert response.json() == {"result": 3}


def test_subtract(client):
    response = client.get("/subtract/1/2")
    assert response.status_code == 200
    assert response.json() == {"result": -1}


def test_multiply(client):
    response = client.get("/multiply/1/2")
    assert response.status_code == 200
    assert response.json() == {"result": 2}


def test_divide(client):
    response = client.get("/divide/1/2")
    assert response.status_code == 200
    assert response.json() == {"result": 0.5}


def test_predict_next_price(client):
    response = client.get("/predict_next_price/120/121.4/126.9/128.0/127.8/129.1/130.1")
    assert response.status_code == 200
    assert response.json() == {"result": 119.99357}
