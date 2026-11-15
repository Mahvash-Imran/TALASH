"""Test fixtures: run the API against an isolated temporary working directory."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    work = tmp_path_factory.mktemp("talash_work")
    os.environ["TALASH_ADMIN_PASSWORD"] = "AdminPass-For-Tests1"
    prev = os.getcwd()
    os.chdir(work)
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as c:
            yield c
    finally:
        os.chdir(prev)


def _register(client, email, password="CorrectHorse9"):
    r = client.post("/api/v1/auth/register", json={"name": "T", "email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.fixture()
def make_user(client):
    counter = {"n": 0}

    def _make():
        counter["n"] += 1
        email = f"user{counter['n']}_{os.urandom(3).hex()}@example.com"
        token = _register(client, email)
        return email, {"Authorization": f"Bearer {token}"}

    return _make
