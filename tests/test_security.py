import io

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


def test_health_is_public(client):
    assert client.get("/api/v1/health").status_code == 200


def test_protected_endpoints_require_token(client):
    for method, url in [
        ("get", "/api/v1/candidates"),
        ("get", "/api/v1/candidates/x"),
        ("get", "/api/v1/reports/x"),
        ("get", "/api/v1/jd/list"),
        ("get", "/api/v1/departments"),
        ("get", "/api/v1/auth/me"),
        ("post", "/api/v1/upload"),
    ]:
        assert getattr(client, method)(url).status_code == 401, url


def test_x_user_email_header_no_longer_grants_access(client):
    r = client.get("/api/v1/candidates", headers={"X-User-Email": "admin@talash.pk"})
    assert r.status_code == 401


def test_forged_and_legacy_tokens_rejected(client):
    for tok in ["token_admin@talash.pk", "garbage", ""]:
        r = client.get("/api/v1/candidates", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 401


def test_old_default_admin_password_rejected(client):
    r = client.post("/api/v1/auth/login", json={"email": "admin@talash.pk", "password": "talash2026"})
    assert r.status_code == 401


def test_admin_login_with_configured_password(client):
    r = client.post("/api/v1/auth/login", json={"email": "admin@talash.pk", "password": "AdminPass-For-Tests1"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "admin"


def test_register_login_logout_flow(client, make_user):
    email, h = make_user()
    me = client.get("/api/v1/auth/me", headers=h)
    assert me.status_code == 200 and me.json()["email"] == email
    assert client.post("/api/v1/auth/logout", headers=h).status_code == 200
    assert client.get("/api/v1/auth/me", headers=h).status_code == 401


def test_password_rules_and_duplicates(client, make_user):
    r = client.post("/api/v1/auth/register", json={"name": "a", "email": "short@example.com", "password": "1234567"})
    assert r.status_code == 400
    r = client.post("/api/v1/auth/register", json={"name": "a", "email": "not-an-email", "password": "longenough1"})
    assert r.status_code == 400
    email, _ = make_user()
    r = client.post("/api/v1/auth/register", json={"name": "a", "email": email, "password": "longenough1"})
    assert r.status_code == 400


def test_password_hash_is_salted_pbkdf2():
    from api import auth
    a, b = auth._hash_password("same"), auth._hash_password("same")
    assert a != b and a.startswith("pbkdf2$")
    assert auth._verify_password("same", a) and not auth._verify_password("other", a)


def test_login_lockout(client):
    for _ in range(5):
        assert client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x" * 9}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x" * 9}).status_code == 429


def test_tenants_are_isolated(client, make_user):
    _, h1 = make_user()
    _, h2 = make_user()
    assert client.get("/api/v1/candidates", headers=h1).json() == []
    assert client.get("/api/v1/candidates", headers=h2).json() == []
    jd = client.post("/api/v1/jd/upload", headers=h1, data={"jd_text": "Assistant Professor of Computer Science. PhD required. 3 years experience. Python, machine learning."})
    assert jd.status_code == 200, jd.text
    jd_id = jd.json()["jd_id"]
    assert client.get(f"/api/v1/jd/{jd_id}/results", headers=h1).status_code == 200
    assert client.get(f"/api/v1/jd/{jd_id}/results", headers=h2).status_code == 404
    assert [j["jd_id"] for j in client.get("/api/v1/jd/list", headers=h2).json()["jds"]] == []


def test_jd_id_path_traversal_blocked(client, make_user):
    _, h = make_user()
    r = client.get("/api/v1/jd/..%2F..%2Fx/results", headers=h)
    assert r.status_code in (400, 404)
    assert client.get("/api/v1/jd/../results", headers=h).status_code in (400, 404)


def test_upload_rejects_non_pdf_and_fake_pdf(client, make_user):
    _, h = make_user()
    r = client.post("/api/v1/upload", headers=h, files={"file": ("cv.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/v1/upload", headers=h, files={"file": ("cv.pdf", b"not really a pdf", "application/pdf")})
    assert r.status_code == 400


def test_upload_size_limit(client, make_user, monkeypatch):
    from api import uploads
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 1024)
    _, h = make_user()
    r = client.post("/api/v1/upload", headers=h, files={"file": ("big.pdf", PDF + b"0" * 5000, "application/pdf")})
    assert r.status_code == 413


def test_safe_pdf_name_strips_traversal():
    from api.uploads import safe_pdf_name
    assert safe_pdf_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert safe_pdf_name(r"..\..\win\evil cv.PDF") == "evil_cv.pdf"
    assert "/" not in safe_pdf_name("a/b/c.pdf")


def test_static_analysis_no_longer_public(client):
    assert client.get("/static/analysis/composite_evaluations.csv").status_code == 404


def test_cors_not_wildcard(client):
    r = client.options("/api/v1/candidates", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-origin" not in r.headers
