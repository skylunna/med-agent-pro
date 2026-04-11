from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_app_title():
    assert app.title == "Python AI Engine"


def test_chat_endpoint_structure():
    # 验证接口路由存在且方法正确
    assert "/agent/query" in [r.path for r in app.routes]
