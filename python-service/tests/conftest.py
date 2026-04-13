import os


def pytest_sessionstart(session):

    if "LLM_API_KEY" not in os.environ:
        os.environ["LLM_API_KEY"] = "fake-key-for-ci-test-123"
