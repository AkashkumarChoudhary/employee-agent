from employee_agent.api.store import JobStore


def test_create_and_get():
    store = JobStore()
    store.create("j1", owner="key-a", role="hr_analyst")
    rec = store.get("j1")
    assert rec.owner == "key-a"
    assert rec.status == "running"


def test_set_status():
    store = JobStore()
    store.create("j1", owner="key-a", role="hr_analyst")
    store.set_status("j1", "done")
    assert store.get("j1").status == "done"


def test_get_missing_returns_none():
    assert JobStore().get("nope") is None
