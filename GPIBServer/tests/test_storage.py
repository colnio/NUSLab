import json

import pytest

from GPIBServer import storage


def windows_lock():
    error = PermissionError('Transient Windows file lock')
    error.winerror = 5
    return error


def test_atomic_metadata_retries_transient_lock(tmp_path, monkeypatch):
    path = tmp_path / 'run.meta.json'
    path.write_text('{"old": true}')
    original = storage.os.replace
    calls = []
    waits = []

    def replace(source, destination):
        calls.append((source, destination))
        if len(calls) <= 2:
            assert json.loads(path.read_text()) == {'old': True}
            raise windows_lock()
        original(source, destination)

    monkeypatch.setattr(storage.os, 'replace', replace)
    monkeypatch.setattr(storage.time, 'sleep', waits.append)
    storage.atomic_json(path, {'new': True})
    assert json.loads(path.read_text()) == {'new': True}
    assert waits == [0.01, 0.02]
    assert not path.with_name(path.name + '.tmp').exists()


def test_persistent_lock_preserves_old_and_pending_data(tmp_path, monkeypatch):
    path = tmp_path / 'run.meta.json'
    path.write_text('{"old": true}')
    waits = []

    def replace(source, destination):
        raise windows_lock()

    monkeypatch.setattr(storage.os, 'replace', replace)
    monkeypatch.setattr(storage.time, 'sleep', waits.append)
    with pytest.raises(PermissionError):
        storage.atomic_json(path, {'new': True})
    assert json.loads(path.read_text()) == {'old': True}
    assert json.loads(path.with_name(path.name + '.tmp').read_text()) == {'new': True}
    assert len(waits) == 6
    assert sum(waits) == pytest.approx(0.63)


def test_other_storage_errors_fail_immediately(tmp_path, monkeypatch):
    def replace(source, destination):
        raise OSError('Disk failure')

    def sleep(seconds):
        pytest.fail('Unexpected retry for non-lock error')

    monkeypatch.setattr(storage.os, 'replace', replace)
    monkeypatch.setattr(storage.time, 'sleep', sleep)
    with pytest.raises(OSError, match='Disk failure'):
        storage.atomic_json(tmp_path / 'run.meta.json', {'new': True})


def test_dropbox_mirror_retries_lock_and_preserves_exact_data(tmp_path, monkeypatch):
    store = storage.RunStore(tmp_path / 'recovery', tmp_path / 'dropbox')
    record = store.create('reference', {})
    record.append_rows([{'current_a': 1e-7, 'voltage_v': .1}])
    original = storage.os.replace
    locked = set()

    def replace(source, destination):
        if str(destination).startswith(str(store.dropbox_root)) and destination not in locked:
            locked.add(destination)
            raise windows_lock()
        original(source, destination)

    monkeypatch.setattr(storage.os, 'replace', replace)
    monkeypatch.setattr(storage.time, 'sleep', lambda seconds: None)
    assert record.sync()['state'] == 'synced'
    assert len(locked) == 3
    for name in ['data', 'events', 'metadata']:
        relative = record.metadata['files'][name]
        assert (store.recovery_root / relative).read_bytes() == (store.dropbox_root / relative).read_bytes()
