def test_file_watcher_import():
    from sreejita.automation.file_watcher import start_watcher
    assert callable(start_watcher)
