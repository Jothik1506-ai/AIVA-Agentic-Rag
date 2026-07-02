"""Gunicorn configuration — picked up automatically from the working directory.

System initialization must start inside the WORKER process, after the fork.
Starting it at import time runs it in the master; forking while that thread
holds import/SSL locks leaves the worker's init permanently deadlocked.
"""


def post_fork(server, worker):
    """Runs in the worker process right after the fork."""
    from app import start_system_init
    start_system_init()
    worker.log.info("system-init thread started in worker pid %s", worker.pid)
