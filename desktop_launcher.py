import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import webview


BASE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = BASE_DIR / "frontend_server.py"
APP_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{APP_URL}/api/health"


def _is_server_ready(timeout: float = 1.0) -> bool:
    try:
        with urlopen(HEALTH_URL, timeout=timeout) as response:
            return response.status == 200
    except URLError:
        return False
    except Exception:
        return False


def _wait_for_server(max_wait_seconds: float = 25.0) -> bool:
    start = time.time()
    while time.time() - start < max_wait_seconds:
        if _is_server_ready():
            return True
        time.sleep(0.35)
    return False


def _start_server_process() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        cwd=str(BASE_DIR),
    )


def launch_desktop_app() -> None:
    owns_server = False
    server_process = None

    if not _is_server_ready():
        server_process = _start_server_process()
        owns_server = True

        if not _wait_for_server():
            if server_process.poll() is None:
                server_process.terminate()
            raise RuntimeError("Nie udalo sie uruchomic lokalnego serwera API w wymaganym czasie.")

    try:
        webview.create_window(
            "Local File Assistant",
            APP_URL,
            width=1460,
            height=920,
            min_size=(1100, 720),
        )
        webview.start(debug=False)
    finally:
        if owns_server and server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except Exception:
                server_process.kill()


if __name__ == "__main__":
    launch_desktop_app()
