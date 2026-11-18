"""Behavior-focused tests using a local HTTP server and real CLI calls."""

import contextlib
import subprocess
import sys
import threading
import time
import unittest
from _thread import LockType
from collections.abc import Generator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

import speed_test

SCRIPT = Path(speed_test.__file__).resolve()
PAYLOAD = b"cat!" * 50_000  # Greater than one read chunk.


@dataclass(slots=True)
class ServerState:
    requests: int = 0
    active: int = 0
    max_active: int = 0
    lock: LockType = field(default_factory=threading.Lock)


@contextlib.contextmanager
def local_server(
    payload: bytes = PAYLOAD, status: int = 200
) -> Generator[tuple[str, ServerState]]:
    """Serve deterministic bytes and record observable HTTP concurrency."""
    state = ServerState()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with state.lock:
                state.requests += 1
                state.active += 1
                state.max_active = max(state.max_active, state.active)

            try:
                self.send_response(status)
                length = len(payload) if status == 200 else 0
                self.send_header("Content-Length", str(length))
                self.end_headers()
                if status == 200:
                    halfway = len(payload) // 2
                    _ = self.wfile.write(payload[:halfway])
                    self.wfile.flush()
                    time.sleep(0.03)
                    _ = self.wfile.write(payload[halfway:])
            finally:
                with state.lock:
                    state.active -= 1

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Keep the test run free of HTTP access logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/test.bin", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_cli(url: str) -> subprocess.CompletedProcess[str]:
    """Invoke the public command-line interface in a separate process."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), url],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


class StatisticsTests(unittest.TestCase):
    def test_statistics_use_total_bytes_and_total_duration(self) -> None:
        results = [
            speed_test.DownloadResult(bytes_received=1_000_000, elapsed_seconds=0.5),
            speed_test.DownloadResult(bytes_received=3_000_000, elapsed_seconds=1.5),
        ]
        stats = speed_test.calculate_statistics(results)

        self.assertEqual(stats.request_count, 2)
        self.assertEqual(stats.total_bytes, 4_000_000)
        self.assertEqual(stats.average_seconds, 1.0)
        self.assertEqual(stats.speed_mbps, 16.0)

    def test_empty_results_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ = speed_test.calculate_statistics([])


class DownloadTests(unittest.TestCase):
    def test_download_reads_entire_body(self) -> None:
        with local_server() as (url, state):
            result = speed_test.download(url)

        self.assertEqual(result.bytes_received, len(PAYLOAD))
        self.assertGreater(result.elapsed_seconds, 0)
        self.assertEqual(state.requests, 1)

    def test_measure_downloads_ten_times_without_overlap(self) -> None:
        with local_server() as (url, state):
            results = list(speed_test.measure(url))

        self.assertEqual(len(results), 10)
        self.assertTrue(all(item.bytes_received == len(PAYLOAD) for item in results))
        self.assertEqual(state.requests, 10)
        self.assertEqual(state.max_active, 1)


class CliTests(unittest.TestCase):
    def test_cli_succeeds(self) -> None:
        with local_server() as (url, state):
            completed = run_cli(url)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(state.requests, 10)
        self.assertIn("Mbit/s", completed.stdout)

    def test_cli_rejects_invalid_url(self) -> None:
        completed = run_cli("ftp://example.com/test.bin")

        self.assertEqual(completed.returncode, 2)
        self.assertIn("URL", completed.stderr)

    def test_cli_reports_http_errors(self) -> None:
        with local_server(status=503) as (url, state):
            completed = run_cli(url)

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(state.requests, 1)
        self.assertIn("503", completed.stderr)


if __name__ == "__main__":
    _ = unittest.main()
