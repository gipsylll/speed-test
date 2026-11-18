import argparse
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

REQUEST_COUNT = 10
CHUNK_SIZE = 64 * 1024
TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class DownloadResult:
    bytes_received: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SpeedStatistics:
    request_count: int
    total_bytes: int
    average_seconds: float
    speed_mbps: float


class CliArgs(argparse.Namespace):
    url: str = ""


def download(url: str) -> DownloadResult:
    """Read the entire HTTP response without retaining its body in memory."""
    started_at = time.perf_counter()

    with httpx.stream("GET", url, timeout=30) as response:
        _ = response.raise_for_status()

        downloaded_bytes = 0

        for chunk in response.iter_raw():
            downloaded_bytes += len(chunk)

    return DownloadResult(downloaded_bytes, time.perf_counter() - started_at)


def measure(url: str, count: int = REQUEST_COUNT) -> Iterator[DownloadResult]:
    """Yield each completed download before starting the next one."""
    for _ in range(count):
        yield download(url)


def calculate_statistics(results: Sequence[DownloadResult]) -> SpeedStatistics:
    """Calculate throughput over the total time, not mean per-request speeds."""
    if not results:
        raise ValueError("At least one download result is required")

    total_bytes = sum(result.bytes_received for result in results)
    total_seconds = sum(result.elapsed_seconds for result in results)
    if total_seconds <= 0:
        raise ValueError("Total elapsed time must be positive")

    return SpeedStatistics(
        request_count=len(results),
        total_bytes=total_bytes,
        average_seconds=total_seconds / len(results),
        speed_mbps=total_bytes * 8 / total_seconds / 1_000_000,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure download speed with 10 sequential HTTP requests."
    )

    _ = parser.add_argument(
        "url",
        help="HTTP(S) URL to download ten times",
    )

    args = CliArgs()
    _ = parser.parse_args(argv, namespace=args)

    url = args.url

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("URL must start with http:// or https:// and include a host")

    results: list[DownloadResult] = []
    try:
        for index, result in enumerate(measure(url), start=1):
            results.append(result)
            size = f"{result.bytes_received:,} bytes"
            duration = f"{result.elapsed_seconds:.3f} s"
            print(f"{index:2}/{REQUEST_COUNT}: {size} in {duration}")
    except httpx.HTTPStatusError as exc:
        print(
            f"HTTP {exc.response.status_code}: {exc}",
            file=sys.stderr,
        )
        return 1

    except httpx.RequestError as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1

    stats = calculate_statistics(results)
    size = f"{stats.total_bytes:,} bytes ({stats.total_bytes / 2**20:.2f} MiB)"
    print(f"\nAverage request time: {stats.average_seconds:.3f} s")
    print(f"Total downloaded:     {size}")
    print(f"Download speed:       {stats.speed_mbps:.2f} Mbit/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
