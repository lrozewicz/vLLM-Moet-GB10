#!/usr/bin/env python3
"""Host-memory guard for GB10 bring-up: kill the container before the BOX dies.

Why this exists, when both docker's mem_limit and memguard.sh already do
"something": on this machine neither actually protects the host.

* `mem_limit` does not see the allocations that matter. Measured 2026-08-29
  during a GLM-5.3-Flash load: `docker stats` reported **9.57 GiB / 100 GiB**
  for the container while the host was at **118 GB used** and `Shmem` was
  51 MB. The memory goes through the NVIDIA driver on unified memory and is
  not charged to the container's cgroup, so no limit value helps.
* `memguard.sh` watches PSI and log progress, and shells out per check. Once
  the box is thrashing it stops being scheduled and its `docker`/`grep`
  invocations cannot fork - it slept through two wedges today.

So this guard: one long-lived process, started BEFORE the container, that
reads `/proc/meminfo` directly (no fork) and kills via a raw HTTP request on
the docker unix socket (no fork). It fires on a headroom threshold rather
than on pressure, because pressure arrives too late - by the time PSI is
high, `sshd` can no longer accept a login and the only way back is the power
button.

    setsid ./hostguard.py glm53-moet 25 > ~/glm53-run/hostguard.log 2>&1 &

Args: <container> [min_available_gib] [poll_seconds]
"""

import socket
import sys
import time

CONTAINER = sys.argv[1] if len(sys.argv) > 1 else "glm53-moet"
MIN_AVAIL_GIB = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
POLL_S = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
SOCK = "/var/run/docker.sock"


def available_gib() -> float:
    # Read, don't parse the world: MemAvailable is the only line we need, and
    # this must not allocate meaningfully while the box is under pressure.
    with open("/proc/meminfo", "rb") as f:
        for line in f:
            if line.startswith(b"MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return float("inf")


def kill_container(name: str) -> str:
    """POST /containers/<name>/kill over the docker socket, no subprocess."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(20)
    s.connect(SOCK)
    req = (
        f"POST /containers/{name}/kill?signal=SIGKILL HTTP/1.1\r\n"
        "Host: docker\r\nConnection: close\r\n\r\n"
    ).encode()
    s.sendall(req)
    out = b""
    try:
        while True:
            b = s.recv(4096)
            if not b:
                break
            out += b
    except Exception:  # noqa: BLE001 - a truncated reply still means it fired
        pass
    s.close()
    return out.split(b"\r\n", 1)[0].decode(errors="replace")


def main() -> int:
    print(f"hostguard: watching {CONTAINER}, floor {MIN_AVAIL_GIB} GiB, "
          f"poll {POLL_S}s", flush=True)
    low_streak = 0
    while True:
        avail = available_gib()
        if avail < MIN_AVAIL_GIB:
            low_streak += 1
            print(f"{time.strftime('%H:%M:%S')} available {avail:.1f} GiB "
                  f"< {MIN_AVAIL_GIB} ({low_streak})", flush=True)
            # Two consecutive reads, so a momentary dip during a large but
            # transient allocation does not kill a healthy load.
            if low_streak >= 2:
                status = kill_container(CONTAINER)
                print(f"{time.strftime('%H:%M:%S')} KILLED {CONTAINER}: "
                      f"{status}", flush=True)
                return 0
        else:
            low_streak = 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
