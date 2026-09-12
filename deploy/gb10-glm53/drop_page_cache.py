#!/usr/bin/env python3
"""Drop the page cache of the files a GLM-5.3-Flash boot reads - no sudo.

WHY THIS IS NOT OPTIONAL ON GB10. vLLM's start-up check compares the free
memory torch reports with gpu_memory_utilization x total. On unified memory
that figure is effectively MemFree, and page cache does NOT count as free,
even though the kernel would hand it back without a fight. Every boot reads
all 62 checkpoint shards (306 GB) for the non-expert weights - the experts
come from the planes cache, the rest does not - and leaves ~44 GiB of page
cache behind. The NEXT boot then fails the check (measured: MemFree fell
118 -> 110 -> 74 GiB over consecutive boots until the server would not start).

posix_fadvise(DONTNEED) on every file we can open releases it without root.
Measured on the reference box: +43.6 GiB (MemFree 74.4 -> 118.1 GiB).
Freshly WRITTEN files are the exception: fadvise forces their write-back
first, so running this right after a large copy is slow, not broken.

Docker image layers are root-owned and cannot be opened by a normal user;
boot.sh drops those by running this same script in a container with
/var/lib/docker bind-mounted read-only (measured: 20.8 GiB of layer cache
after `docker save`/`docker run` of a large image).

    drop_page_cache.py DIR [DIR ...] [--min-size BYTES]
"""
import argparse
import os


def memfree_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) / 1048576
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-size", type=int, default=0,
                    help="skip files smaller than this (bytes)")
    a = ap.parse_args()
    before = memfree_gib()
    n = 0
    for root in a.dirs:
        for dp, _, names in os.walk(root):
            for name in names:
                p = os.path.join(dp, name)
                try:
                    if os.path.islink(p) or (a.min_size and os.path.getsize(p) < a.min_size):
                        continue
                    fd = os.open(p, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    after = memfree_gib()
    print(f"fadvise(DONTNEED) on {n} files: MemFree {before:.2f} -> {after:.2f} GiB "
          f"({after - before:+.2f})")


if __name__ == "__main__":
    main()
