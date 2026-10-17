"""Fetch the BeeBot working data from Monash Bridges (figshare) without
downloading the full multi-GB archives.

Big archives are read remotely: we fetch the zip's central directory with
HTTP range requests, then stream only the members we need. Downloads resume
if interrupted, and every extracted member is CRC-checked.

Usage (run from the repo root):
    python -I scripts/fetch_data.py                  # tier 1, about 570 MB
    python -I scripts/fetch_data.py --spatial T1 T7  # add Spatial test videos (about 1 GB each)
    python -I scripts/fetch_data.py --list           # show what is available
"""
import argparse
import io
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

RAW = Path("raw")
CHUNK = 1 << 20

# figshare file ids (CC BY 4.0, cite both datasets)
TRAJECTORIES = 38169609      # Spatial Monitoring: Insect_Trajectory_Dataset.zip (100 MB)
SPATIAL_VIDEOS = 38175570    # Spatial Monitoring: Test_Video_Dataset.zip (11.4 GB)
EXPERIMENT_DATA = 24528083   # Honeybee tracking: Experiment_Data.zip (10.9 GB)

EXP1 = "Experiment_Data/Experiment_1/"
TIER1_MEMBERS = [
    EXP1 + "Experiment_1-Test_Videos.zip",
    EXP1 + "Experiment_1-HyDaT_Results/Experiment_1-CSV_HyDaT.zip",
    EXP1 + "Experiment_1-YOLO_Results/Experiment_1-CSV_YOLO.zip",
    EXP1 + "Experiment1-Results.xlsx",
    EXP1 + "Experiment_1-Background_Changes.zip",
    "Experiment_Data/Experiment_2/Experiment_2-Readme.txt",
    "Experiment_Data/Experiment_2/Experiment_2 - Observations_Record.xlsx",
    "Experiment_Data/Data_Analysis_output/Data_Analysis-Scaevola/Data_Analysis-Scaevola-CSV.zip",
    "Experiment_Data/Data_Analysis_output/Data_Analysis-Lambs-ear/Data_Analysis-Lambs-ear-CSV.zip",
]


def url(file_id):
    return f"https://ndownloader.figshare.com/files/{file_id}"


def get(file_id, start, end=None, retries=5):
    """Open a ranged GET. The figshare redirect is re-resolved each call because
    the signed S3 URL expires after a few minutes."""
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url(file_id), headers={"Range": rng}), timeout=60)
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(2 ** attempt)


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file, used only to read zip metadata."""

    def __init__(self, file_id):
        self.file_id = file_id
        with get(file_id, 0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size or len(buf) == 0:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        with get(self.file_id, self.pos, end) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(file_id):
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(file_id), buffer_size=1 << 16))


def stream_to(file_id, start, length, dest):
    """Download bytes [start, start+length) into dest, resuming a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have >= length:
        return
    t0, done = time.time(), have
    with get(file_id, start + have, start + length - 1) as r, open(dest, "ab") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
            print(f"\r  {done / 1e6:8.1f} / {length / 1e6:.1f} MB  {rate:5.1f} MB/s", end="", flush=True)
    print()


def extract_member(file_id, zf, name, out_dir):
    """Stream one member's compressed bytes, then inflate and CRC-check it."""
    info = zf.getinfo(name)
    out = out_dir / Path(name).name
    if out.exists() and out.stat().st_size == info.file_size:
        print(f"have  {out}")
        return out
    print(f"fetch {name} ({info.compress_size / 1e6:.1f} MB)")
    with get(file_id, info.header_offset, info.header_offset + 29) as r:
        header = r.read()
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len

    part = out_dir / "_parts" / (Path(name).name + ".part")
    stream_to(file_id, data_start, info.compress_size, part)

    crc = 0
    tmp = out.with_suffix(out.suffix + ".tmp")
    inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    if info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
        sys.exit(f"unsupported compression {info.compress_type} for {name}")
    with open(part, "rb") as src, open(tmp, "wb") as dst:
        while chunk := src.read(CHUNK):
            data = inflater.decompress(chunk) if inflater else chunk
            crc = zlib.crc32(data, crc)
            dst.write(data)
        if inflater:
            tail = inflater.flush()
            crc = zlib.crc32(tail, crc)
            dst.write(tail)
    if crc != info.CRC:
        tmp.unlink()
        part.unlink()
        sys.exit(f"CRC mismatch for {name}; deleted the partial download, rerun to retry")
    tmp.replace(out)
    part.unlink()
    return out


def unzip_local(path, out_dir):
    """Unzip a downloaded zip next to itself. zipfile sanitizes member paths."""
    marker = out_dir / f".{path.name}.extracted"
    if marker.exists():
        return
    print(f"unzip {path.name} -> {out_dir}")
    with zipfile.ZipFile(path) as z:
        z.extractall(out_dir)
    marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial", nargs="*", default=[], metavar="T#",
                    help="Spatial Monitoring test videos to fetch, e.g. T1 T7 (about 1 GB each)")
    ap.add_argument("--list", action="store_true", help="list available members and exit")
    args = ap.parse_args()

    if args.list:
        for label, fid in [("Honeybee Experiment_Data.zip", EXPERIMENT_DATA),
                           ("Spatial Test_Video_Dataset.zip", SPATIAL_VIDEOS)]:
            print(f"== {label}")
            for i in open_remote_zip(fid).infolist():
                if not i.is_dir():
                    print(f"  {i.file_size / 1e6:9.1f} MB  {i.filename}")
        return

    # 1. Spatial Monitoring trajectories: small, so download the whole zip
    traj_dir = RAW / "spatial_monitoring"
    traj_zip = traj_dir / "Insect_Trajectory_Dataset.zip"
    with get(TRAJECTORIES, 0, 0) as r:
        size = int(r.headers["Content-Range"].split("/")[1])
    print(f"fetch Insect_Trajectory_Dataset.zip ({size / 1e6:.1f} MB)")
    stream_to(TRAJECTORIES, 0, size, traj_zip)
    unzip_local(traj_zip, traj_dir / "trajectories")

    # 2. Honeybee Experiment_1: short test videos with per-frame reference tracks
    hb_dir = RAW / "honeybee"
    zf = open_remote_zip(EXPERIMENT_DATA)
    for name in TIER1_MEMBERS:
        out = extract_member(EXPERIMENT_DATA, zf, name, hb_dir)
        if out.suffix == ".zip":
            unzip_local(out, hb_dir / out.stem)

    # 3. Optional Spatial Monitoring test videos
    if args.spatial:
        sp_dir = RAW / "spatial_monitoring" / "videos"
        zf = open_remote_zip(SPATIAL_VIDEOS)
        wanted = {t.upper() for t in args.spatial}
        for i in zf.infolist():
            if not i.is_dir() and i.filename.split("/")[0].upper() in wanted:
                extract_member(SPATIAL_VIDEOS, zf, i.filename, sp_dir)

    print("done")


if __name__ == "__main__":
    main()
