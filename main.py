#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_qc.py / main.py - WAV Audio QC / EDA for Call Center recordings

Metrics:
- Metadata: sample rate, channels, duration, subtype/format
- Levels: Peak dBFS, RMS dBFS, Crest factor
- Loudness: Integrated LUFS (approx BS.1770 K-weighted + gating) if scipy available
- Clipping: % of samples near full-scale
- VAD (energy-adaptive): speech ratio, speech segments, silence stats, dead-air
- Noise: noise floor (10th percentile frame energy), estimated SNR
- Dropouts: near-zero % and longest near-zero run
- Stereo: overlap ratio (double-talk), channel correlation, crosstalk proxy
- Spectral: centroid / flatness / rolloff95 (speech/noise)
- Hum: 50/60 Hz band ratios (proxy)
- Echo proxy: max autocorr peak of envelope (20–200ms lag) on first N seconds

Output:
- CSV report
- Optional per-file JSON (segments)

Install:
  pip install numpy soundfile scipy

Usage:
  uv run .\main.py --input ".\data\calls" --recursive --out_csv ".\qc_report.csv" --out_json_dir ".\qc_json"
  uv run .\main.py --input ".\data\calls\sample.wav" --out_csv ".\qc_report.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import glob
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

# Optional deps
try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False

try:
    from scipy import signal
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


EPS = 1e-12


@dataclass
class QCConfig:
    # Frame params for energy/VAD
    frame_ms: float = 30.0
    hop_ms: float = 10.0

    # VAD thresholding
    vad_margin_db: float = 10.0          # speech if frame_db >= noise_floor_db + margin
    vad_abs_floor_db: float = -50.0      # never treat frames below this as speech
    min_speech_ms: int = 200
    min_silence_ms: int = 800

    # Clipping / near-zero
    clip_threshold: float = 0.999
    near_zero_threshold: float = 1e-4

    # Echo proxy
    echo_probe_seconds: float = 120.0
    echo_env_hz: int = 200
    echo_lag_min_ms: float = 20.0
    echo_lag_max_ms: float = 200.0

    # Spectral analysis
    spectral_fft_ms: float = 32.0
    spectral_hop_ms: float = 16.0
    rolloff_pct: float = 0.95

    # Hum detection
    hum_band_hz: float = 2.0
    hum_max_harmonic_hz: float = 300.0

    # QC thresholds (flags)
    flag_min_speech_ratio: float = 0.15
    flag_min_snr_db: float = 10.0
    flag_max_clipping_pct: float = 0.10  # percent
    flag_longest_zero_run_ms: float = 500.0
    flag_high_overlap_ratio: float = 0.20
    flag_lufs_too_quiet: float = -40.0
    flag_lufs_too_loud: float = -12.0


def dbfs_from_rms(rms: float) -> float:
    return 20.0 * math.log10(max(rms, EPS))


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def list_wav_paths(inp: str, recursive: bool) -> List[str]:
    # support directory, glob pattern, or file path
    if os.path.isdir(inp):
        pattern = os.path.join(inp, "**", "*.wav") if recursive else os.path.join(inp, "*.wav")
        paths = glob.glob(pattern, recursive=recursive)
    else:
        # If it's an exact file path, glob will still return it (if exists)
        paths = glob.glob(inp, recursive=recursive)
        if not paths and os.path.isfile(inp):
            paths = [inp]
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort()
    return paths


def smooth_mask(mask: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    """
    Enforce minimum speech and minimum silence lengths on a boolean mask.
    min_on/min_off are in frames.
    """
    if mask.size == 0:
        return mask

    m = mask.astype(np.uint8)
    diffs = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # Remove too-short speech runs
    for s, e in zip(starts, ends):
        if e - s < min_on:
            m[s:e] = 0

    # Recompute
    diffs = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # Fill too-short silence gaps
    mc = 1 - m
    diffs0 = np.diff(np.concatenate(([0], mc, [0])))
    s0 = np.where(diffs0 == 1)[0]
    e0 = np.where(diffs0 == -1)[0]
    for s, e in zip(s0, e0):
        if e - s < min_off:
            m[s:e] = 1

    return m.astype(bool)


def segments_from_mask(mask: np.ndarray, hop_s: float, frame_s: float) -> List[Dict[str, float]]:
    """Convert boolean mask into segments."""
    if mask.size == 0:
        return []
    m = mask.astype(np.uint8)
    diffs = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    segs = []
    for s, e in zip(starts, ends):
        t0 = s * hop_s
        t1 = e * hop_s + frame_s
        segs.append({"start": float(t0), "end": float(t1), "dur": float(t1 - t0)})
    return segs


def k_weighting_filter(sr: int):
    """Approx K-weighting filter (practical proxy)."""
    if not HAS_SCIPY:
        return None
    hp = signal.iirfilter(N=2, Wn=60.0, btype="highpass", ftype="butter", fs=sr, output="sos")
    hs = signal.iirfilter(N=2, Wn=1500.0, btype="highpass", ftype="butter", fs=sr, output="sos")
    return np.vstack([hp, hs])


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Approx integrated loudness (LUFS). Returns NaN if scipy missing."""
    if not HAS_SCIPY:
        return float("nan")
    x = x.reshape(-1).astype(np.float64, copy=False)

    sos = k_weighting_filter(sr)
    xw = signal.sosfilt(sos, x) if sos is not None else x

    block_len = int(round(0.400 * sr))
    step = int(round(0.100 * sr))
    if block_len <= 0 or step <= 0 or xw.size < block_len:
        return float("nan")

    ms = []
    for start in range(0, xw.size - block_len + 1, step):
        blk = xw[start:start + block_len]
        ms.append(float(np.mean(blk * blk)))
    ms = np.array(ms, dtype=np.float64)
    if ms.size == 0:
        return float("nan")

    l = -0.691 + 10.0 * np.log10(np.maximum(ms, EPS))

    gate_abs = -70.0
    keep = l > gate_abs
    if not np.any(keep):
        return float("nan")

    ungated = -0.691 + 10.0 * math.log10(max(float(np.mean(ms[keep])), EPS))
    gate_rel = ungated - 10.0
    keep2 = l > max(gate_abs, gate_rel)
    if not np.any(keep2):
        return float("nan")

    integ = -0.691 + 10.0 * math.log10(max(float(np.mean(ms[keep2])), EPS))
    return float(integ)


def hum_ratio_from_signal(x: np.ndarray, sr: int, hum_hz: float, band: float, max_hz: float) -> float:
    """Hum ratio proxy: energy around hum harmonics / energy under max_hz."""
    x = x.reshape(-1)
    if x.size < sr:
        return float("nan")

    if HAS_SCIPY:
        f, pxx = signal.welch(x, fs=sr, nperseg=min(8192, x.size))
    else:
        n = min(65536, x.size)
        w = np.hanning(n)
        X = np.fft.rfft(x[:n] * w)
        pxx = (np.abs(X) ** 2)
        f = np.fft.rfftfreq(n, 1.0 / sr)

    total_sel = (f > 0) & (f <= max_hz)
    total = float(np.sum(pxx[total_sel]))
    if total <= 0:
        return float("nan")

    hsum = 0.0
    k = 1
    while k * hum_hz <= max_hz:
        center = k * hum_hz
        sel = (f >= center - band) & (f <= center + band)
        hsum += float(np.sum(pxx[sel]))
        k += 1

    return float(hsum / total)


def echo_proxy_envelope_autocorr(x: np.ndarray, sr: int, cfg: QCConfig) -> float:
    """Echo proxy via envelope autocorr peak (20-200ms)."""
    x = x.reshape(-1)
    nmax = int(min(x.size, cfg.echo_probe_seconds * sr))
    if nmax < sr:
        return float("nan")

    x = x[:nmax]
    env = np.abs(x).astype(np.float64)

    target = cfg.echo_env_hz
    step = max(1, int(round(sr / target)))
    env_ds = env[::step]
    if env_ds.size < 500:
        return float("nan")

    env_ds = env_ds - np.mean(env_ds)
    env_ds = env_ds / (np.std(env_ds) + EPS)

    n = int(2 ** math.ceil(math.log2(env_ds.size * 2)))
    F = np.fft.rfft(env_ds, n=n)
    ac = np.fft.irfft(F * np.conj(F), n=n)[:env_ds.size]
    ac = ac / (ac[0] + EPS)

    lag_min = int(round((cfg.echo_lag_min_ms / 1000.0) * target))
    lag_max = int(round((cfg.echo_lag_max_ms / 1000.0) * target))
    lag_min = max(1, lag_min)
    lag_max = min(lag_max, ac.size - 1)
    if lag_max <= lag_min:
        return float("nan")

    return float(np.max(ac[lag_min:lag_max + 1]))


def analyze_wav(path: str, cfg: QCConfig, compute_spectral: bool = True) -> Dict[str, Any]:
    if not HAS_SF:
        raise RuntimeError("Missing dependency: soundfile. Install with `pip install soundfile`")

    # ===== PASS 1: streaming stats + frame energy per channel =====
    with sf.SoundFile(path) as f:
        sr = int(f.samplerate)
        ch = int(f.channels)
        fmt = str(getattr(f, "format", ""))
        subtype = str(getattr(f, "subtype", ""))

        frame_len = int(round(cfg.frame_ms / 1000.0 * sr))
        hop_len = int(round(cfg.hop_ms / 1000.0 * sr))
        frame_s = frame_len / sr
        hop_s = hop_len / sr

        if frame_len <= 0 or hop_len <= 0:
            raise ValueError("Invalid frame/hop configuration")

        total_samples = 0
        peak = np.zeros(ch, dtype=np.float64)
        sumsq = np.zeros(ch, dtype=np.float64)
        clip_count = np.zeros(ch, dtype=np.int64)
        zero_count = np.zeros(ch, dtype=np.int64)

        longest_zero_run = np.zeros(ch, dtype=np.int64)

        # stereo corr accumulators
        sum_xy = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0

        frame_db_list = [[] for _ in range(ch)]

        global_read = 0
        next_frame_start = 0
        buf = np.zeros((0, ch), dtype=np.float32)

        blocksize = sr * 10

        while True:
            block = f.read(blocksize, dtype="float32", always_2d=True)
            if block.size == 0:
                break

            n = block.shape[0]
            total_samples += n

            absb = np.abs(block)
            peak = np.maximum(peak, np.max(absb, axis=0))
            sumsq += np.sum(block * block, axis=0)

            clip_count += np.sum(absb >= cfg.clip_threshold, axis=0)
            zmask = absb <= cfg.near_zero_threshold
            zero_count += np.sum(zmask, axis=0)

            # longest near-zero run per channel in this block
            for c in range(ch):
                zm = zmask[:, c]
                run = 0
                best = int(longest_zero_run[c])
                for v in zm:
                    if v:
                        run += 1
                        if run > best:
                            best = run
                    else:
                        run = 0
                longest_zero_run[c] = best

            # stereo correlation (ch0,ch1)
            if ch >= 2:
                x = block[:, 0].astype(np.float64, copy=False)
                y = block[:, 1].astype(np.float64, copy=False)
                sum_xy += float(np.sum(x * y))
                sum_x += float(np.sum(x))
                sum_y += float(np.sum(y))
                sum_x2 += float(np.sum(x * x))
                sum_y2 += float(np.sum(y * y))

            # frame energy extraction (buffered)
            buf = np.vstack([buf, block])

            while (global_read + buf.shape[0]) - next_frame_start >= frame_len:
                local_next = next_frame_start - global_read

                x2 = buf * buf
                cs = np.cumsum(x2, axis=0, dtype=np.float64)
                cs0 = np.vstack([np.zeros((1, ch), dtype=np.float64), cs])
                window_sums = cs0[frame_len:] - cs0[:-frame_len]  # (L-frame_len+1, ch)

                if local_next >= window_sums.shape[0]:
                    break

                idx = np.arange(local_next, window_sums.shape[0], hop_len, dtype=np.int64)
                if idx.size == 0:
                    break

                sums_sel = window_sums[idx, :]
                rms_f = np.sqrt(np.maximum(sums_sel / frame_len, EPS))
                db = 20.0 * np.log10(rms_f + EPS)

                for c in range(ch):
                    frame_db_list[c].extend(db[:, c].tolist())

                next_frame_start += int(idx.size) * hop_len

                drop = next_frame_start - global_read
                if drop > 0:
                    buf = buf[drop:, :]
                    global_read += drop
                else:
                    break

        duration = total_samples / sr if sr > 0 else 0.0

    frame_db = [np.array(lst, dtype=np.float32) for lst in frame_db_list]
    n_frames = int(frame_db[0].size) if ch >= 1 else 0

    rms = np.sqrt(np.maximum(sumsq / max(total_samples, 1), EPS))
    peak_dbfs = np.array([dbfs_from_rms(p) for p in peak], dtype=np.float64)
    rms_dbfs = np.array([dbfs_from_rms(r) for r in rms], dtype=np.float64)
    crest_db = peak_dbfs - rms_dbfs

    clipping_pct = (clip_count / max(total_samples, 1)) * 100.0
    zero_pct = (zero_count / max(total_samples, 1)) * 100.0
    longest_zero_run_ms = (longest_zero_run / sr) * 1000.0

    noise_floor_db = np.array(
        [float(np.percentile(fd, 10)) if fd.size else float("nan") for fd in frame_db],
        dtype=np.float64
    )

    hop_s = (int(round(cfg.hop_ms / 1000.0 * sr)) / sr) if sr > 0 else 0.0
    frame_s = (int(round(cfg.frame_ms / 1000.0 * sr)) / sr) if sr > 0 else 0.0

    min_on = int(round(cfg.min_speech_ms / max(cfg.hop_ms, 1e-6)))
    min_off = int(round(cfg.min_silence_ms / max(cfg.hop_ms, 1e-6)))

    vad_masks = []
    vad_segs = []
    speech_ratio = []
    max_silence = []
    num_segs = []
    avg_seg = []
    avg_sil = []
    initial_silence = []
    est_snr_db = []

    for c in range(ch):
        fd = frame_db[c]
        if fd.size == 0:
            mask = np.zeros((0,), dtype=bool)
        else:
            thr = max(noise_floor_db[c] + cfg.vad_margin_db, cfg.vad_abs_floor_db)
            mask = fd >= thr
            mask = smooth_mask(mask, min_on=min_on, min_off=min_off)

        vad_masks.append(mask)
        segs = segments_from_mask(mask, hop_s=hop_s, frame_s=frame_s)
        vad_segs.append(segs)

        sp_dur = float(np.sum(mask) * hop_s) if mask.size else 0.0
        speech_ratio.append(sp_dur / duration if duration > 0 else 0.0)

        sil_mask = ~mask
        sil_segs = segments_from_mask(sil_mask, hop_s=hop_s, frame_s=frame_s)
        max_silence.append(max((s["dur"] for s in sil_segs), default=0.0))
        num_segs.append(len(segs))
        avg_seg.append(safe_mean(np.array([s["dur"] for s in segs], dtype=np.float32)))
        avg_sil.append(safe_mean(np.array([s["dur"] for s in sil_segs], dtype=np.float32)))

        if segs:
            initial_silence.append(float(segs[0]["start"]))
        else:
            initial_silence.append(float(duration))

        if fd.size:
            speech_fd = fd[mask] if np.any(mask) else np.array([], dtype=np.float32)
            noise_fd = fd[~mask] if np.any(~mask) else np.array([], dtype=np.float32)
            speech_rms_db = float(np.median(speech_fd)) if speech_fd.size else float("nan")
            noise_rms_db = float(np.median(noise_fd)) if noise_fd.size else float("nan")
            est_snr_db.append(
                float(speech_rms_db - noise_rms_db)
                if np.isfinite(speech_rms_db) and np.isfinite(noise_rms_db)
                else float("nan")
            )
        else:
            est_snr_db.append(float("nan"))

    vad_masks = np.vstack(vad_masks).T if ch > 0 else np.zeros((0, 0), dtype=bool)  # (frames, ch)
    speech_any = np.any(vad_masks, axis=1) if vad_masks.size else np.zeros((0,), dtype=bool)
    speech_ratio_any = float(np.sum(speech_any) * hop_s / duration) if duration > 0 and speech_any.size else 0.0
    segs_any = segments_from_mask(speech_any, hop_s=hop_s, frame_s=frame_s)

    overlap_ratio = float("nan")
    if duration > 0 and ch >= 2 and vad_masks.shape[0] > 0:
        overlap_ratio = float(np.sum(np.all(vad_masks[:, :2], axis=1)) * hop_s / duration)

    channel_corr = float("nan")
    crosstalk_db = float("nan")
    if ch >= 2 and total_samples > 0:
        n = float(total_samples)
        cov = (sum_xy - (sum_x * sum_y) / n)
        varx = (sum_x2 - (sum_x * sum_x) / n)
        vary = (sum_y2 - (sum_y * sum_y) / n)
        denom = math.sqrt(max(varx, EPS) * max(vary, EPS))
        channel_corr = float(cov / denom) if denom > 0 else float("nan")

        if n_frames > 0:
            m0 = vad_masks[:, 0]
            m1 = vad_masks[:, 1]
            fd0 = frame_db[0]
            leak_frames = (~m0) & m1
            sil_frames = (~m0) & (~m1)
            if np.any(leak_frames) and np.any(sil_frames):
                leak = float(np.median(fd0[leak_frames]))
                sil = float(np.median(fd0[sil_frames]))
                crosstalk_db = float(leak - sil)

    speech_dropout_ratio = float("nan")
    if ch >= 1 and n_frames > 0:
        fd0 = frame_db[0]
        if speech_any.size == fd0.size and np.any(speech_any):
            nf = float(np.percentile(fd0, 10))
            near_nf = speech_any & (fd0 <= (nf + 3.0))
            speech_dropout_ratio = float(np.sum(near_nf) / max(np.sum(speech_any), 1))

    # ===== PASS 2: spectral + hum + echo + LUFS (mono mix) =====
    spectral = {
        "centroid_hz_speech": float("nan"),
        "flatness_speech": float("nan"),
        "rolloff95_hz_speech": float("nan"),
        "centroid_hz_noise": float("nan"),
        "flatness_noise": float("nan"),
        "rolloff95_hz_noise": float("nan"),
    }
    hum50 = float("nan")
    hum60 = float("nan")
    echo_proxy = float("nan")
    lufs_i = float("nan")

    # If you want to cap LUFS computation for memory, set max_lufs_seconds (e.g. 600.0)
    max_lufs_seconds = None  # None = full file

    with sf.SoundFile(path) as f:
        mono_for_hum: List[np.ndarray] = []
        mono_for_echo: List[np.ndarray] = []
        mono_for_lufs: List[np.ndarray] = []

        max_hum_seconds = 60.0
        hum_samples_max = int(max_hum_seconds * sr)
        echo_samples_max = int(cfg.echo_probe_seconds * sr)
        lufs_samples_max = int(max_lufs_seconds * sr) if max_lufs_seconds is not None else None

        hum_collected = 0
        echo_collected = 0
        lufs_collected = 0

        spec_len = int(round(cfg.spectral_fft_ms / 1000.0 * sr))
        spec_hop = int(round(cfg.spectral_hop_ms / 1000.0 * sr))
        if spec_len <= 0 or spec_hop <= 0:
            spec_len, spec_hop = 1024, 512

        hann = np.hanning(spec_len).astype(np.float64)

        def spec_acc():
            return {"n": 0, "cent_sum": 0.0, "flat_sum": 0.0, "roll_sum": 0.0}

        acc_sp = spec_acc()
        acc_ns = spec_acc()

        buf2 = np.zeros((0, ch), dtype=np.float32)
        global_pos = 0
        next_spec_start = 0

        blocksize = sr * 10
        while True:
            block = f.read(blocksize, dtype="float32", always_2d=True)
            if block.size == 0:
                break

            mono = np.mean(block, axis=1).astype(np.float32, copy=False)

            # ---- FIXED collection (no concatenate on empty list, no O(n^2) concat every loop) ----
            if hum_collected < hum_samples_max:
                take = min(mono.size, hum_samples_max - hum_collected)
                if take > 0:
                    mono_for_hum.append(mono[:take])
                    hum_collected += take

            if echo_collected < echo_samples_max:
                take = min(mono.size, echo_samples_max - echo_collected)
                if take > 0:
                    mono_for_echo.append(mono[:take])
                    echo_collected += take

            if lufs_samples_max is None:
                mono_for_lufs.append(mono)
            else:
                if lufs_collected < lufs_samples_max:
                    take = min(mono.size, lufs_samples_max - lufs_collected)
                    if take > 0:
                        mono_for_lufs.append(mono[:take])
                        lufs_collected += take

            # spectral (optional)
            if compute_spectral and duration > 0 and speech_any.size > 0:
                buf2 = np.vstack([buf2, block])
                while (global_pos + buf2.shape[0]) - next_spec_start >= spec_len:
                    local = next_spec_start - global_pos
                    frame = buf2[local:local + spec_len, :]
                    m = np.mean(frame, axis=1).astype(np.float64, copy=False)
                    m *= hann

                    t_center = (next_spec_start + (spec_len // 2)) / sr
                    vad_idx = int(t_center / max(hop_s, 1e-9))
                    is_speech = bool(speech_any[vad_idx]) if 0 <= vad_idx < speech_any.size else False

                    X = np.fft.rfft(m)
                    p = (np.abs(X) ** 2) + EPS
                    freqs = np.fft.rfftfreq(spec_len, 1.0 / sr)

                    p_sum = float(np.sum(p))
                    if p_sum > 0:
                        centroid = float(np.sum(freqs * p) / p_sum)
                        flatness = float(np.exp(np.mean(np.log(p))) / (np.mean(p)))
                        cumsum = np.cumsum(p)
                        thr = cfg.rolloff_pct * cumsum[-1]
                        ridx = int(np.searchsorted(cumsum, thr))
                        rolloff = float(freqs[min(ridx, freqs.size - 1)])
                    else:
                        centroid = flatness = rolloff = float("nan")

                    acc = acc_sp if is_speech else acc_ns
                    acc["n"] += 1
                    acc["cent_sum"] += centroid
                    acc["flat_sum"] += flatness
                    acc["roll_sum"] += rolloff

                    next_spec_start += spec_hop
                    drop = next_spec_start - global_pos
                    if drop > 0:
                        buf2 = buf2[drop:, :]
                        global_pos += drop
                    else:
                        break

        if compute_spectral and acc_sp["n"] > 0:
            spectral["centroid_hz_speech"] = acc_sp["cent_sum"] / acc_sp["n"]
            spectral["flatness_speech"] = acc_sp["flat_sum"] / acc_sp["n"]
            spectral["rolloff95_hz_speech"] = acc_sp["roll_sum"] / acc_sp["n"]
        if compute_spectral and acc_ns["n"] > 0:
            spectral["centroid_hz_noise"] = acc_ns["cent_sum"] / acc_ns["n"]
            spectral["flatness_noise"] = acc_ns["flat_sum"] / acc_ns["n"]
            spectral["rolloff95_hz_noise"] = acc_ns["roll_sum"] / acc_ns["n"]

    mono_h = np.concatenate(mono_for_hum) if mono_for_hum else np.zeros((0,), dtype=np.float32)
    mono_e = np.concatenate(mono_for_echo) if mono_for_echo else np.zeros((0,), dtype=np.float32)
    mono_l = np.concatenate(mono_for_lufs) if mono_for_lufs else np.zeros((0,), dtype=np.float32)

    if mono_h.size:
        hum50 = hum_ratio_from_signal(mono_h, sr, hum_hz=50.0, band=cfg.hum_band_hz, max_hz=cfg.hum_max_harmonic_hz)
        hum60 = hum_ratio_from_signal(mono_h, sr, hum_hz=60.0, band=cfg.hum_band_hz, max_hz=cfg.hum_max_harmonic_hz)

    if mono_e.size:
        echo_proxy = echo_proxy_envelope_autocorr(mono_e, sr, cfg)

    if mono_l.size:
        lufs_i = integrated_lufs(mono_l.astype(np.float64), sr)

    # ===== FLAGS =====
    flags = []
    if sr not in (8000, 16000, 48000, 44100):
        flags.append("unusual_sample_rate")

    if speech_ratio_any < cfg.flag_min_speech_ratio:
        flags.append("low_speech_ratio")

    best_snr = np.nanmax(np.array(est_snr_db, dtype=np.float64)) if len(est_snr_db) else float("nan")
    if np.isfinite(best_snr) and best_snr < cfg.flag_min_snr_db:
        flags.append("low_snr")

    if float(np.nanmax(clipping_pct)) > cfg.flag_max_clipping_pct:
        flags.append("clipping")

    if float(np.nanmax(longest_zero_run_ms)) > cfg.flag_longest_zero_run_ms:
        flags.append("dropouts_or_dead_samples")

    if np.isfinite(overlap_ratio) and overlap_ratio > cfg.flag_high_overlap_ratio:
        flags.append("high_overlap_double_talk")

    if np.isfinite(lufs_i):
        if lufs_i < cfg.flag_lufs_too_quiet:
            flags.append("too_quiet_lufs")
        if lufs_i > cfg.flag_lufs_too_loud:
            flags.append("too_loud_lufs")

    out: Dict[str, Any] = {
        "file_path": path,
        "file_name": os.path.basename(path),
        "format": fmt,
        "subtype": subtype,
        "sample_rate": sr,
        "channels": ch,
        "duration_sec": float(duration),

        "peak_dbfs_max": float(np.max(peak_dbfs)) if peak_dbfs.size else float("nan"),
        "rms_dbfs_mean": float(np.mean(rms_dbfs)) if rms_dbfs.size else float("nan"),
        "crest_db_mean": float(np.mean(crest_db)) if crest_db.size else float("nan"),
        "lufs_i": float(lufs_i),

        "clipping_pct_max": float(np.max(clipping_pct)) if clipping_pct.size else float("nan"),
        "zero_pct_max": float(np.max(zero_pct)) if zero_pct.size else float("nan"),
        "longest_zero_run_ms_max": float(np.max(longest_zero_run_ms)) if longest_zero_run_ms.size else float("nan"),

        "noise_floor_dbfs_mean": float(np.nanmean(noise_floor_db)) if noise_floor_db.size else float("nan"),
        "est_snr_db_best": float(best_snr),

        "speech_ratio_any": float(speech_ratio_any),
        "num_speech_segments_any": int(len(segs_any)),
        "avg_speech_segment_s_any": safe_mean(np.array([s["dur"] for s in segs_any], dtype=np.float32)),
        "max_silence_s_ch0": float(max_silence[0]) if max_silence else float("nan"),
        "initial_silence_s_ch0": float(initial_silence[0]) if initial_silence else float("nan"),

        "overlap_ratio": float(overlap_ratio),
        "channel_corr_01": float(channel_corr),
        "crosstalk_db_01": float(crosstalk_db),

        "speech_dropout_ratio_proxy": float(speech_dropout_ratio),

        "hum50_ratio": float(hum50),
        "hum60_ratio": float(hum60),
        "echo_proxy_corr": float(echo_proxy),

        "spectral_centroid_hz_speech": float(spectral["centroid_hz_speech"]),
        "spectral_flatness_speech": float(spectral["flatness_speech"]),
        "spectral_rolloff95_hz_speech": float(spectral["rolloff95_hz_speech"]),
        "spectral_centroid_hz_noise": float(spectral["centroid_hz_noise"]),
        "spectral_flatness_noise": float(spectral["flatness_noise"]),
        "spectral_rolloff95_hz_noise": float(spectral["rolloff95_hz_noise"]),

        "flags": "|".join(flags),
    }

    for c in range(ch):
        out[f"peak_dbfs_ch{c}"] = float(peak_dbfs[c])
        out[f"rms_dbfs_ch{c}"] = float(rms_dbfs[c])
        out[f"clipping_pct_ch{c}"] = float(clipping_pct[c])
        out[f"noise_floor_dbfs_ch{c}"] = float(noise_floor_db[c])
        out[f"speech_ratio_ch{c}"] = float(speech_ratio[c])
        out[f"est_snr_db_ch{c}"] = float(est_snr_db[c])
        out[f"num_speech_segments_ch{c}"] = int(num_segs[c])
        out[f"avg_speech_segment_s_ch{c}"] = float(avg_seg[c]) if np.isfinite(avg_seg[c]) else float("nan")
        out[f"max_silence_s_ch{c}"] = float(max_silence[c])
        out[f"initial_silence_s_ch{c}"] = float(initial_silence[c])
        out[f"longest_zero_run_ms_ch{c}"] = float(longest_zero_run_ms[c])

    out["_segments"] = {
        "frame_ms": cfg.frame_ms,
        "hop_ms": cfg.hop_ms,
        "speech_any": segs_any,
        "speech_per_channel": vad_segs,
    }

    return out


def write_csv(rows: List[Dict[str, Any]], out_csv: str):
    if not rows:
        return
    flat_rows = []
    for r in rows:
        rr = dict(r)
        rr.pop("_segments", None)
        flat_rows.append(rr)

    fieldnames = sorted(set().union(*[set(r.keys()) for r in flat_rows]))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)


def write_json_segments(rows: List[Dict[str, Any]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for r in rows:
        seg = r.get("_segments", {})
        payload = {
            "file_path": r.get("file_path"),
            "sample_rate": r.get("sample_rate"),
            "channels": r.get("channels"),
            "duration_sec": r.get("duration_sec"),
            "speech_ratio_any": r.get("speech_ratio_any"),
            "flags": r.get("flags"),
            "segments": seg,
        }
        base = os.path.basename(str(r.get("file_path", "audio.wav")))
        name = os.path.splitext(base)[0] + ".json"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory of wavs OR glob pattern OR file path")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_json_dir", default="", help="Optional: directory to write per-file JSON segments")
    ap.add_argument("--recursive", action="store_true", help="Recursive search when --input is a directory or glob **")
    ap.add_argument("--no_spectral", action="store_true", help="Disable spectral features (faster)")
    args = ap.parse_args()

    cfg = QCConfig()

    if not HAS_SF:
        print("ERROR: soundfile not installed. Run: pip install soundfile", file=sys.stderr)
        sys.exit(1)

    paths = list_wav_paths(args.input, recursive=args.recursive)
    if not paths:
        print("No wav files found.", file=sys.stderr)
        sys.exit(2)

    rows = []
    for i, p in enumerate(paths, 1):
        try:
            r = analyze_wav(p, cfg, compute_spectral=(not args.no_spectral))
            rows.append(r)
            print(f"[{i}/{len(paths)}] OK: {os.path.basename(p)} | flags={r.get('flags','')}")
        except Exception as e:
            print(f"[{i}/{len(paths)}] FAIL: {p} | {e}", file=sys.stderr)

    write_csv(rows, args.out_csv)
    print(f"\nWrote CSV: {args.out_csv}")

    if args.out_json_dir:
        write_json_segments(rows, args.out_json_dir)
        print(f"Wrote JSON segments to: {args.out_json_dir}")


if __name__ == "__main__":
    main()
