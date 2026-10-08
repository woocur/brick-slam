"""Find a song's tempo and first downbeat, for charting a recorded song in Beat Slam.

usage: python3 tools/beatgrid.py song.mp3 [--bpm 128]
prints BPM, offset (seconds of the first downbeat) and length in bars; needs ffmpeg and numpy.
"""
import subprocess, sys
import numpy as np

SR, HOP = 11025, 128
LAG = 0.055   # the flux peaks this much before the audible attack (measured on songs with a known grid)

def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def onsets(x):
    # spectral flux over a short-time spectrum; also a low band (kick) flux for finding the downbeat
    n = 512
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n // 2, n // 2)), n)[::HOP] * np.hanning(n)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 10)
    flux = np.maximum(0, np.diff(mag, axis=0, prepend=mag[:1]))
    low = flux[:, : int(150 / (SR / n)) + 1].sum(1)
    env = flux.sum(1)
    env -= np.convolve(env, np.ones(16) / 16, 'same')
    return np.maximum(env, 0), low

def tempo(env, lo=70, hi=190):
    fps = SR / HOP
    ac = np.correlate(env, env, 'full')[len(env) - 1:]
    best, score = 0, -1
    for bpm in np.arange(lo, hi, 0.05):
        lag = 60 * fps / bpm
        s = sum(np.interp(lag * k, np.arange(len(ac)), ac) / k ** 0.5 for k in (1, 2, 4))   # beat, half bar, bar
        if s > score: best, score = bpm, s
    return best

def phase(env, bpm, low):
    fps = SR / HOP; spb = 60 * fps / bpm
    t = np.arange(len(env))
    grid = lambda ph, step: np.interp(np.arange(ph, len(env) - 1, step), t, env).sum()
    beat = max(np.arange(0, spb, 0.25), key=lambda ph: grid(ph, spb))
    lowsum = lambda ph: np.interp(np.arange(ph, len(env) - 1, spb * 4), t, low).sum()
    down = max((beat + k * spb for k in range(4)), key=lowsum)
    # first downbeat that has real sound around it
    thr = 0.15 * env.max()
    while down - 4 * spb >= 0 and env[int(down - 4 * spb)] > thr * 0.2: down -= 4 * spb
    while down < len(env) and env[int(down):int(down + 4 * spb)].max(initial=0) < thr: down += 4 * spb
    return down / fps + LAG

if __name__ == '__main__':
    x = load(sys.argv[1])
    env, low = onsets(x)
    if '--bpm' in sys.argv: bpm = float(sys.argv[sys.argv.index('--bpm') + 1])
    else:
        bpm = tempo(env)
        if abs(bpm - round(bpm)) < 0.6: bpm = float(round(bpm))   # produced songs are almost always a whole BPM
    off = phase(env, bpm, low)
    bars = (len(x) / SR - off) / (240 / bpm)
    print(f'bpm {bpm}  offset {off:.3f}s  length {len(x) / SR:.1f}s  bars after offset {bars:.1f}')
