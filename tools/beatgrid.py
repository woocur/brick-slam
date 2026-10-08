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
    onsets.mag = mag
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

def beatmap(env, bpm, off, dur, win=16, smooth=8, reach=0.06):
    """Beat times that follow small tempo drift (AI-generated songs wander by up to ~0.15 s).

    Walks the song beat by beat: folds the onset envelope over the surrounding `win` beats and looks for the best
    phase within `reach` seconds of the previous beat's phase (so it can't jump to the off-beat), then smooths."""
    fps = SR / HOP; spb = 60 / bpm; t = np.arange(len(env))
    n = int((dur - off) / spb) + 1
    dev, cur = [], 0.0
    for i in range(n):
        ks = np.arange(i - win // 2, i + win // 2)
        tt = off + ks * spb - LAG
        tt = tt[(tt > 0) & (tt < dur - 0.1)]
        if len(tt) >= 4:
            ds = np.arange(cur - reach, cur + reach, 0.002)
            prof = [np.interp((tt + d) * fps, t, env).mean() for d in ds]
            cur = ds[int(np.argmax(prof))]
        dev.append(cur)
    dev = np.array(dev); k = np.ones(smooth) / smooth
    dev = np.convolve(np.pad(dev, (smooth // 2, smooth - smooth // 2 - 1), mode='edge'), k, 'valid')
    return off + np.arange(n) * spb + dev

def align(beats, env, low, mag):
    """Put the beats on the kick (strongest low-band hit within a beat) and start the map on a downbeat
    (the beat where the sound changes most, i.e. where chords and phrases turn over)."""
    fps = SR / HOP; t = np.arange(len(low)); spb = np.median(np.diff(beats))
    ds = np.arange(-spb / 2, spb / 2, 0.002)
    prof = [np.interp((beats[2:-2] + d - LAG) * fps, t, low).mean() for d in ds]
    beats = beats + ds[int(np.argmax(prof))]
    def spec(a, b):
        a, b = int(max(0, a) * fps), int(max(0, b) * fps)
        return mag[a:b].mean(0) if b > a else mag[a:a + 1].mean(0)
    nov = np.zeros(len(beats))
    for i, b in enumerate(beats):
        u, v = spec(b - spb, b - 0.03), spec(b + 0.03, b + spb)
        nov[i] = 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    k = int(np.argmax([nov[j::4].mean() for j in range(4)]))
    return beats[k:]

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
    if '--map' in sys.argv:
        b = beatmap(env, bpm, off, len(x) / SR)
        b = align(b, env, low, onsets.mag)
        b = b[b > 0.05]
        while len(b) % 4: b = b[:-1]
        dev = (b - (off + np.arange(len(b)) * 60 / bpm)) * 1000
        print(f'beat map: {len(b)} beats, drift from a steady grid {dev.min():+.0f}..{dev.max():+.0f} ms')
        open(sys.argv[sys.argv.index('--map') + 1], 'w').write('[' + ','.join(f'{v:.3f}' for v in b) + ']')
