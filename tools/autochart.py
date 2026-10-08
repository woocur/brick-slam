"""Chart a recorded song from its sound: bricks drop on the 8th notes where the song actually hits.

usage: python3 tools/autochart.py song.mp3 beats.json "8:h,8:ht,16:tt,..." [--dump]
beats.json: file time of every beat (tools/beatgrid.py --map); the map gives bars and the bar patterns of
the normal chart (the same letters as the song's map in index.html), which set how many bricks each bar gets
per difficulty. Prints one string per difficulty (easy, normal, hard): 8 digits per bar, a digit = bricks on
that 8th note.
"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from beatgrid import load, SR, HOP, LAG

PAT = {'r': '00000000', 'h': '10001000', 't': '10001010', 'q': '10101010', 's': '10101011',
       'd': '20001010', 'x': '10111011', 'D': '20102010'}
DIFF = [{'t': 'h', 'q': 't', 's': 'q', 'd': 't'}, {}, {'h': 't', 't': 'q', 'q': 's', 's': 'x', 'd': 'D'}]

def bands(x):
    n = 1024
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n // 2, n // 2)), n)[::HOP] * np.hanning(n)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 10)
    flux = np.maximum(0, np.diff(mag, axis=0, prepend=mag[:1]))
    f = np.fft.rfftfreq(n, 1 / SR)
    out = []
    for lo, hi in ((30, 160), (160, 1800), (1800, 5500)):
        b = flux[:, (f >= lo) & (f < hi)].sum(1)
        b = b - np.convolve(b, np.ones(64) / 64, 'same')      # keep the attacks, drop the sustained level
        b = np.maximum(b, 0); out.append(b / (np.percentile(b, 95) + 1e-9))
    return out

def slot_scores(x, beats):
    low, mid, high = bands(x)
    fps = SR / HOP; w = int(0.03 * fps)
    times = []
    for k in range(len(beats) - 1):
        times += [beats[k], (beats[k] + beats[k + 1]) / 2]
    def at(env, t):
        c = int(round((t - LAG) * fps)); return env[max(0, c - w): c + w + 1].max(initial=0)
    # snare/claps/stabs (mid) carry the groove; the kick (low) is on every beat in dance music so it counts less
    return np.array([[0.6 * at(low, t), at(mid, t), 0.5 * at(high, t)] for t in times])

def chart(scores, bars, pats):
    sc = scores.sum(1)
    loud = np.array([sc[b * 8:(b + 1) * 8].mean() for b in range(bars)])
    out = []
    for d in range(3):
        s = ''
        for b in range(bars):
            p = DIFF[d].get(pats[b], pats[b]); want = sum(int(c) for c in PAT[p])
            seg = sc[b * 8:(b + 1) * 8].copy()
            seg[0] += 0.15                          # a small pull towards the bar's first beat
            seg[[0, 2, 4, 6]] += 0.05 * (2 - d)     # easier charts lean on the beats
            cnt = [0] * 8
            if want and loud[b] > 0.25 * np.median(loud):
                order = list(np.argsort(-seg))
                for i in order[:min(want, 8)]: cnt[i] += 1
                for i in order[:max(0, want - 8)]: cnt[i] += 1
                # don't leave three 8ths in a row on easy
                if d == 0:
                    for i in range(6):
                        if cnt[i] and cnt[i + 1] and cnt[i + 2]: cnt[i + 1] = 0
            s += ''.join(map(str, cnt))
        out.append(s)
    return out

if __name__ == '__main__':
    x = load(sys.argv[1]); beats = json.load(open(sys.argv[2]))
    pats = []
    for part in sys.argv[3].split(','):
        n, p = part.split(':'); pats += [p[i % len(p)] for i in range(int(n))]
    sc = slot_scores(x, beats)[:len(pats) * 8]
    res = chart(sc, len(pats), pats)
    if '--dump' in sys.argv:
        for b in range(len(pats)):
            print(b, ' '.join(r[b * 8:(b + 1) * 8] for r in res), ' '.join('%.1f' % v for v in sc[b * 8:(b + 1) * 8].sum(1)))
    print(json.dumps(res))
