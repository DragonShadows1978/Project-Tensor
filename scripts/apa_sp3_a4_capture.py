"""Sequential layer-range captures with bounded diagnostic rows.

Prior art: checkpoint/restart and NumPy memory maps (standard systems practice);
OpenBMB MiniCPM3 (2024) layer loop reused verbatim in substance. New: SP3
range/activation receipts. Existing APA/BLASST selection arithmetic is reused,
not redesigned. Every observational replay must match native output bitwise.
"""
import gc
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from apa_sp3_common import ART, Red, publish, read, sha, require_pass
from apa_sp3_model import Model, fraction_summary


def stat_pin(path):
    s = Path(path).stat()
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def save_array(path, array):
    with Path(path).open('xb') as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    return dict(sha256=sha(path), stat=stat_pin(path))


def check_file(path, pin, rehash=True):
    if stat_pin(path) != pin['stat'] or (rehash and sha(path) != pin['sha256']):
        raise Red(f'capture file changed: {path}')


class LayerWriter:
    def __init__(self, owner, q, k, kq, causal, path_kind):
        self.owner = owner
        B, H, L, D = q.shape
        S = k.shape[2]
        if (B, H, D) != (1, 40, 96) or L != S or not causal:
            raise Red('range capture requires full square causal MiniCPM3 prefill')
        self.S, self.cursor, self.selected = S, 0, 0
        self.meta = dict(layer=owner.layer, pairs=40*S*(S+1)//2,
                         path=path_kind, shapes={'q': list(q.shape), 'k': list(k.shape)},
                         causal=True, arm=owner.arm, bits=owner.bits, delta=owner.delta,
                         scale=96**-.5, dtype=q.dtype, native_bulk_chunks=None,
                         token_sha256=owner.capture_token_sha,
                         capture_id=owner.capture_id,
                         mask_evidence='bounded native diagnostic replay, bit parity against uninstrumented full-layer output')
        self.p = Path(owner.capture_dir)/f'layer{owner.layer:02d}'
        self.p.mkdir(exist_ok=False)
        self.pins = {}
        for name, t in [('q', q), ('k', k), ('kq', kq)]:
            self.pins[name+'.npy'] = save_array(self.p/(name+'.npy'), t.float().numpy())
        self.mask = np.lib.format.open_memmap(self.p/'selected.pack.npy', mode='w+',
                    dtype=np.uint8, shape=(1, 40, S, (S+7)//8))

    def write(self, lo, mask):
        # Prior art: packed boolean bitmap and row streaming, standard storage.
        # New: preserve original little-endian capture layout without S^2 RAM.
        mn = np.asarray(mask, dtype=np.uint8)
        n, keys = mn.shape[2:]
        if lo != self.cursor or mn.shape[:2] != (1, 40) or lo+n > self.S or keys > self.S:
            raise Red('missing/duplicate/out-of-order diagnostic rows')
        valid = np.arange(keys)[None, None, None, :] <= np.arange(lo, lo+n)[None, None, :, None]
        if np.any(mn > 1) or np.any(mn & ~valid):
            raise Red('nonbinary/causally invalid diagnostic mask')
        packed = np.packbits(mn, axis=-1, bitorder='little')
        self.mask[:, :, lo:lo+n] = 0
        self.mask[:, :, lo:lo+n, :packed.shape[-1]] = packed
        self.selected += int(mn.sum(dtype=np.int64))
        self.cursor += n

    def finish(self, bulk_pins=None):
        if self.cursor != self.S:
            raise Red('incomplete diagnostic mask')
        self.mask.flush()
        del self.mask
        path = self.p/'selected.pack.npy'
        self.pins[path.name] = dict(sha256=sha(path), stat=stat_pin(path))
        self.meta.update(selected=self.selected, fraction=self.selected/self.meta['pairs'],
                         native_bulk_chunks=bulk_pins,
                         files={p: v['sha256'] for p, v in self.pins.items()}, file_pins=self.pins)
        publish(self.p/'capture.json', self.meta)
        self.owner.rows.append({k: self.meta[k] for k in ('layer', 'selected', 'pairs', 'fraction', 'path')})


class RangeModel(Model):
    def blend(self, q, k, kq, v, group, scale, z, causal, blk):
        if not self.observe:
            return self.original_blend(q, k, kq, v, group, scale, z, causal, blk)
        if group != 1:
            raise Red('range capture is MHA only')
        out = self.original_blend(q, k, kq, v, group, scale, z, causal, blk)
        writer = LayerWriter(self, q, k, kq, causal, 'B_native_blend')
        bulk_pins = []
        tc = self.tc
        for lo in range(0, q.shape[2], blk):
            n = min(blk, q.shape[2]-lo)
            qi = q.slice(2, lo, n)
            bulk = tc.matmul(qi, kq.transpose(-2, -1))*scale
            bulk_pins.append(dict(row0=lo, length=n, sha256=hashlib.sha256(bulk.float().numpy().tobytes()).hexdigest()))
            rank = tc.matmul(qi, k.transpose(-2, -1))*scale
            w, mask = self.diag.blend(bulk, rank, z, q.shape[2], lo)
            replay = tc.matmul(w, v)
            if not np.array_equal(out.slice(2, lo, n).numpy(), replay.numpy()):
                raise Red('B ranged diagnostic changes native output')
            writer.write(lo, mask.numpy())
            del qi, bulk, rank, w, mask, replay
        writer.finish(bulk_pins)
        return out

    def selective(self, q, k, kq, v, scale, z, causal=False):
        if not self.observe:
            return super().selective(q, k, kq, v, scale, z, causal)
        if self.arm not in ('B', 'C', 'D', 'E') or not causal or q.shape[2] != k.shape[2]:
            raise Red('unexpected ranged SP capture path')
        # Prior art: ordinary causal prefix restriction: a query cannot depend
        # on future keys. Reuse native prefill kernel, never its L=1 split path.
        # New: bounded replay with exact native-output acceptance, not a proof.
        native = self.tc._C.apa_selective_attention_sp
        out = (self.original_selective(q, k, kq, v, scale, z, True) if self.arm == 'B'
               else native(q, k, kq, v, scale, self.delta, True, None, False))
        writer = LayerWriter(self, q, k, kq, True, 'B_native_fused' if self.arm == 'B' else 'SP_prefill')
        for lo in range(0, q.shape[2], 128):
            n = min(128, q.shape[2]-lo)
            if n == 1:
                raise Red('diagnostic replay must stay on prefill kernel')
            end = lo+n
            args = (q.slice(2, lo, n), k.slice(2, 0, end), kq.slice(2, 0, end), v.slice(2, 0, end))
            if self.arm == 'B':
                other, mask = self.diag.selective(*args, scale, z, True)
            else:
                other, mask = native(*args, scale, self.delta, True, None, True)
            if not np.array_equal(out.slice(2, lo, n).numpy(), other.numpy()):
                raise Red('SP ranged diagnostic changes native output')
            writer.write(lo, mask.numpy())
            del other, mask
        writer.finish()
        return out


def advance_layers(model, hidden, lo, hi):
    """Same MiniCPM3 block API as the pinned full forward; no cache between jobs."""
    for i in range(lo, hi):
        hidden, kv = model.layers[i](hidden, model.rope_cos, model.rope_sin, 0, None)
        del kv  # teacher-forced full prefill needs no retained per-layer KV
    return hidden


def required_space(S, lo=0):
    # Prior art: standard array-size accounting; unchanged registered layout.
    return int((62-lo)*(3*40*S*96*4 + 40*S*((S+7)//8))*1.15 + S*2560*4)


def capture_range(cell, model, ids, delta):
    S, lo, hi = cell['S'], cell['layer_start'], cell['layer_stop']
    dest = ART/'captures'/f"b{cell['bits']}_{cell['arm']}_{S}"
    # Disk rail accounts for ALL layers, not merely the first range. Existing
    # completed layer files consume disk already; require only remaining bytes.
    required = required_space(S, lo)
    if shutil.disk_usage(ART).free < required:
        raise Red(f'CAPTURE_DISK_OOM: need {required} free bytes')
    model.set(cell['arm'], cell['bits'], delta, observe=False)
    model.model.extend_rope(S)
    model.capture_dir, model.capture_id = dest, cell['capture_id']
    model.capture_token_sha = hashlib.sha256(ids[:S].astype('<i8').tobytes()).hexdigest()
    if lo == 0:
        dest.mkdir(parents=True, exist_ok=False)
    model.rows, model.observe = [], True
    tc = model.tc
    tc.synchronize()
    model.peak.reset()
    start = time.perf_counter()
    with tc.no_grad():
        if lo:
            prior = require_pass(cell['predecessor'])['result']
            if (prior['layer_stop'] != lo or prior['capture_id'] != cell['capture_id']
                    or prior['token_sha256'] != model.capture_token_sha):
                raise Red('wrong range predecessor')
            checkpoint = dest/prior['checkpoint']['path']
            check_file(checkpoint, prior['checkpoint'])
            a = np.load(checkpoint, allow_pickle=False)
            if list(a.shape) != [1, S, model.model.config.hidden_dim]:
                raise Red('hidden checkpoint shape')
            hidden = tc.tensor(a).astype(prior['checkpoint']['dtype'])
        else:
            idx = tc.tensor(np.ascontiguousarray(ids[:S][None].astype(np.int64)), dtype='int64')
            hidden = model.model.embed_tokens(idx)*model.model.config.scale_emb
        hidden = advance_layers(model.model, hidden, lo, hi)
        tc.synchronize()
        if sorted(r['layer'] for r in model.rows) != list(range(lo, hi)):
            raise Red('capture range missed/duplicated layers')
        checkpoints = dest/'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        checkpoint = checkpoints/f'hidden_{hi:02d}.npy'
        pin = dict(save_array(checkpoint, hidden.float().numpy()),
                   path=str(checkpoint.relative_to(dest)), dtype=hidden.dtype)
    layers = {f'layer{i:02d}/capture.json': sha(dest/f'layer{i:02d}/capture.json') for i in range(lo, hi)}
    return dict(evidence_class='kernel sweep / activation capture', capture_id=cell['capture_id'],
                layer_start=lo, layer_stop=hi, checkpoint=pin, layers=layers,
                token_sha256=model.capture_token_sha, delta=delta, bits=cell['bits'], arm=cell['arm'], S=S,
                refinement=fraction_summary(model.rows), wall_ms=(time.perf_counter()-start)*1000,
                **model.peak.result())


def aggregate(cell):
    dest = ART/'captures'/f"b{cell['bits']}_{cell['arm']}_{cell['S']}"
    layers, cursor, token_sha, rows = {}, 0, None, []
    for dep in cell['depends']:
        r = require_pass(dep)['result']
        if (r['layer_start'] != cursor or r['capture_id'] != cell['id']
                or any(r[k] != cell[k] for k in ('arm', 'bits', 'S'))):
            raise Red('capture range gap/overlap/wrong identity')
        if token_sha is not None and r['token_sha256'] != token_sha:
            raise Red('capture token stream mismatch')
        token_sha, cursor = r['token_sha256'], r['layer_stop']
        if set(r['layers']) != {f'layer{i:02d}/capture.json' for i in range(r['layer_start'], cursor)}:
            raise Red('range layer membership mismatch')
        for rel, digest in r['layers'].items():
            if rel in layers or sha(dest/rel) != digest:
                raise Red('duplicate/changed layer metadata')
            meta = read(dest/rel)
            i = int(Path(rel).parent.name[5:])
            if (meta['layer'] != i or meta['capture_id'] != cell['id']
                    or meta['token_sha256'] != token_sha
                    or any(meta[k] != cell[k] for k in ('arm', 'bits'))
                    or meta['shapes']['q'] != [1, 40, cell['S'], 96]
                    or meta['shapes']['k'] != [1, 40, cell['S'], 96]
                    or set(meta['files']) != {'q.npy', 'k.npy', 'kq.npy', 'selected.pack.npy'}
                    or set(meta['file_pins']) != set(meta['files'])):
                raise Red('layer metadata identity/geometry mismatch')
            for name, pin in meta['file_pins'].items():
                if pin['sha256'] != meta['files'][name]:
                    raise Red('capture hash disagreement')
                # Prior art: immutable stat identity after a completed hash,
                # same as existing SP3 weight identity. Avoid a 400GB re-read.
                check_file(dest/Path(rel).parent/name, pin, rehash=False)
            layers[rel] = digest
            rows.append({k: meta[k] for k in ('layer', 'selected', 'pairs', 'fraction', 'path')})
    expected = {f'layer{i:02d}/capture.json' for i in range(62)}
    if cursor != 62 or set(layers) != expected:
        raise Red('capture aggregation needs all 62 layers')
    seal = dict(evidence_class='kernel sweep / capture integrity', arm=cell['arm'], bits=cell['bits'],
                S=cell['S'], layers=layers, token_sha256=token_sha,
                set_sha256=hashlib.sha256(__import__('json').dumps(layers, sort_keys=True).encode()).hexdigest(),
                refinement=fraction_summary(rows),
                verification='all metadata rehashed; every array retains exact stat identity from range SHA256; margin workers rehash arrays')
    publish(dest/'capture_set.json', seal)
    return seal
