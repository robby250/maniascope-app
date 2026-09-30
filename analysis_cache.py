"""Bounded, exact selected-chart analyses shared by the viewer and its helper.

Only chart results are cached: never personal predictions, current form, scores
or preferences. Entries require the exact bytes, rate, SV mode and calculator
identity. JSON avoids executing cache contents; invalid/partial entries are
ordinary misses. Calibration code can continue using skill_calc.compute directly.
"""
import collections
import hashlib
import json
import math
import os

import paths
from pathlib import Path
import tempfile
import threading

import skill_calc

FORMAT = 1
MAX_ENTRY_BYTES = 8*1024*1024


def file_stamp(path):
    s = os.stat(path)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _nonfinite(value):
    raise ValueError('Non-finite cached value: '+value)


class AnalysisCache:
    def __init__(self, directory=None, memory_entries=64, memory_bytes=32*1024*1024,
                 disk_entries=512, disk_bytes=64*1024*1024):
        self.directory = Path(directory) if directory is not None else Path(
            os.environ.get('MANIASCOPE_ANALYSIS_CACHE', os.path.join(paths.CACHE, 'analysis'))).expanduser()
        self.memory_entries, self.memory_budget = memory_entries, memory_bytes
        self.disk_entries, self.disk_budget = disk_entries, disk_bytes
        self._memory, self._digests = collections.OrderedDict(), collections.OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def clear_memory(self):
        with self._lock:
            self._memory.clear(); self._digests.clear(); self._bytes = 0

    def _cached(self, key):
        with self._lock:
            hit = self._memory.get(key)
            if hit:
                self._memory.move_to_end(key)
                return hit[:2]

    def get_cached(self, path, rate=1., sv=True):
        """A current memory hit only; never parse, calculate or read disk cache."""
        import recdata
        path = os.fspath(path)
        try:
            key = (path, file_stamp(path), float(rate), bool(sv), recdata.calc_id())
        except OSError:
            return None
        return self._cached(key)

    def _remember(self, key, chart, result, encoded_size):
        # Conservative accounting includes the parsed note tuples retained by
        # this viewer entry, not just the small serialized result dictionaries.
        size = encoded_size*3+len(chart.notes)*160+len(chart.sv)*112+4096
        with self._lock:
            old = self._memory.pop(key, None)
            if old:
                self._bytes -= old[2]
            if size <= self.memory_budget and self.memory_entries:
                self._memory[key] = (chart, result, size)
                self._bytes += size
            while len(self._memory)>self.memory_entries or self._bytes>self.memory_budget:
                self._bytes -= self._memory.popitem(last=False)[1][2]

    def _prune(self):
        entries = []
        for path in self.directory.glob('*.json'):
            try:
                s = path.stat()
                entries.append((s.st_mtime_ns, path, s.st_size))
            except OSError:
                continue
        total = sum(size for _,_,size in entries)
        count = len(entries)
        for _,path,size in sorted(entries):
            if count<=self.disk_entries and total<=self.disk_budget:
                break
            try:
                path.unlink()
                total -= size; count -= 1
            except OSError:
                continue

    def analyze(self, path, rate=1., sv=True, *, calculator=None, cancelled=None):
        rate = float(rate)
        if rate <= 0 or not math.isfinite(rate):
            raise ValueError('rate must be positive and finite')
        skill_calc._check_cancelled(cancelled)
        path = os.fspath(path)
        if calculator is None:
            import recdata
            calculator = recdata.calc_id()
        stamp = file_stamp(path)
        key = (path, stamp, rate, bool(sv), calculator)
        hit = self._cached(key)
        if hit:
            return hit
        with self._lock:
            digest = self._digests.get((path, stamp))
        if digest is None:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if file_stamp(path) != stamp:
                raise OSError('Chart changed while being read; select it again')
            with self._lock:
                self._digests[path,stamp] = digest
                self._digests.move_to_end((path,stamp))
                while len(self._digests)>128:
                    self._digests.popitem(last=False)
        chart = skill_calc.parse_osu(path)
        tag = [FORMAT, calculator, digest, rate, bool(sv)]
        name = hashlib.sha256(json.dumps(tag,separators=(',',':')).encode()).hexdigest()+'.json'
        cached = self.directory/name
        try:
            if cached.stat().st_size > MAX_ENTRY_BYTES:
                raise ValueError('Oversized cache entry')
            raw = cached.read_bytes()
            entry = json.loads(raw, parse_constant=_nonfinite)
            result = entry['result']
            if (entry['key']!=tag or result['model']!=skill_calc.MODEL_VERSION or result['rate']!=rate
                    or result['keys']!=chart.keys or result['notes']!=len(chart.notes)
                    or not isinstance(result['scores'],dict) or not isinstance(result['timeline'],list)
                    or set(result['scores'])!={'overall',*skill_calc.skills(chart.keys)}
                    or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in result['scores'].values())):
                raise ValueError('Mismatched chart cache')
            # JSON changes float dictionary keys and tuples; restore the public
            # calculator result types exactly, including every archetype's parts.
            for view in (result, result.get('preunit_result', {})):
                if 'levels' in view:
                    view['levels'] = {float(k):v for k,v in view['levels'].items()}
            for field in ('archetypes', 'unit_card'):
                for archetype in result.get(field, ()):
                    archetype['parts'] = tuple(archetype['parts'])
            skill_calc._check_cancelled(cancelled)
            if file_stamp(path)!=stamp:
                raise OSError('Chart changed during cached analysis')
            self._remember(key,chart,result,len(raw))
            return chart,result
        except (OSError,ValueError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
            pass
        skill_calc._check_cancelled(cancelled)
        result = skill_calc.compute(chart,rate,sv,cancelled=cancelled)
        skill_calc._check_cancelled(cancelled)
        if file_stamp(path)!=stamp:
            raise OSError('Chart changed during analysis; select it again')
        try:
            raw = json.dumps({'key':tag,'result':result},separators=(',',':'),allow_nan=False).encode()
        except (ValueError,TypeError):
            return chart,result
        self._remember(key,chart,result,len(raw))
        if len(raw) <= MAX_ENTRY_BYTES and self.disk_entries:
            temp = None
            try:
                self.directory.mkdir(parents=True,exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.directory,suffix='.part',delete=False) as handle:
                    temp = handle.name
                    handle.write(raw)
                os.replace(temp,cached); temp = None
                self._prune()
            except OSError:
                pass  # A read-only/full cache must not make a playable chart fail.
            finally:
                if temp:
                    try: os.unlink(temp)
                    except OSError: pass
        return chart,result


selected = AnalysisCache()
