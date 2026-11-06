"""Dependency-free resource observations and conservative allocation guards."""
import ctypes
import os
from pathlib import Path
import shutil
import sys


def cpu_memory():
    """Windows working set and lifetime peak; Linux RSS where available.

    Working set is resident physical memory, not total committed virtual memory.
    The process lifetime peak cannot be reset between benchmark stages.
    """
    if sys.platform == 'win32':
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
        return {'rss_bytes': counters.WorkingSetSize,
                'lifetime_peak_rss_bytes': counters.PeakWorkingSetSize,
                'private_commit_bytes': counters.PagefileUsage}
    if sys.platform.startswith('linux'):
        import resource
        pages = int(Path('/proc/self/statm').read_text().split()[1])
        return {'rss_bytes': pages * os.sysconf('SC_PAGE_SIZE'),
                'lifetime_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    return {'rss_bytes': None, 'lifetime_peak_rss_bytes': None}


def host_preflight(required_bytes):
    if sys.platform == 'win32':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    'total', 'available', 'total_page', 'available_page',
                    'total_virtual', 'available_virtual', 'extended')]
        status = Status()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('GlobalMemoryStatusEx failed')
        available = status.available
    elif sys.platform.startswith('linux'):
        entries = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(entries['MemAvailable'].split()[0]) * 1024
    else:
        raise RuntimeError('host memory preflight is supported on Windows/Linux')
    if required_bytes > available * 0.65:
        raise RuntimeError(f'host preflight rejected {required_bytes} bytes; available={available}')
    return {'required_bytes': required_bytes, 'available_bytes': available, 'fraction_limit': 0.65}


def disk_preflight(path, required_bytes):
    parent = Path(path).resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required_bytes + (1 << 30):
        raise RuntimeError(f'disk preflight rejected {required_bytes} bytes; free={free}')
    return {'free_bytes': free, 'required_bytes': required_bytes, 'reserve_bytes': 1 << 30}
