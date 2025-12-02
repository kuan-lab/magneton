# interrupts.py
import signal, threading, weakref, multiprocessing as mp
import concurrent.futures as cf
import time

class InterruptException(Exception):
    """Raised when user interrupts execution (Ctrl+C)."""

class InterruptController:
    """Graceful Ctrl+C controller that cancels threads and processes without killing main process."""

    def __init__(self):
        self._stop = threading.Event()
        self._registry_threads = weakref.WeakSet()
        self._registry_procs = weakref.WeakSet()
        self._registry_pools = weakref.WeakSet()
        self._orig_handlers = {}
        self._patched = False

    def _patch(self):
        if self._patched:
            return
        self._patched = True

        # Patch ThreadPoolExecutor
        self._orig_tpe_init = cf.ThreadPoolExecutor.__init__
        def tpe_init(obj, *a, **kw):
            self._orig_tpe_init(obj, *a, **kw)
            self._registry_threads.add(obj)
        cf.ThreadPoolExecutor.__init__ = tpe_init

        # Patch ProcessPoolExecutor
        self._orig_ppe_init = cf.ProcessPoolExecutor.__init__
        def ppe_init(obj, *a, **kw):
            self._orig_ppe_init(obj, *a, **kw)
            self._registry_procs.add(obj)
        cf.ProcessPoolExecutor.__init__ = ppe_init

    def _sig_handler(self, sig, frame):
        if self._stop.is_set():
            return
        self._stop.set()
        print("\n[Interrupt] Ctrl+C detected — stopping running tasks...")

        # Cancel thread pools
        for ex in list(self._registry_threads):
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        # Cancel process pools
        for ex in list(self._registry_procs):
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        # Terminate multiprocessing children
        for child in mp.active_children():
            try:
                child.terminate()
            except Exception:
                pass

        time.sleep(0.2)

        raise InterruptException("User interrupted execution")

    def __enter__(self):
        self._patch()
        for s in (signal.SIGINT, signal.SIGTERM):
            self._orig_handlers[s] = signal.getsignal(s)
            signal.signal(s, self._sig_handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        for s, h in self._orig_handlers.items():
            signal.signal(s, h)
        self._orig_handlers.clear()
        if exc_type is InterruptException:
            print("[Info] Task interrupted gracefully, returning to main menu.")
            return True 
        return False
