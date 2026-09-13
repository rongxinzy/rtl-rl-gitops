"""Non-authoritative, single-flight catalog snapshots; never used for admission."""
import copy
import threading
import time


class CatalogSnapshot:
    def __init__(self, builder, ttl=60, monotonic=time.monotonic, wall_time=time.time):
        self.builder = builder
        self.ttl = ttl
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.lock = threading.Lock()
        self.value = None
        self.updated_at = None
        self.updated_monotonic = None
        self.next_refresh = 0
        self.refreshing = False
        self.error_type = None

    def _refresh(self):
        try:
            value = self.builder()
            if not isinstance(value, dict):
                raise ValueError('invalid catalog snapshot')
        except Exception as exc:
            with self.lock:
                # Do not expose exception messages, paths or producer responses.
                self.error_type = type(exc).__name__
                self.next_refresh = self.monotonic() + self.ttl
                self.refreshing = False
            return
        with self.lock:
            self.value = value
            self.updated_at = self.wall_time()
            self.updated_monotonic = self.monotonic()
            self.next_refresh = self.updated_monotonic + self.ttl
            self.error_type = None
            self.refreshing = False

    def read(self, empty):
        with self.lock:
            now = self.monotonic()
            if not self.refreshing and now >= self.next_refresh:
                self.refreshing = True
                threading.Thread(target=self._refresh, name='catalog-refresh', daemon=True).start()
            result = copy.deepcopy(self.value if self.value is not None else empty)
            age = None if self.updated_monotonic is None else max(0, now - self.updated_monotonic)
            result['catalog_snapshot'] = {
                'authoritative_for_admission': False,
                'status': 'warming' if self.value is None else 'ready',
                'updated_at': self.updated_at,
                'age_seconds': age,
                'stale': age is None or age >= self.ttl or self.error_type is not None,
                'refreshing': self.refreshing,
                'error_type': self.error_type,
            }
            return result
