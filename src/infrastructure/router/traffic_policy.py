"""Credential-based workload identity; never infer identities from request content."""
from datetime import datetime
import hmac
import threading
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


class TrafficPolicy:
    """One instance per router process. Keys must originate from trusted Secret mounts.

    Call authenticate(), then admit() for authenticated inference requests before
    backend allocation. Only accepted requests call begin()/end(). All backend
    inflight counts must still participate in drain, including background work.
    """

    def __init__(self, protected_key, background_key=None, *, night_start='22:30',
                 night_end='07:30', timezone='Asia/Shanghai', clock=time.time):
        if not isinstance(protected_key, str) or not protected_key:
            raise ValueError('protected credential required')
        if background_key is not None and (not isinstance(background_key, str) or not background_key):
            raise ValueError('background credential must be nonempty')
        if background_key is not None and hmac.compare_digest(protected_key.encode(), background_key.encode()):
            raise ValueError('workload credentials must differ')
        self._protected = protected_key.encode()
        self._background = background_key.encode() if background_key is not None else None
        self.window_start, self.window_end = self._minutes(night_start), self._minutes(night_end)
        if self.window_start == self.window_end:
            raise ValueError('training window must have distinct bounds')
        self.zone = ZoneInfo(timezone)
        self.clock = clock
        self.lock = threading.Lock()
        now = clock()
        # A new replica requires a full protected quiet period before reclamation.
        self.last = {'protected': now, 'background': 0.0}
        self.active = {}
        self.sequence = 0
        self.rejected_background = 0

    @staticmethod
    def _minutes(value):
        if not isinstance(value, str) or len(value) != 5 or value[2] != ':':
            raise ValueError('window bounds must be HH:MM')
        try:
            hour, minute = int(value[:2]), int(value[3:])
        except ValueError:
            raise ValueError('window bounds must be HH:MM') from None
        if not 0 <= hour < 24 or not 0 <= minute < 60:
            raise ValueError('window bound outside clock range')
        return hour * 60 + minute

    def authenticate(self, authorization):
        """Return protected/background or None. Unknown credentials never downgrade."""
        if not isinstance(authorization, str) or not authorization.startswith('Bearer '):
            return None
        candidate = authorization[7:].encode()
        protected = hmac.compare_digest(candidate, self._protected)
        background = self._background is not None and hmac.compare_digest(candidate, self._background)
        if protected:
            return 'protected'
        return 'background' if background else None

    @staticmethod
    def is_business(method, path):
        # Probes/models/status reads do not represent generation work.
        return method.upper() == 'POST' and urlsplit(path).path.startswith('/v1/')

    def in_night_window(self, now=None):
        local = datetime.fromtimestamp(self.clock() if now is None else now, self.zone)
        minute = local.hour * 60 + local.minute
        if self.window_start < self.window_end:
            return self.window_start <= minute < self.window_end
        return minute >= self.window_start or minute < self.window_end

    def admit(self, identity, method, path, *, now=None):
        """Return HTTP status: 200 allowed, 401 unauthenticated, 503 background paused.

        Caller must separately restrict admin APIs to protected credentials.
        Rejected requests must not reset the idle timer. Return Retry-After with
        a bounded delay on 503 so clients can retry or use their fallback route.
        """
        if identity not in self.last:
            return 401
        if identity == 'background' and self.is_business(method, path) and self.in_night_window(now):
            with self.lock:
                self.rejected_background += 1
            return 503
        return 200

    def begin(self, identity, method, path):
        """Track accepted work; return opaque ticket (None for nonbusiness)."""
        if identity not in self.last:
            raise ValueError('authenticated workload identity required')
        if not self.is_business(method, path):
            return None
        with self.lock:
            self.sequence += 1
            ticket = self.sequence
            self.active[ticket] = identity
            self.last[identity] = self.clock()
            return ticket

    def end(self, ticket):
        """Finish exactly once, refreshing class idle time after streaming completes."""
        if ticket is None:
            return
        with self.lock:
            identity = self.active.pop(ticket, None)
            if identity is None:
                raise ValueError('unknown or completed request ticket')
            self.last[identity] = self.clock()

    def snapshot(self):
        with self.lock:
            return {'traffic_policy_version': 1,
                    'last_protected_at': self.last['protected'],
                    'last_background_at': self.last['background'],
                    'protected_inflight': sum(v == 'protected' for v in self.active.values()),
                    'background_inflight': sum(v == 'background' for v in self.active.values()),
                    'background_rejected_total': self.rejected_background}
