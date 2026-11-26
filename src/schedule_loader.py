"""Resilient schedule reads without silently publishing stale data."""
from __future__ import annotations

import logging
import time

import nflreadpy as nfl
import requests

RETRY_DELAYS = (5, 15, 30, 60, 120)


def _retryable(error: Exception) -> bool:
    """Inspect nflreadpy's wrapped HTTP error; don't retry permanent auth errors."""
    cause = error
    while cause is not None:
        if isinstance(cause, requests.HTTPError) and cause.response is not None:
            status = cause.response.status_code
            return status in (404, 408, 429) or 500 <= status < 600
        cause = cause.__cause__
    return isinstance(error, (ConnectionError, TimeoutError,
                              requests.ConnectionError, requests.Timeout))


def load_nflverse_schedules(seasons):
    """Retry transient release-asset failures; propagate the final failure."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return nfl.load_schedules(seasons)
        except (ConnectionError, TimeoutError, requests.RequestException) as error:
            if attempt == len(RETRY_DELAYS) or not _retryable(error):
                raise
            delay = RETRY_DELAYS[attempt]
            logging.warning("nflverse schedule download failed (%s); retrying in %ss",
                            error, delay)
            time.sleep(delay)
