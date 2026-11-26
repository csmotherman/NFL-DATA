import unittest
from unittest.mock import patch

import requests

from schedule_loader import RETRY_DELAYS, load_nflverse_schedules


def wrapped_http_error(status):
    response = requests.Response()
    response.status_code = status
    error = ConnectionError("schedule download failed")
    error.__cause__ = requests.HTTPError(response=response)
    return error


class ScheduleLoaderTests(unittest.TestCase):
    @patch("schedule_loader.time.sleep")
    @patch("schedule_loader.nfl.load_schedules")
    def test_missing_release_asset_recovers(self, load, sleep):
        expected = object()
        load.side_effect = [wrapped_http_error(404), expected]
        self.assertIs(load_nflverse_schedules([2026]), expected)
        self.assertEqual(load.call_count, 2)
        sleep.assert_called_once_with(RETRY_DELAYS[0])

    @patch("schedule_loader.time.sleep")
    @patch("schedule_loader.nfl.load_schedules")
    def test_exhausted_retries_fail_closed(self, load, sleep):
        load.side_effect = wrapped_http_error(404)
        with self.assertRaises(ConnectionError):
            load_nflverse_schedules([2026])
        self.assertEqual(load.call_count, len(RETRY_DELAYS) + 1)
        self.assertEqual(sleep.call_count, len(RETRY_DELAYS))

    @patch("schedule_loader.time.sleep")
    @patch("schedule_loader.nfl.load_schedules")
    def test_auth_error_not_retried(self, load, sleep):
        load.side_effect = wrapped_http_error(403)
        with self.assertRaises(ConnectionError):
            load_nflverse_schedules([2026])
        load.assert_called_once_with([2026])
        sleep.assert_not_called()

    @patch("schedule_loader.time.sleep")
    @patch("schedule_loader.nfl.load_schedules")
    def test_schema_errors_not_retried(self, load, sleep):
        load.side_effect = ValueError("invalid schema")
        with self.assertRaises(ValueError):
            load_nflverse_schedules([2026])
        sleep.assert_not_called()
