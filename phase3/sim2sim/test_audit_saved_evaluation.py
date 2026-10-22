"""Scoring edge cases; synthetic data, no model weights or simulator needed."""
import unittest

import numpy as np

from audit_saved_evaluation import audit_episode


class SavedEvaluationAuditTests(unittest.TestCase):
    dt = .02
    thresholds = dict(stopped_xy_speed_m_s=.1, stopped_yaw_speed_rad_s=.15,
                      moving_xy_error_m_s=.15, moving_yaw_error_rad_s=.2,
                      mean_tilt_deg=15, stop_stable_window_s=1., stop_stable_by_s=3.)
    case = dict(duration_s=6, stop_time_s=1., segments=[[0, 1, [.4, 0, 0]], [1, 6, [0, 0, 0]]],
                measurement_windows=[[1, 6, 'stopped']])

    def result(self, metrics, *, early=False, passed=False, stable=None):
        values = metrics[50:300]
        complete = len(metrics) == 300 and not early
        return dict(steps=len(metrics), duration_s=len(metrics)*self.dt, early_termination=early,
                    windows=[dict(start_s=1, end_s=6, kind='stopped', samples=len(values),
                                  expected_samples=250, means=values.mean(axis=0).tolist() if len(values) else None,
                                  passed=len(values) == 250 and bool(np.all(values.mean(axis=0) <= [.1, .15, 15])))],
                    stop_stable_time_s=stable, episode_passed=passed, pilot_episode_passed=passed,
                    full_episode_and_windows_passed=complete)

    def score(self, result, metrics):
        return audit_episode(result, metrics, self.case, self.thresholds, self.dt)

    def test_average_pass_does_not_hide_interrupted_stop(self):
        metrics = np.zeros((300, 3))
        metrics[50::10, 1] = .2
        passed, diagnostic = self.score(self.result(metrics), metrics)
        self.assertFalse(passed)
        self.assertAlmostEqual(diagnostic['longest_stable_seconds'], .18)

    def test_exact_one_second_stable_window_passes(self):
        metrics = np.zeros((300, 3))
        passed, diagnostic = self.score(self.result(metrics, passed=True, stable=1.), metrics)
        self.assertTrue(passed)
        self.assertEqual(diagnostic['stable_time_s'], 1.)

    def test_late_stability_still_fails_deadline(self):
        metrics = np.zeros((300, 3))
        metrics[50:201:10, 1] = .2
        # Last interruption at global t=4; first full window ends at 5.02.
        passed, diagnostic = self.score(self.result(metrics, stable=4.02), metrics)
        self.assertFalse(passed)
        self.assertAlmostEqual(diagnostic['stable_time_s'], 4.02)

    def test_early_end_cannot_pass_empty_window(self):
        metrics = np.zeros((1, 3))
        passed, _ = self.score(self.result(metrics, early=True), metrics)
        self.assertFalse(passed)

    def test_tampered_mean_is_rejected(self):
        metrics = np.zeros((300, 3))
        result = self.result(metrics, passed=True, stable=1.)
        result['windows'][0]['means'][0] = .05
        with self.assertRaises(AssertionError):
            self.score(result, metrics)

    def test_tampered_pass_flag_is_rejected(self):
        metrics = np.zeros((300, 3))
        with self.assertRaises(ValueError):
            self.score(self.result(metrics, passed=False, stable=1.), metrics)

    def test_nonfinite_metrics_are_rejected(self):
        metrics = np.zeros((300, 3))
        result = self.result(metrics, passed=True, stable=1.)
        metrics[100, 1] = np.nan
        with self.assertRaises(ValueError):
            self.score(result, metrics)


if __name__ == '__main__':
    unittest.main()
