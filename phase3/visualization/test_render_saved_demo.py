import unittest
from render_saved_demo import frame_indices, command_at


class ReplayTimelineTests(unittest.TestCase):
    def test_realtime_130_seconds_is_3250_frames(self):
        counts = [len(frame_indices(seconds*50, .02, 25)) for seconds in (20,20,20,20,20,30)]
        self.assertEqual(counts, [500,500,500,500,500,750])
        self.assertEqual(sum(counts), 3250)

    def test_does_not_append_extra_terminal_frame(self):
        frames = list(frame_indices(1500, .02, 25))
        self.assertEqual(frames[-1], 1498)

    def test_reject_nonintegral_resampling(self):
        with self.assertRaises(ValueError):
            frame_indices(1000, .02, 30)

    def test_command_boundary_uses_new_command(self):
        case = {'segments': [[0, 12, [.4,0,0]], [12,22,[0,0,0]], [22,30,[.4,0,0]]]}
        self.assertEqual(command_at(case, 11.98), [.4,0,0])
        self.assertEqual(command_at(case, 12), [0,0,0])
        self.assertEqual(command_at(case, 22), [.4,0,0])
        with self.assertRaises(ValueError):
            command_at(case, 30)


if __name__ == '__main__':
    unittest.main()
