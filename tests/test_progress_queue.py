import unittest

from gui.progress_queue import FreshQueue


class FreshQueueTests(unittest.TestCase):
    def test_drops_oldest_item_when_full(self):
        queue = FreshQueue[int](maxsize=2)

        queue.put(1)
        queue.put(2)
        queue.put(3)

        self.assertEqual(queue.drain(), [2, 3])


if __name__ == "__main__":
    unittest.main()
