import sys
import unittest

sys.path.append("..")

from surprise_deep import ModelOption


class RunOptionsTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_options(self):
        option1 = ModelOption(root_dir="test_rootDir1", force_new=True)
        self.assertEqual(option1["num_epoch"], 50)
        self.assertEqual(option1.num_epoch, 50)
        option1.deleteOption()

        option2 = ModelOption(root_dir="test_rootDir2")
        self.assertEqual(option2["batch_size"], 128)
        option2.deleteOption()

        option3 = ModelOption(root_dir="test_rootDir3")
        option3["batch_size"] = 200
        option3.num_epoch = 100
        option3.save()

        option4 = ModelOption(root_dir="test_rootDir3")
        self.assertEqual(option4["batch_size"], 200)
        self.assertEqual(option4.num_epoch, 100)
        option3.deleteOption()
