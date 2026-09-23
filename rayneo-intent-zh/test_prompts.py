import json
import unittest

from evaluate import QUESTIONS, payload_for
from prompts import PROFILES


class PromptProfilesTest(unittest.TestCase):
    def test_baseline_stays_training_default(self):
        self.assertIs(QUESTIONS, PROFILES["baseline"])
        self.assertEqual(QUESTIONS["assist"]["instructions"],
                         "当前说话者是否正在明确求助，或提出需要眼镜助手立即提供简短事实或解释的问题？普通闲聊、自问自答、假设对话不算。")
        self.assertEqual(QUESTIONS["schedule"]["instructions"],
                         "当前发言是否包含说话者确实打算参与、尚未取消且值得确认保存的日程安排？假设、取消、他人的无关事项不算。")

    def test_candidates_keep_boolean_contract_and_exclude_gold(self):
        suite = {"timeZone": "Asia/Singapore", "recordedAt": 1}
        case = {"id": "synthetic", "text": "明天我确定去开会。", "context": ["先前的话"],
                "group": "confirmed", "assist": False, "schedule": True}
        for name, questions in PROFILES.items():
            with self.subTest(name=name):
                self.assertEqual(set(questions), {"assist", "schedule"})
                for question in questions.values():
                    self.assertEqual(question["type"], "boolean")
                    self.assertTrue(question["instructions"].strip())
                    self.assertLess(len(question["instructions"]), 250)
                    self.assertEqual(set(question.get("criteria", {})) - {"false", "true"}, set())
                payload = payload_for(suite, [case], questions)
                self.assertEqual(payload["states"][0]["questions"], questions)
                state = json.loads(payload["states"][0]["state"])
                self.assertEqual(set(state), {"transcript", "speakerId", "context", "timeZone", "recordedAt"})
                self.assertNotIn("group", state)
                self.assertNotIn("assist", state)
                self.assertNotIn("schedule", state)


if __name__ == "__main__":
    unittest.main()
