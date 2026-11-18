#!/usr/bin/env python3
"""Run checker boundary tests without GEM5 or a RISC-V toolchain."""

import unittest

import check


class ObservationTests(unittest.TestCase):
    model = {
        "locations": ["1:x5", "1:x7"],
        "allowed": [(0, 0), (0, 1), (1, 1)],
    }

    @staticmethod
    def outcome(first, second, count=1):
        return f"LITMUS_OUTCOME {count:016x} " f"{first:016x} {second:016x}\n"

    def status(self, text, trials=2):
        return check.check_observations(text, self.model, trials)["status"]

    def test_legal_sample_does_not_need_every_allowed_state(self):
        text = self.outcome(0, 0, count=2) + "LITMUS_DONE 0000000000000002\n"
        self.assertEqual(self.status(text), "pass")

    def test_early_forbidden_state_is_detected(self):
        text = self.outcome(0, 0) + self.outcome(1, 0)
        text += "LITMUS_FORBIDDEN 0000000000000002\n"
        self.assertEqual(self.status(text, trials=32), "forbidden")

    def test_forbidden_state_cannot_pass_with_done_marker(self):
        text = self.outcome(1, 0, count=2) + "LITMUS_DONE 0000000000000002\n"
        result = check.check_observations(text, self.model, 2)
        self.assertNotEqual(result["status"], "pass")
        self.assertEqual(result["forbidden"], [(1, 0)])

    def test_missing_or_invalid_completion_cannot_pass(self):
        outcomes = self.outcome(0, 0, count=2)
        for suffix in (
            "",
            "LITMUS_DONE 00000000",
            "LITMUS_DONE 0000000000000001\n",
            "LITMUS_DONE 0000000000000002junk\n",
            "LITMUS_DONE 0000000000000002\n" * 2,
        ):
            with self.subTest(suffix=suffix):
                self.assertNotEqual(self.status(outcomes + suffix), "pass")

    def test_done_marker_does_not_replace_missing_rounds(self):
        text = self.outcome(0, 0) + "LITMUS_DONE 0000000000000002\n"
        self.assertNotEqual(self.status(text), "pass")

    def test_malformed_or_duplicate_observations_are_rejected(self):
        for text in (
            self.outcome(0, 0) * 2,
            self.outcome(0, 0, count=0),
            "LITMUS_OUTCOME 1 0 0\n",
            "LITMUS_OUTCOME 0000000000000001 0000000000000000\n",
            "LITMUS_OUTCOME invalid\n",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.status(text)

    def test_known_uart_diagnostic_boundary_is_accepted(self):
        text = self.outcome(0, 1, count=2)
        text += "LITMUS_DONE 0000000000000002Profile report: example\n"
        self.assertEqual(self.status(text), "pass")

    def test_signed_64_bit_observation_is_preserved(self):
        model = {"locations": ["0:x5"], "allowed": [(-1,)]}
        text = "LITMUS_OUTCOME 0000000000000001 ffffffffffffffff\n"
        text += "LITMUS_DONE 0000000000000001\n"
        result = check.check_observations(text, model, 1)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["outcomes"][0]["values"], (-1,))


class ModelStateTests(unittest.TestCase):
    def test_pointer_and_integer_observations_are_preserved(self):
        text = "States 1\n0:x5=x; 0:x6=18446744073709551615; [x]=2;\n"
        names, states = check.parse_states(text)
        self.assertEqual(names, ["0:x5", "0:x6", "[x]"])
        self.assertEqual(states, [("x", -1, 2)])

    def test_truncated_or_ambiguous_state_sets_are_rejected(self):
        for text in (
            "States 2\n1:x5=0;\n",
            "States 2\n1:x5=0;\n1:x5=0;\n",
            "States 1\n1:x5=0; 1:x5=1;\n",
            "States 1\n1:x5=0; trailing text\n",
            "States 1\n1:x5=18446744073709551616;\n",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    check.parse_states(text)

    def test_instruction_only_test_keeps_completion_observation(self):
        names, states = check.parse_states("States 1\n\n")
        self.assertEqual(names, ["__completion"])
        self.assertEqual(states, [(0,)])


if __name__ == "__main__":
    unittest.main()
