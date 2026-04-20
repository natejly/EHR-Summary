from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from mlx_qwen_inference import (
    Qwen35MlxInference,
    build_arg_parser,
    print_terminal_output,
)


class MlxQwenInferenceStopStringTests(unittest.TestCase):
    def test_stop_stream_at_strings_handles_split_markers(self) -> None:
        chunks = [
            "Brief Hospital Course:\nImproved with diuresis.\n<<END_",
            "OF_SUMMARY>>\nExtra text that should not appear.",
        ]

        result = "".join(
            Qwen35MlxInference._stop_stream_at_strings(
                chunks,
                ("<<END_OF_SUMMARY>>",),
            )
        )

        self.assertEqual(result, "Brief Hospital Course:\nImproved with diuresis.\n")

    def test_stop_stream_at_strings_passthrough_without_stops(self) -> None:
        chunks = ["a", "b", "c"]

        result = "".join(
            Qwen35MlxInference._stop_stream_at_strings(
                chunks,
                None,
            )
        )

        self.assertEqual(result, "abc")

    def test_print_terminal_output_plain_text(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_terminal_output("hello", render_markdown=False)

        self.assertEqual(buffer.getvalue(), "hello\n")

    def test_arg_parser_accepts_render_markdown_flag(self) -> None:
        args = build_arg_parser().parse_args(["--prompt", "hi", "--render-markdown"])

        self.assertTrue(args.render_markdown)


if __name__ == "__main__":
    unittest.main()
