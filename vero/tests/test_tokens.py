from vero.utils.tokens import run_result_to_messages


def test_run_result_to_messages_handles_empty_response_content() -> None:
    messages = run_result_to_messages(
        [
            {
                "type": "reasoning",
                "content": [],
                "summary": [],
            }
        ]
    )

    assert messages[0]["content"] == ""


def test_run_result_to_messages_uses_first_content_item() -> None:
    messages = run_result_to_messages(
        [{"role": "assistant", "content": ["first", "second"]}]
    )

    assert messages[0]["content"] == "first"
