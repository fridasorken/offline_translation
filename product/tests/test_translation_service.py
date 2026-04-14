from app.services.translation_service import translate_message


class DummyTranslator:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.inputs: list[str] = []

    def translate(self, text: str) -> str:
        self.inputs.append(text)
        return f"{self.prefix}:{text}"


class DummyRuntime:
    def __init__(self) -> None:
        self.sender = DummyTranslator("outgoing")
        self.receiver = DummyTranslator("incoming")
        self.direction_calls: list[bool] = []

    def for_direction(self, is_outgoing: bool) -> DummyTranslator:
        self.direction_calls.append(is_outgoing)
        return self.sender if is_outgoing else self.receiver


def test_translate_message_uses_direction_selected_by_runtime() -> None:
    runtime = DummyRuntime()

    translated = translate_message(runtime, True, "Hold position")

    assert translated == "outgoing:Hold position"
    assert runtime.direction_calls == [True]
    assert runtime.sender.inputs == ["Hold position"]
