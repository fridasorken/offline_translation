from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        raise NotImplementedError
