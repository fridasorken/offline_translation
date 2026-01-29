import streamlit as st
import requests
import time
import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
TRANSLATE_ENDPOINT = "/translate"


class Language(Enum):
    """Supported languages with their codes and display names."""
    NORWEGIAN_BOKMAL = ("nob", "Norwegian Bokmål")
    ENGLISH = ("en", "English")
    
    @property
    def code(self) -> str:
        return self.value[0]
    
    @property
    def display_name(self) -> str:
        return self.value[1]
    
    @classmethod
    def from_code(cls, code: str) -> Optional["Language"]:
        for lang in cls:
            if lang.code == code:
                return lang
        return None
    
    @classmethod
    def get_display_names(cls) -> list[str]:
        return [lang.display_name for lang in cls]
    
    @classmethod
    def from_display_name(cls, name: str) -> Optional["Language"]:
        for lang in cls:
            if lang.display_name == name:
                return lang
        return None


@dataclass
class TranslationRequest:
    """Request payload for the translation API."""
    src_lang: str
    tgt_lang: str
    source: str
    model_id: str
    
    def to_dict(self) -> dict:
        return {
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "source": self.source,
            "model_id": self.model_id,
        }


@dataclass
class TranslationResponse:
    """Response from the translation API."""
    translated_value: str
    latency: float
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    source: Optional[str] = None
    model_id: Optional[str] = None


def fetch_available_models() -> list[str]:
    """
    Fetch available models from the backend API.
    Falls back to empty list if API is unavailable.
    """
    try:
        url = f"{API_BASE_URL}/models"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["model_id"] for model in data.get("models", [])]
    except requests.RequestException as e:
        st.warning(f"Could not fetch models from API: {e}")
        return []


def translate_text(request: TranslationRequest) -> TranslationResponse:
    """
    Send translation request to the API.
    """
    url = f"{API_BASE_URL}{TRANSLATE_ENDPOINT}"
    
    response = requests.post(
        url,
        json=request.to_dict(),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    
    data = response.json()
    # backend returns latency_ms, convert to seconds
    latency_ms = data.get("latency_ms", 0)
    latency_seconds = latency_ms / 1000.0
    return TranslationResponse(
        translated_value=data.get("translated_value", ""),
        latency=latency_seconds,
        src_lang=data.get("src_lang"),
        tgt_lang=data.get("tgt_lang"),
        source=data.get("source"),
        model_id=data.get("model_id"),
    )


def mock_translate(request: TranslationRequest) -> TranslationResponse:
    """
    Mock translation for testing without the API.
    Remove this in prod.
    """
    import random
    simulated_latency = random.uniform(0.1, 0.5)
    time.sleep(simulated_latency)
    
    return TranslationResponse(
        translated_value=f"[mock translation of: {request.source}]",
        latency=simulated_latency,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        source=request.source,
        model_id=request.model_id,
    )


def init_session_state():
    """Initialise session state variables."""
    if "translation_result" not in st.session_state:
        st.session_state.translation_result = None
    if "last_latency" not in st.session_state:
        st.session_state.last_latency = None
    if "available_models" not in st.session_state:
        st.session_state.available_models = None
    if "selected_src_lang" not in st.session_state:
        st.session_state.selected_src_lang = "English"
    if "selected_tgt_lang" not in st.session_state:
        st.session_state.selected_tgt_lang = "Norwegian Bokmål"


def main():

    init_session_state()

    # fetch available models from backend if not using mock
    if not USE_MOCK and st.session_state.available_models is None:
        with st.spinner("Loading available models..."):
            st.session_state.available_models = fetch_available_models()

    # determine which models to show
    if USE_MOCK:
        available_models = ["mock-model"]
    else:
        available_models = st.session_state.available_models or []

    if not available_models:
        st.error("No translation models available. Please check the backend configuration.")
        return

    # model selection
    col_model, col_spacer = st.columns([2, 3])
    with col_model:
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=0,
        )
    
    # language selector
    col_src, col_swap, col_tgt = st.columns([2, 1, 2])

    def swap_languages():
        """Callback to swap source and target languages."""
        temp = st.session_state.selected_src_lang
        st.session_state.selected_src_lang = st.session_state.selected_tgt_lang
        st.session_state.selected_tgt_lang = temp

    with col_src:
        src_lang_name = st.selectbox(
            "Source language",
            options=Language.get_display_names(),
            index=Language.get_display_names().index(st.session_state.selected_src_lang),
            key="src_lang_widget",
        )
        # Update the session state when user selects manually
        if src_lang_name != st.session_state.selected_src_lang:
            st.session_state.selected_src_lang = src_lang_name
        src_lang = Language.from_display_name(st.session_state.selected_src_lang)

    with col_swap:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("⇄ Swap", use_container_width=True, on_click=swap_languages)

    with col_tgt:
        tgt_lang_name = st.selectbox(
            "Target language",
            options=Language.get_display_names(),
            index=Language.get_display_names().index(st.session_state.selected_tgt_lang),
            key="tgt_lang_widget",
        )
        # Update the session state when user selects manually
        if tgt_lang_name != st.session_state.selected_tgt_lang:
            st.session_state.selected_tgt_lang = tgt_lang_name
        tgt_lang = Language.from_display_name(st.session_state.selected_tgt_lang)
    
    # input/output areas
    col_input, col_output = st.columns(2)
    
    with col_input:
        source_text = st.text_area(
            "Source text",
            height=200,
            placeholder="Enter text to translate...",
            label_visibility="collapsed",
        )
    
    with col_output:
        output_placeholder = st.empty()
        
        if st.session_state.translation_result:
            output_placeholder.text_area(
                "Translation output",
                value=st.session_state.translation_result,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            output_placeholder.text_area(
                "Translation output",
                value="",
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )
    
    # translate button, latency display
    col_btn, col_latency = st.columns([1, 2])
    
    with col_btn:
        translate_clicked = st.button(
            "Translate",
            type="primary",
            use_container_width=True,
            disabled=not source_text.strip(),
        )
    
    with col_latency:
        if st.session_state.last_latency is not None:
            latency_ms = st.session_state.last_latency * 1000
            st.metric(
                label="API latency",
                value=f"{latency_ms:.1f} ms",
            )
    
    # handle translation
    if translate_clicked and source_text.strip():
        request = TranslationRequest(
            src_lang=src_lang.code,
            tgt_lang=tgt_lang.code,
            source=source_text.strip(),
            model_id=selected_model,
        )

        with st.spinner("Translating..."):
            try:
                if USE_MOCK:
                    response = mock_translate(request)
                else:
                    response = translate_text(request)

                st.session_state.translation_result = response.translated_value
                st.session_state.last_latency = response.latency
                st.rerun()

            except requests.RequestException as e:
                st.error(f"Translation failed: {e}")
            except Exception as e:
                st.error(f"error: {e}")


if __name__ == "__main__":
    main()