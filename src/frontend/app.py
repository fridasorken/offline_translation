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
    ENGLISH = ("en", "English")
    NORWEGIAN_BOKMAL = ("nob", "Norwegian Bokmål")
    NORWEGIAN_NYNORSK = ("nno", "Norwegian Nynorsk")

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
    model_was_warm: bool = True
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    source: Optional[str] = None
    model_id: Optional[str] = None


def wait_for_backend(timeout: int = 60) -> bool:
    """
    Poll the backend health endpoint until it's ready or timeout is reached.
    Returns True if backend is ready, False if timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            url = f"{API_BASE_URL}/health"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            # Backend not ready yet, wait and retry
            pass
        time.sleep(1)
    return False


def fetch_available_models() -> dict[str, dict]:
    """
    Fetch available models from the backend API.
    Returns dict mapping model_id to full model info including supported pairs.
    Falls back to empty dict if API is unavailable.
    """
    try:
        url = f"{API_BASE_URL}/models"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            model["model_id"]: {
                "adapter": model["adapter"],
                "supported_pairs": model["supported_pairs"]
            }
            for model in data.get("models", [])
        }
    except requests.RequestException as e:
        st.warning(f"Could not fetch models from API: {e}")
        return {}


def unload_model(model_id: str) -> None:
    """Unload a model from backend memory to force cold start."""
    try:
        url = f"{API_BASE_URL}/models/{model_id}/unload"
        response = requests.post(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        st.warning(f"Could not unload model: {e}")


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
        model_was_warm=data.get("model_was_warm", True),
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
        model_was_warm=True,
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
    if "last_was_warm" not in st.session_state:
        st.session_state.last_was_warm = None
    if "last_word_count" not in st.session_state:
        st.session_state.last_word_count = None
    if "cold_latency" not in st.session_state:
        st.session_state.cold_latency = None
    if "warm_latency" not in st.session_state:
        st.session_state.warm_latency = None
    if "show_comparison" not in st.session_state:
        st.session_state.show_comparison = False
    if "benchmark_mode" not in st.session_state:
        st.session_state.benchmark_mode = True
    if "available_models" not in st.session_state:
        st.session_state.available_models = None
    if "selected_src_lang" not in st.session_state:
        st.session_state.selected_src_lang = None
    if "selected_tgt_lang" not in st.session_state:
        st.session_state.selected_tgt_lang = None


def get_supported_languages(model_info: dict) -> tuple[list[str], list[str]]:
    """
    Extract supported source and target languages from model info.
    Returns (source_langs, target_langs) as lists of language codes.
    """
    supported_pairs = model_info.get("supported_pairs", [])
    src_langs = sorted(set(pair[0] for pair in supported_pairs))
    tgt_langs = sorted(set(pair[1] for pair in supported_pairs))
    return src_langs, tgt_langs


def get_valid_target_languages(model_info: dict, src_lang_code: str) -> list[str]:
    """
    Get valid target languages for a given source language.
    """
    supported_pairs = model_info.get("supported_pairs", [])
    return sorted(set(pair[1] for pair in supported_pairs if pair[0] == src_lang_code))


def get_valid_source_languages(model_info: dict, tgt_lang_code: str) -> list[str]:
    """
    Get valid source languages for a given target language.
    """
    supported_pairs = model_info.get("supported_pairs", [])
    return sorted(set(pair[0] for pair in supported_pairs if pair[1] == tgt_lang_code))


def code_to_display_name(lang_code: str) -> str:
    """Convert language code to display name, fallback to code if not found."""
    lang = Language.from_code(lang_code)
    return lang.display_name if lang else lang_code


def display_name_to_code(display_name: str) -> str:
    """Convert display name to language code, fallback to display name if not found."""
    lang = Language.from_display_name(display_name)
    return lang.code if lang else display_name


def main():

    init_session_state()

    # wait for backend, fetch available models if not using mock
    if not USE_MOCK and st.session_state.available_models is None:
        with st.spinner("Waiting for backend..."):
            if not wait_for_backend(timeout=120):
                st.error("Backend is not responding. Please ensure the backend service is running.")
                st.stop()
        with st.spinner("Loading available models..."):
            st.session_state.available_models = fetch_available_models()

    # determine which models to show
    if USE_MOCK:
        available_models_dict = {"mock-model": {"adapter": "mock", "supported_pairs": [("en", "nob"), ("nob", "en")]}}
    else:
        available_models_dict = st.session_state.available_models or {}

    if not available_models_dict:
        st.error("No translation models available. Please check the backend configuration.")
        return

    model_ids = list(available_models_dict.keys())

    # model selection and benchmark mode
    col_model, col_benchmark = st.columns([2, 3])
    with col_model:
        selected_model = st.selectbox(
            "Model",
            options=model_ids,
            index=0,
            key="selected_model",
        )

    # get model info and supported languages
    model_info = available_models_dict[selected_model]
    src_lang_codes, tgt_lang_codes = get_supported_languages(model_info)

    # convert to display names and sort alphabetically
    src_lang_options = sorted([code_to_display_name(code) for code in src_lang_codes])
    tgt_lang_options = sorted([code_to_display_name(code) for code in tgt_lang_codes])

    # Initialize language selections if not set or invalid for current model
    if st.session_state.selected_src_lang is None or display_name_to_code(st.session_state.selected_src_lang) not in src_lang_codes:
        st.session_state.selected_src_lang = src_lang_options[0] if src_lang_options else None
    if st.session_state.selected_tgt_lang is None or display_name_to_code(st.session_state.selected_tgt_lang) not in tgt_lang_codes:
        st.session_state.selected_tgt_lang = tgt_lang_options[0] if tgt_lang_options else None

    if not src_lang_options or not tgt_lang_options:
        st.error(f"Model {selected_model} has no supported language pairs.")
        return

    # language selector
    col_src, col_swap, col_tgt = st.columns([2, 1, 2])

    def swap_languages():
        """Callback to swap source and target languages if the pair is valid."""
        current_src_code = display_name_to_code(st.session_state.selected_src_lang)
        current_tgt_code = display_name_to_code(st.session_state.selected_tgt_lang)

        # Check if reverse pair is supported
        if (current_tgt_code, current_src_code) in model_info["supported_pairs"]:
            st.session_state.selected_src_lang = code_to_display_name(current_tgt_code)
            st.session_state.selected_tgt_lang = code_to_display_name(current_src_code)

    with col_src:
        src_lang_index = src_lang_options.index(st.session_state.selected_src_lang) if st.session_state.selected_src_lang in src_lang_options else 0
        st.selectbox(
            "Source language",
            options=src_lang_options,
            index=src_lang_index,
            key="src_lang_widget",
            on_change=lambda: setattr(st.session_state, "selected_src_lang", st.session_state.src_lang_widget)
        )
        src_lang_code = display_name_to_code(st.session_state.selected_src_lang)

    with col_swap:
        st.markdown("<br>", unsafe_allow_html=True)
        # Check if reverse pair is valid to enable/disable swap button
        current_src_code = display_name_to_code(st.session_state.selected_src_lang)
        current_tgt_code = display_name_to_code(st.session_state.selected_tgt_lang)
        can_swap = (current_tgt_code, current_src_code) in model_info["supported_pairs"]
        st.button("⇄ Swap", use_container_width=True, on_click=swap_languages, disabled=not can_swap)

    with col_tgt:
        # Filter target languages based on selected source language
        current_src_code = display_name_to_code(st.session_state.selected_src_lang)
        valid_tgt_codes = get_valid_target_languages(model_info, current_src_code)
        # Sort target options alphabetically
        valid_tgt_options = sorted([code_to_display_name(code) for code in valid_tgt_codes])

        if not valid_tgt_options:
            st.error(f"No target languages available for source language {st.session_state.selected_src_lang}")
            return

        # Ensure selected target is valid for current source - pick alphabetically first if invalid
        if st.session_state.selected_tgt_lang not in valid_tgt_options:
            st.session_state.selected_tgt_lang = valid_tgt_options[0]  # Already sorted alphabetically

        tgt_lang_index = valid_tgt_options.index(st.session_state.selected_tgt_lang) if st.session_state.selected_tgt_lang in valid_tgt_options else 0
        st.selectbox(
            "Target language",
            options=valid_tgt_options,
            index=tgt_lang_index,
            key="tgt_lang_widget",
            on_change=lambda: setattr(st.session_state, "selected_tgt_lang", st.session_state.tgt_lang_widget)
        )
        tgt_lang_code = display_name_to_code(st.session_state.selected_tgt_lang)
    
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
    
    # translate button and metrics display
    col_btn = st.columns(1)[0]

    with col_btn:
        translate_clicked = st.button(
            "Translate",
            type="primary",
            use_container_width=True,
            disabled=not source_text.strip(),
        )

    # Display performance metrics (cold vs warm comparison for research)
    if st.session_state.last_latency is not None:
        st.divider()
        st.subheader("Metrics")

        # Show cold vs warm comparison only if we just did both measurements
        if st.session_state.show_comparison and st.session_state.cold_latency is not None and st.session_state.warm_latency is not None:
            # st.divider()
            # st.subheader("Warmup impact")

            comparison_cols = st.columns(2)

            with comparison_cols[0]:
                cold_ms = st.session_state.cold_latency * 1000
                if st.session_state.last_word_count and st.session_state.cold_latency > 0:
                    cold_wps = st.session_state.last_word_count / st.session_state.cold_latency
                    st.metric(
                        label="Cold Start",
                        value=f"{cold_ms:.1f} ms",
                        delta=f"{cold_wps:.2f} WPS",
                    )
                else:
                    st.metric(
                        label="Cold Start",
                        value=f"{cold_ms:.1f} ms",
                    )

            with comparison_cols[1]:
                warm_ms = st.session_state.warm_latency * 1000
                if st.session_state.last_word_count and st.session_state.warm_latency > 0:
                    warm_wps = st.session_state.last_word_count / st.session_state.warm_latency
                    speedup = st.session_state.cold_latency / st.session_state.warm_latency if st.session_state.warm_latency > 0 else 0
                    st.metric(
                        label="Warm Start",
                        value=f"{warm_ms:.1f} ms",
                        delta=f"{warm_wps:.2f} WPS ({speedup:.1f}x faster)",
                    )
                else:
                    st.metric(
                        label="Warm Start",
                        value=f"{warm_ms:.1f} ms",
                    )

    with col_benchmark:
        st.session_state.benchmark_mode = st.checkbox(
            "Benchmark mode",
            value=st.session_state.benchmark_mode,
            help="Unload model before translation to measure cold start vs warm start performance"
        )

    # handle translation
    if translate_clicked and source_text.strip():
        request = TranslationRequest(
            src_lang=src_lang_code,
            tgt_lang=tgt_lang_code,
            source=source_text.strip(),
            model_id=selected_model,
        )

        word_count = len(source_text.strip().split())
        st.session_state.last_word_count = word_count

        with st.spinner("Translating..."):
            try:
                # force cold start if benchamrk mode
                if st.session_state.benchmark_mode and not USE_MOCK:
                    # Force cold start
                    unload_model(selected_model)

                    # Cold run
                    cold_response = translate_text(request)
                    st.session_state.cold_latency = cold_response.latency

                    # Warm run
                    warm_response = translate_text(request)
                    st.session_state.warm_latency = warm_response.latency

                    st.session_state.translation_result = warm_response.translated_value
                    st.session_state.last_latency = warm_response.latency
                    st.session_state.last_was_warm = True
                    st.session_state.show_comparison = True

                else:
                    # Normal single run
                    response = translate_text(request) if not USE_MOCK else mock_translate(request)

                    st.session_state.translation_result = response.translated_value
                    st.session_state.last_latency = response.latency
                    st.session_state.last_was_warm = response.model_was_warm
                    st.session_state.warm_latency = response.latency
                    st.session_state.show_comparison = False

                st.rerun()

            except requests.RequestException as e:
                st.error(f"Translation failed: {e}")
            except Exception as e:
                st.error(f"error: {e}")


if __name__ == "__main__":
    main()