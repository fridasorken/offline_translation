import streamlit as st
import requests
import time
import os
import pandas as pd
import io
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# Page config must be first Streamlit command
st.set_page_config(
    page_title="frontend",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
TRANSLATE_ENDPOINT = "/translate"

# Fallback model used for UI rendering when backend is not yet reachable.
# The real model list is fetched once the backend responds.
DEFAULT_FALLBACK_MODELS: dict[str, dict] = {
    "opus-mt-en-gmq": {
        "adapter": "transformers",
        "supported_pairs": [("en", "nob"), ("en", "nno")],
    }
}


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


@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def fetch_available_models(silent: bool = False) -> dict[str, dict]:
    """
    Fetch available models from the backend API.
    Returns dict mapping model_id to full model info including supported pairs.
    Falls back to empty dict if API is unavailable.
    Pass silent=True to suppress the warning (e.g. during background load attempts).
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
        if not silent:
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
    if "models_preloaded" not in st.session_state:
        st.session_state.models_preloaded = False
    if "csv_results" not in st.session_state:
        st.session_state.csv_results = None
    if "csv_translation_started" not in st.session_state:
        st.session_state.csv_translation_started = False


@st.cache_data
def get_supported_languages(model_id: str, supported_pairs: tuple) -> tuple[list[str], list[str]]:
    """
    Extract supported source and target languages from model info.
    Returns (source_langs, target_langs) as lists of language codes.
    Uses tuple for hashable cache key.
    """
    src_langs = sorted(set(pair[0] for pair in supported_pairs))
    tgt_langs = sorted(set(pair[1] for pair in supported_pairs))
    return src_langs, tgt_langs


@st.cache_data
def get_valid_target_languages(model_id: str, supported_pairs: tuple, src_lang_code: str) -> list[str]:
    """
    Get valid target languages for a given source language.
    Uses tuple for hashable cache key.
    """
    return sorted(set(pair[1] for pair in supported_pairs if pair[0] == src_lang_code))


@st.cache_data
def get_valid_source_languages(model_id: str, supported_pairs: tuple, tgt_lang_code: str) -> list[str]:
    """
    Get valid source languages for a given target language.
    Uses tuple for hashable cache key.
    """
    return sorted(set(pair[0] for pair in supported_pairs if pair[1] == tgt_lang_code))


@st.cache_data
def code_to_display_name(lang_code: str) -> str:
    """Convert language code to display name, fallback to code if not found."""
    lang = Language.from_code(lang_code)
    return lang.display_name if lang else lang_code


@st.cache_data
def display_name_to_code(display_name: str) -> str:
    """Convert display name to language code, fallback to display name if not found."""
    lang = Language.from_display_name(display_name)
    return lang.code if lang else display_name


@st.fragment
def translation_fragment(src_lang_code: str, tgt_lang_code: str, selected_model: str):
    """
    Fragment for translation input/output UI.
    This isolates reruns to just this section when text changes.
    """
    # input/output areas
    col_input, col_output = st.columns(2)

    with col_input:
        source_text = st.text_area(
            "Source text",
            height=200,
            placeholder="Enter text to translate...",
            label_visibility="collapsed",
            key="source_text_input",
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

    # translate button (always enabled)
    col_btn = st.columns(1)[0]

    with col_btn:
        translate_clicked = st.button(
            "Translate",
            type="primary",
            use_container_width=True,
        )

    # handle translation
    if translate_clicked and source_text.strip():
        word_count = len(source_text.strip().split())
        st.session_state.last_word_count = word_count

        try:
            # Step 1: ensure backend is reachable before doing anything
            if not USE_MOCK:
                with st.spinner("Waiting for backend..."):
                    if not wait_for_backend(timeout=120):
                        st.error("Backend is not responding. Please ensure the backend service is running.")
                        st.stop()

                # If we were using the fallback model list, now fetch the real one
                if not st.session_state.available_models:
                    st.session_state.available_models = fetch_available_models()

            # Resolve model_id
            backend_offline = not USE_MOCK and not st.session_state.available_models
            if backend_offline and st.session_state.available_models:
                resolved_model_id = list(st.session_state.available_models.keys())[0]
            else:
                resolved_model_id = selected_model

            request = TranslationRequest(
                src_lang=src_lang_code,
                tgt_lang=tgt_lang_code,
                source=source_text.strip(),
                model_id=resolved_model_id,
            )

            # Step 2: translate
            with st.spinner("Translating..."):
                # force cold start if benchmark mode
                if st.session_state.benchmark_mode and not USE_MOCK:
                    unload_model(resolved_model_id)

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

            st.rerun(scope="fragment")

        except requests.RequestException as e:
            st.error(f"Translation failed: {e}")
        except Exception as e:
            st.error(f"error: {e}")


@st.cache_data
def get_language_options_for_model(model_id: str, supported_pairs: tuple) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Get all language options for a model, fully processed and ready for UI.
    Returns: (src_lang_codes, tgt_lang_codes, src_lang_options, tgt_lang_options)
    Cached per model to avoid recomputation on every render.
    """
    # Extract codes
    src_lang_codes = sorted(set(pair[0] for pair in supported_pairs))
    tgt_lang_codes = sorted(set(pair[1] for pair in supported_pairs))

    # Convert to display names
    src_lang_options = sorted([code_to_display_name(code) for code in src_lang_codes])
    tgt_lang_options = sorted([code_to_display_name(code) for code in tgt_lang_codes])

    return src_lang_codes, tgt_lang_codes, src_lang_options, tgt_lang_options


def batch_translate_csv(
    df: pd.DataFrame,
    text_column: str,
    src_lang: str,
    tgt_lang: str,
    model_id: str,
    benchmark_mode: bool,
    progress_bar
) -> pd.DataFrame:
    """
    Batch translate a CSV file with progress tracking.

    Args:
        df: Input DataFrame
        text_column: Column name containing text to translate
        src_lang: Source language code
        tgt_lang: Target language code
        model_id: Model ID to use for translation
        benchmark_mode: If True, do cold+warm translation per row
        progress_bar: Streamlit progress bar object

    Returns:
        DataFrame with translation results
    """
    results = []
    total_rows = len(df)

    for idx, row in df.iterrows():
        source_text = str(row[text_column])

        if not source_text or source_text == "nan":
            # Skip empty rows
            result_row = {
                "source": source_text,
                "translation": "",
            }
            if benchmark_mode:
                result_row["t_cold_ms"] = None
                result_row["t_warm_ms"] = None
            else:
                result_row["latency_ms"] = None
            results.append(result_row)
            progress_bar.progress((idx + 1) / total_rows)
            continue

        request = TranslationRequest(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            source=source_text,
            model_id=model_id,
        )

        try:
            if benchmark_mode:
                # Unload model for cold start
                if not USE_MOCK:
                    unload_model(model_id)

                # Cold translation
                cold_response = translate_text(request) if not USE_MOCK else mock_translate(request)

                # Warm translation
                warm_response = translate_text(request) if not USE_MOCK else mock_translate(request)

                result_row = {
                    "source": source_text,
                    "translation": warm_response.translated_value,
                    "t_cold_ms": cold_response.latency * 1000,
                    "t_warm_ms": warm_response.latency * 1000,
                }
            else:
                # Normal single translation
                response = translate_text(request) if not USE_MOCK else mock_translate(request)

                result_row = {
                    "source": source_text,
                    "translation": response.translated_value,
                    "latency_ms": response.latency * 1000,
                }

            results.append(result_row)

        except Exception as e:
            # Handle translation errors
            result_row = {
                "source": source_text,
                "translation": f"ERROR: {str(e)}",
            }
            if benchmark_mode:
                result_row["t_cold_ms"] = None
                result_row["t_warm_ms"] = None
            else:
                result_row["latency_ms"] = None
            results.append(result_row)

        # Update progress
        progress_bar.progress((idx + 1) / total_rows)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Merge with original dataframe (keep all original columns)
    final_df = df.copy()
    final_df["translation"] = results_df["translation"]

    if benchmark_mode:
        final_df["t_cold_ms"] = results_df["t_cold_ms"]
        final_df["t_warm_ms"] = results_df["t_warm_ms"]
    else:
        final_df["latency_ms"] = results_df["latency_ms"]

    return final_df


def main():

    init_session_state()

    # Attempt a quick, non-blocking model fetch on every load until it succeeds.
    # This means the UI renders immediately; the translate button handles waiting.
    if not USE_MOCK and not st.session_state.available_models:
        st.session_state.available_models = fetch_available_models(silent=True)

    # Determine which models to show, falling back to a placeholder so the
    # UI always renders even when the backend hasn't started yet.
    if USE_MOCK:
        available_models_dict = {"mock-model": {"adapter": "mock", "supported_pairs": [("en", "nob"), ("nob", "en")]}}
    elif st.session_state.available_models:
        available_models_dict = st.session_state.available_models
    else:
        available_models_dict = DEFAULT_FALLBACK_MODELS

    backend_offline = not USE_MOCK and not st.session_state.available_models

    model_ids = list(available_models_dict.keys())

    # Preload language options for all models to warm the cache (only on first load)
    # This ensures the UI renders smoothly without delays between widgets
    if not st.session_state.models_preloaded:
        with st.spinner("Loading models..."):
            for model_id in model_ids:
                supported_pairs = tuple(available_models_dict[model_id].get("supported_pairs", []))
                get_language_options_for_model(model_id, supported_pairs)
            st.session_state.models_preloaded = True

    # Now all caches are warm - UI will render instantly from here on

    # Create tabs for Text, CSV, and Evaluation
    tab_text, tab_csv, tab_eval = st.tabs(["Text", "CSV", "Evaluation"])

    with tab_text:
        # model selection
        selected_model = st.selectbox(
            "Model",
            options=model_ids,
            index=0,
            key="selected_model",
        )

        # get model info and supported languages (cached per model)
        model_info = available_models_dict[selected_model]
        supported_pairs_tuple = tuple(model_info.get("supported_pairs", []))
        src_lang_codes, tgt_lang_codes, src_lang_options, tgt_lang_options = get_language_options_for_model(
            selected_model, supported_pairs_tuple
        )

        # Initialize language selections if not set or invalid for current model
        if st.session_state.selected_src_lang is None or display_name_to_code(st.session_state.selected_src_lang) not in src_lang_codes:
            st.session_state.selected_src_lang = src_lang_options[0] if src_lang_options else None
        if st.session_state.selected_tgt_lang is None or display_name_to_code(st.session_state.selected_tgt_lang) not in tgt_lang_codes:
            st.session_state.selected_tgt_lang = tgt_lang_options[0] if tgt_lang_options else None

        if not src_lang_options or not tgt_lang_options:
            st.error(f"Model {selected_model} has no supported language pairs.")
            st.stop()

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
            valid_tgt_codes = get_valid_target_languages(selected_model, supported_pairs_tuple, current_src_code)
            # Sort target options alphabetically
            valid_tgt_options = sorted([code_to_display_name(code) for code in valid_tgt_codes])

            if not valid_tgt_options:
                st.error(f"No target languages available for source language {st.session_state.selected_src_lang}")
                st.stop()

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

        # Translation UI fragment (isolated reruns for better performance)
        translation_fragment(src_lang_code, tgt_lang_code, selected_model)

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
                            delta_color="off",
                        )
                    else:
                        st.metric(
                            label="Cold start",
                            value=f"{cold_ms:.1f} ms",
                        )

                with comparison_cols[1]:
                    warm_ms = st.session_state.warm_latency * 1000
                    if st.session_state.last_word_count and st.session_state.warm_latency > 0:
                        warm_wps = st.session_state.last_word_count / st.session_state.warm_latency
                        speedup = st.session_state.cold_latency / st.session_state.warm_latency if st.session_state.warm_latency > 0 else 0
                        st.metric(
                            label="Warm start",
                            value=f"{warm_ms:.1f} ms",
                            delta=f"{warm_wps:.2f} WPS ({speedup:.1f}x faster)",
                        )
                    else:
                        st.metric(
                            label="Warm Start",
                            value=f"{warm_ms:.1f} ms",
                        )

    with tab_csv:
        # Model selection for CSV
        selected_model_csv = st.selectbox(
            "Model",
            options=model_ids,
            index=0,
            key="selected_model_csv",
        )

        # Get model info for CSV tab
        model_info_csv = available_models_dict[selected_model_csv]
        supported_pairs_tuple_csv = tuple(model_info_csv.get("supported_pairs", []))
        src_lang_codes_csv, tgt_lang_codes_csv, src_lang_options_csv, tgt_lang_options_csv = get_language_options_for_model(
            selected_model_csv, supported_pairs_tuple_csv
        )

        # Language selectors for CSV
        col_src_csv, col_tgt_csv = st.columns(2)

        with col_src_csv:
            src_lang_csv = st.selectbox(
                "Source language",
                options=src_lang_options_csv,
                index=0,
                key="src_lang_csv",
            )
            src_lang_code_csv = display_name_to_code(src_lang_csv)

        with col_tgt_csv:
            # Filter valid target languages
            valid_tgt_codes_csv = get_valid_target_languages(selected_model_csv, supported_pairs_tuple_csv, src_lang_code_csv)
            valid_tgt_options_csv = sorted([code_to_display_name(code) for code in valid_tgt_codes_csv])

            if valid_tgt_options_csv:
                tgt_lang_csv = st.selectbox(
                    "Target language",
                    options=valid_tgt_options_csv,
                    index=0,
                    key="tgt_lang_csv",
                )
                tgt_lang_code_csv = display_name_to_code(tgt_lang_csv)
            else:
                st.error(f"No target languages available for source language {src_lang_csv}")
                st.stop()

        # CSV file uploader
        st.divider()
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file with text to translate. The first column will be translated."
        )

        if uploaded_file is not None:
            # Read and preview the CSV
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"**Preview** ({len(df)} rows):")
                st.dataframe(df.head(10), use_container_width=True)

                # Column selector
                text_column = st.selectbox(
                    "Select column to translate",
                    options=df.columns.tolist(),
                    index=0,
                    key="text_column_selector"
                )

                # Translate button
                if st.button("Translate CSV", type="primary", use_container_width=True):
                    # Perform batch translation
                    with st.spinner("Translating..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            # Ensure backend is reachable
                            if not USE_MOCK:
                                status_text.text("Waiting for backend...")
                                if not wait_for_backend(timeout=120):
                                    st.error("Backend is not responding. Please ensure the backend service is running.")
                                    st.stop()

                            status_text.text(f"Translating {len(df)} rows...")

                            # Perform batch translation
                            results_df = batch_translate_csv(
                                df=df,
                                text_column=text_column,
                                src_lang=src_lang_code_csv,
                                tgt_lang=tgt_lang_code_csv,
                                model_id=selected_model_csv,
                                benchmark_mode=st.session_state.benchmark_mode,
                                progress_bar=progress_bar
                            )

                            # Store results in session state
                            st.session_state.csv_results = results_df

                            status_text.text("Translation complete!")
                            st.success(f"Successfully translated {len(df)} rows!")

                        except Exception as e:
                            st.error(f"Translation failed: {e}")

                # Display results if available
                if st.session_state.csv_results is not None:
                    st.divider()
                    st.subheader("Translation Results")

                    # Show preview
                    st.write(f"**Preview** ({len(st.session_state.csv_results)} rows):")
                    st.dataframe(st.session_state.csv_results.head(20), use_container_width=True)

                    # Download button
                    csv_buffer = io.StringIO()
                    st.session_state.csv_results.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="Download Translated CSV",
                        data=csv_data,
                        file_name="translated_output.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )

                    # Show summary statistics if benchmark mode
                    if st.session_state.benchmark_mode and "t_cold_ms" in st.session_state.csv_results.columns:
                        st.divider()
                        st.subheader("Benchmark Statistics")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            avg_cold = st.session_state.csv_results["t_cold_ms"].mean()
                            st.metric("Avg Cold Start", f"{avg_cold:.1f} ms")

                        with col2:
                            avg_warm = st.session_state.csv_results["t_warm_ms"].mean()
                            st.metric("Avg Warm Start", f"{avg_warm:.1f} ms")

                        with col3:
                            speedup = avg_cold / avg_warm if avg_warm > 0 else 0
                            st.metric("Avg Speedup", f"{speedup:.2f}x")

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    with tab_eval:
        st.subheader("Translation Evaluation")
        st.info("Evaluation metrics and tools will be available here.")

    # Benchmark mode toggle (placed at the bottom)
    st.divider()
    st.session_state.benchmark_mode = st.toggle(
        "Benchmark mode",
        value=st.session_state.benchmark_mode,
        help="Unload model before translation to measure cold start vs warm start performance"
    )


if __name__ == "__main__":
    main()