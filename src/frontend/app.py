import streamlit as st
import requests
import time
import os
import pandas as pd
import io
from typing import Optional
from dataclasses import dataclass, asdict
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
EVALUATE_ENDPOINT = "/evaluate"

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
        return next((lang for lang in cls if lang.code == code), None)

    @classmethod
    def get_display_names(cls) -> list[str]:
        return [lang.display_name for lang in cls]

    @classmethod
    def from_display_name(cls, name: str) -> Optional["Language"]:
        return next((lang for lang in cls if lang.display_name == name), None)


@dataclass
class TranslationRequest:
    """Request payload for the translation API."""
    src_lang: str
    tgt_lang: str
    source: str
    model_id: str
    
    def to_dict(self) -> dict:
        return asdict(self)


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


@dataclass
class EvaluateItem:
    """Single item for evaluation."""
    source: str
    reference: Optional[str] = None
    item_id: Optional[str] = None


@dataclass
class EvaluateRequest:
    """Request payload for the evaluation API."""
    model_id: str
    src_lang: str
    tgt_lang: str
    items: list[EvaluateItem]
    metrics: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "items": [asdict(item) for item in self.items],
            "metrics": self.metrics,
        }


@dataclass
class EvaluateItemResult:
    """Single item result from evaluation."""
    source: str
    translated_value: str
    latency_ms: int
    metrics: dict[str, float]
    reference: Optional[str] = None
    item_id: Optional[str] = None
    cpu_percent_per_core: Optional[float] = None
    ram_mean_mb: Optional[float] = None
    ram_peak_mb: Optional[float] = None


@dataclass
class EvaluateResponse:
    """Response from the evaluation API."""
    model_id: str
    src_lang: str
    tgt_lang: str
    results: list[EvaluateItemResult]
    aggregates: dict[str, float]
    average_latency_ms: float
    baseline_rss_mb: Optional[float] = None


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


def evaluate_translations(request: EvaluateRequest) -> EvaluateResponse:
    """
    Send evaluation request to the API.
    """
    url = f"{API_BASE_URL}{EVALUATE_ENDPOINT}"

    response = requests.post(
        url,
        json=request.to_dict(),
        headers={"Content-Type": "application/json"},
        timeout=300,  # Longer timeout for evaluation (can take time with COMET)
    )
    response.raise_for_status()

    data = response.json()

    # Parse results
    results = [
        EvaluateItemResult(
            source=item["source"],
            translated_value=item["translated_value"],
            latency_ms=item["latency_ms"],
            metrics=item.get("metrics", {}),
            reference=item.get("reference"),
            item_id=item.get("item_id"),
            cpu_percent_per_core=item.get("cpu_percent_per_core"),
            ram_mean_mb=item.get("ram_mean_mb"),
            ram_peak_mb=item.get("ram_peak_mb"),
        )
        for item in data.get("results", [])
    ]

    return EvaluateResponse(
        model_id=data["model_id"],
        src_lang=data["src_lang"],
        tgt_lang=data["tgt_lang"],
        results=results,
        aggregates=data.get("aggregates", {}),
        average_latency_ms=data.get("average_latency_ms", 0.0),
        baseline_rss_mb=data.get("baseline_rss_mb"),
    )


def init_session_state():
    """Initialise session state variables."""
    defaults = {
        "translation_result": None,
        "last_latency": None,
        "last_was_warm": None,
        "last_word_count": None,
        "available_models": None,
        "selected_src_lang": None,
        "selected_tgt_lang": None,
        "models_preloaded": False,
        "csv_results": None,
        "csv_translation_started": False,
        "eval_results": None,
        "eval_selected_metrics": ["bleu", "chrf"],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


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
                response = translate_text(request) if not USE_MOCK else mock_translate(request)

                st.session_state.translation_result = response.translated_value
                st.session_state.last_latency = response.latency
                st.session_state.last_was_warm = response.model_was_warm

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
) -> pd.DataFrame:
    """
    Batch translate a CSV file.

    Args:
        df: Input DataFrame
        text_column: Column name containing text to translate
        src_lang: Source language code
        tgt_lang: Target language code
        model_id: Model ID to use for translation

    Returns:
        DataFrame with translation results
    """
    results = []

    for idx, row in df.iterrows():
        source_text = str(row[text_column])

        if not source_text or source_text == "nan":
            # Skip empty rows
            result_row = {
                "source": source_text,
                "translation": "",
                "latency_ms": None,
            }
            results.append(result_row)
            continue

        request = TranslationRequest(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            source=source_text,
            model_id=model_id,
        )

        try:
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
                "latency_ms": None,
            }
            results.append(result_row)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Merge with original dataframe (keep all original columns)
    final_df = df.copy()
    final_df["translation"] = results_df["translation"]
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

    # Create tabs for Single, Batch, and Evaluation
    tab_text, tab_csv, tab_eval = st.tabs(["Single", "Batch", "Evaluation"])

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

        # Display performance metrics
        if st.session_state.last_latency is not None:
            st.divider()
            st.subheader("Metrics")

            metric_cols = st.columns(2)

            with metric_cols[0]:
                latency_ms = st.session_state.last_latency * 1000
                st.metric(
                    label="Latency",
                    value=f"{latency_ms:.1f} ms",
                )

            with metric_cols[1]:
                if st.session_state.last_word_count and st.session_state.last_latency > 0:
                    wps = st.session_state.last_word_count / st.session_state.last_latency
                    st.metric(
                        label="Throughput",
                        value=f"{wps:.2f} WPS",
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
                    try:
                        # Ensure backend is reachable
                        if not USE_MOCK:
                            with st.spinner("Waiting for backend..."):
                                if not wait_for_backend(timeout=120):
                                    st.error("Backend is not responding. Please ensure the backend service is running.")
                                    st.stop()

                        # Perform batch translation
                        results_df = batch_translate_csv(
                            df=df,
                            text_column=text_column,
                            src_lang=src_lang_code_csv,
                            tgt_lang=tgt_lang_code_csv,
                            model_id=selected_model_csv,
                        )

                        # Store results in session state
                        st.session_state.csv_results = results_df
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

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    with tab_eval:
        # Model selection for evaluation
        selected_model_eval = st.selectbox(
            "Model",
            options=model_ids,
            index=0,
            key="selected_model_eval",
        )

        # Get model info for evaluation tab
        model_info_eval = available_models_dict[selected_model_eval]
        supported_pairs_tuple_eval = tuple(model_info_eval.get("supported_pairs", []))
        src_lang_codes_eval, tgt_lang_codes_eval, src_lang_options_eval, tgt_lang_options_eval = get_language_options_for_model(
            selected_model_eval, supported_pairs_tuple_eval
        )

        # Language selectors for evaluation
        col_src_eval, col_tgt_eval = st.columns(2)

        with col_src_eval:
            src_lang_eval = st.selectbox(
                "Source language",
                options=src_lang_options_eval,
                index=0,
                key="src_lang_eval",
            )
            src_lang_code_eval = display_name_to_code(src_lang_eval)

        with col_tgt_eval:
            # Filter valid target languages
            valid_tgt_codes_eval = get_valid_target_languages(selected_model_eval, supported_pairs_tuple_eval, src_lang_code_eval)
            valid_tgt_options_eval = sorted([code_to_display_name(code) for code in valid_tgt_codes_eval])

            if valid_tgt_options_eval:
                tgt_lang_eval = st.selectbox(
                    "Target language",
                    options=valid_tgt_options_eval,
                    index=0,
                    key="tgt_lang_eval",
                )
                tgt_lang_code_eval = display_name_to_code(tgt_lang_eval)
            else:
                st.error(f"No target languages available for source language {src_lang_eval}")
                st.stop()

        # Metrics selection
        st.divider()
        st.write("**Metrics**")

        # Available metrics
        AVAILABLE_METRICS = {
            "bleu": "BLEU (requires reference)",
            "chrf": "chrF (requires reference)",
            "ter": "TER (requires reference)",
            "comet": "COMET (requires reference)",
            "cometkiwi": "COMET-KIWI (reference-free)",
        }

        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            use_bleu = st.checkbox("BLEU", value="bleu" in st.session_state.eval_selected_metrics, key="use_bleu")
            use_chrf = st.checkbox("chrF", value="chrf" in st.session_state.eval_selected_metrics, key="use_chrf")
        with col_metric2:
            use_ter = st.checkbox("TER", value="ter" in st.session_state.eval_selected_metrics, key="use_ter")
            use_comet = st.checkbox("COMET", value="comet" in st.session_state.eval_selected_metrics, key="use_comet")
        with col_metric3:
            use_cometkiwi = st.checkbox("COMET-KIWI", value="cometkiwi" in st.session_state.eval_selected_metrics, key="use_cometkiwi")

        selected_metrics = [
            name for name, enabled in [
                ("bleu", use_bleu),
                ("chrf", use_chrf),
                ("ter", use_ter),
                ("comet", use_comet),
                ("cometkiwi", use_cometkiwi),
            ] if enabled
        ]

        st.session_state.eval_selected_metrics = selected_metrics

        if not selected_metrics:
            st.warning("Please select at least one metric.")

        # CSV file uploader for evaluation data
        st.divider()
        uploaded_eval_file = st.file_uploader(
            "Upload evaluation CSV",
            type=["csv"],
            help="Upload a CSV file with 'source' and 'reference' columns. Optional 'item_id' column for tracking.",
            key="eval_file_uploader"
        )

        if uploaded_eval_file is not None:
            try:
                df_eval = pd.read_csv(uploaded_eval_file)
                st.write(f"**Preview** ({len(df_eval)} rows):")
                st.dataframe(df_eval.head(10), use_container_width=True)

                if "source" not in df_eval.columns:
                    st.error("CSV must contain a 'source' column.")
                    st.stop()

                # Check for reference column
                has_reference = "reference" in df_eval.columns
                has_item_id = "item_id" in df_eval.columns

                reference_required = any(metric in selected_metrics for metric in ["bleu", "chrf", "ter", "comet"])
                if reference_required and not has_reference:
                    st.error("Selected metrics require a 'reference' column in the CSV.")
                    st.stop()

                # Evaluate button
                if st.button("Evaluate", type="primary", use_container_width=True, disabled=not selected_metrics):
                    try:
                        # Ensure backend is reachable
                        if not USE_MOCK:
                            with st.spinner("Waiting for backend..."):
                                if not wait_for_backend(timeout=120):
                                    st.error("Backend is not responding. Please ensure the backend service is running.")
                                    st.stop()

                        # Build evaluation request
                        eval_items = [
                            EvaluateItem(
                                source=source,
                                reference=str(row["reference"]) if has_reference and pd.notna(row.get("reference")) else None,
                                item_id=str(row["item_id"]) if has_item_id and pd.notna(row.get("item_id")) else None,
                            )
                            for _, row in df_eval.iterrows()
                            if (source := str(row["source"])) and source != "nan"
                        ]

                        if not eval_items:
                            st.error("No valid items to evaluate.")
                            st.stop()

                        # Call evaluation API
                        with st.spinner(f"Evaluating {len(eval_items)} items with {len(selected_metrics)} metrics..."):
                            eval_request = EvaluateRequest(
                                model_id=selected_model_eval,
                                src_lang=src_lang_code_eval,
                                tgt_lang=tgt_lang_code_eval,
                                items=eval_items,
                                metrics=selected_metrics,
                            )

                            eval_response = evaluate_translations(eval_request)

                        # Store results in session state
                        st.session_state.eval_results = eval_response
                        st.success(f"Successfully evaluated {len(eval_items)} items!")

                    except requests.RequestException as e:
                        st.error(f"Evaluation failed: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")

                # Display results if available
                if st.session_state.eval_results is not None:
                    st.divider()
                    st.subheader("Results")

                    eval_resp: EvaluateResponse = st.session_state.eval_results

                    # Display aggregate metrics
                    st.write("**Aggregate metrics**")

                    # Group aggregates by metric type
                    metric_names = {key.replace("_mean", "") for key in eval_resp.aggregates if "_mean" in key}

                    # Display metrics in columns
                    num_metrics = len(metric_names)
                    if num_metrics > 0:
                        cols = st.columns(min(num_metrics, 4))
                        for idx, metric_name in enumerate(sorted(metric_names)):
                            with cols[idx % len(cols)]:
                                mean_val = eval_resp.aggregates.get(f"{metric_name}_mean", 0.0)
                                median_val = eval_resp.aggregates.get(f"{metric_name}_median", 0.0)
                                stdev_val = eval_resp.aggregates.get(f"{metric_name}_stdev", 0.0)

                                st.metric(
                                    label=metric_name.upper(),
                                    value=f"{mean_val:.4f}",
                                    delta=f"σ={stdev_val:.4f}",
                                    delta_color="off",
                                )

                    # Performance metrics
                    st.divider()
                    st.write("**Performance Metrics**")

                    perf_cols = st.columns(4)
                    with perf_cols[0]:
                        st.metric("Avg Latency", f"{eval_resp.average_latency_ms:.1f} ms")
                    with perf_cols[1]:
                        cpu_mean = eval_resp.aggregates.get("cpu_percent_per_core_mean")
                        if cpu_mean is not None:
                            st.metric("Avg CPU %", f"{cpu_mean:.1f}%")
                    with perf_cols[2]:
                        ram_mean = eval_resp.aggregates.get("ram_mean_mb_mean")
                        if ram_mean is not None:
                            st.metric("Avg RAM", f"{ram_mean:.1f} MB")
                    with perf_cols[3]:
                        ram_peak = eval_resp.aggregates.get("ram_peak_mb_mean")
                        if ram_peak is not None:
                            st.metric("Peak RAM", f"{ram_peak:.1f} MB")

                    # Per-item results table
                    st.divider()
                    st.write(f"**Per-Item Results** ({len(eval_resp.results)} items)")

                    # Build dataframe for display
                    results_data = []
                    for item in eval_resp.results:
                        row_data = {
                            "item_id": item.item_id or "-",
                            "source": item.source[:50] + "..." if len(item.source) > 50 else item.source,
                            "reference": (item.reference[:50] + "..." if len(item.reference) > 50 else item.reference) if item.reference else "-",
                            "translation": item.translated_value[:50] + "..." if len(item.translated_value) > 50 else item.translated_value,
                            "latency_ms": item.latency_ms,
                        }
                        # Add metric scores
                        for metric_name, metric_value in item.metrics.items():
                            row_data[metric_name] = f"{metric_value:.4f}"

                        # Add resource metrics
                        if item.cpu_percent_per_core is not None:
                            row_data["cpu_%"] = f"{item.cpu_percent_per_core:.1f}"
                        if item.ram_mean_mb is not None:
                            row_data["ram_mb"] = f"{item.ram_mean_mb:.1f}"

                        results_data.append(row_data)

                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)

                    # Download button for full results
                    st.divider()

                    # Build full CSV with all columns
                    full_results_data = []
                    for item in eval_resp.results:
                        row_data = {
                            "item_id": item.item_id or "",
                            "source": item.source,
                            "reference": item.reference or "",
                            "translation": item.translated_value,
                            "latency_ms": item.latency_ms,
                        }
                        # Add all metrics
                        for metric_name, metric_value in item.metrics.items():
                            row_data[metric_name] = metric_value

                        # Add resource metrics
                        if item.cpu_percent_per_core is not None:
                            row_data["cpu_percent_per_core"] = item.cpu_percent_per_core
                        if item.ram_mean_mb is not None:
                            row_data["ram_mean_mb"] = item.ram_mean_mb
                        if item.ram_peak_mb is not None:
                            row_data["ram_peak_mb"] = item.ram_peak_mb

                        full_results_data.append(row_data)

                    full_results_df = pd.DataFrame(full_results_data)

                    csv_buffer = io.StringIO()
                    full_results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="Download Evaluation Results (CSV)",
                        data=csv_data,
                        file_name="evaluation_results.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )

                    # Visualizations
                    if len(eval_resp.results) > 1:
                        st.divider()
                        st.subheader("Visualizations")

                        # Latency distribution
                        st.write("**Latency Distribution**")
                        latency_data = pd.DataFrame({
                            "Item": [item.item_id or f"Item {i+1}" for i, item in enumerate(eval_resp.results)],
                            "Latency (ms)": [item.latency_ms for item in eval_resp.results]
                        })
                        st.bar_chart(latency_data.set_index("Item"))

                        # Metric distributions
                        if metric_names:
                            st.write("**Metric Distributions**")
                            metric_tabs = st.tabs(list(sorted(metric_names)))

                            for idx, metric_name in enumerate(sorted(metric_names)):
                                with metric_tabs[idx]:
                                    metric_data = pd.DataFrame({
                                        "Item": [item.item_id or f"Item {i+1}" for i, item in enumerate(eval_resp.results)],
                                        metric_name.upper(): [item.metrics.get(metric_name, 0.0) for item in eval_resp.results]
                                    })
                                    st.bar_chart(metric_data.set_index("Item"))

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")


if __name__ == "__main__":
    main()