"""Interactive tool for probing an image classifier. Start it with `blindspot app`."""

from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st

from blindspot import SAMPLE_IMAGE
from blindspot.breakpoints import find_breakpoints
from blindspot.cases import DEFAULT_ROOT, CaseStore, is_failure, pixel_hash, read_image
from blindspot.model import Classifier, Prediction
from blindspot.perturb import AXES, Perturbation, apply
from blindspot.sweep import sweep

ARCHES = ["resnet18", "resnet50", "mobilenet_v3_small"]
MAX_SIDE = 1024  # keeps sliders responsive on large photos
NO_LABEL = "Not sure: flag changes from the clean prediction"

# key, label, format
SLIDERS = {
    "Lighting": [
        ("exposure", "Exposure (stops)", "%.1f"),
        ("contrast", "Contrast", "%.2f"),
        ("gamma", "Gamma", "%.2f"),
    ],
    "Blur": [
        ("blur", "Gaussian blur", "%.3f"),
        ("motion", "Motion blur", "%.3f"),
        ("motion_angle", "Motion angle", "%.0f"),
    ],
    "Framing": [
        ("zoom", "Zoom", "%.2f"),
        ("shift_x", "Shift horizontally", "%.2f"),
        ("shift_y", "Shift vertically", "%.2f"),
        ("rotate", "Rotate", "%.0f"),
    ],
}


@st.cache_resource(show_spinner="Loading model")
def load_classifier(arch: str, checkpoint: str | None) -> Classifier:
    return Classifier.pretrained(arch, checkpoint)


def load_settings(settings: Perturbation) -> None:
    st.session_state.update(settings.to_dict())


def use_sample() -> None:
    st.session_state["image_path"] = str(SAMPLE_IMAGE)


def load_case(store: CaseStore, case_id: str) -> None:
    """Put a saved case's image and settings back into the controls."""
    case = store.load(case_id)
    st.session_state.update(case.perturbation)
    st.session_state["image_path"] = str(store.image_dir / case.image_file)
    st.session_state["expected"] = case.true_label or NO_LABEL


def fit(image: np.ndarray) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / longest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def choose_image() -> np.ndarray | None:
    left, right = st.columns(2)
    upload = left.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    path = right.text_input("Or a path on this computer", key="image_path").strip().strip('"')
    right.button("Use the sample photo", on_click=use_sample)
    try:
        if path:
            return fit(read_image(path))
        if upload is not None:
            return fit(read_image(upload.getvalue()))
    except (OSError, ValueError) as error:
        st.error(f"Could not read that image: {error}")
    return None


def show_predictions(column, title: str, image: np.ndarray, top: list[Prediction]) -> None:
    column.subheader(title)
    column.image(image)
    column.dataframe(
        [{"Class": p.label, "Confidence": f"{p.prob:.1%}"} for p in top], hide_index=True
    )


def show_breakpoints(
    store: CaseStore, image: np.ndarray, classifier: Classifier, true_label: str | None
) -> None:
    st.caption(
        "Moves each setting away from normal until the prediction fails, "
        "then narrows in on the smallest change that does it."
    )
    token = (pixel_hash(image), str(classifier.identity), true_label)
    if st.button("Find breaking points"):
        with st.spinner("Searching"):
            st.session_state["breakpoints"] = (
                token,
                find_breakpoints(image, classifier, true_label=true_label),
            )
    stored = st.session_state.get("breakpoints")
    if not stored or stored[0] != token:
        return

    result = stored[1]
    if result.already_wrong:
        st.warning("The model is already wrong on the unmodified image.")
        return
    st.write(f"**{len(result.broken)} of {len(result.points)}** directions broke the model.")
    st.dataframe(
        [
            {
                "Direction": f"{p.direction.axis} toward {p.direction.limit:g}",
                "Breaks at": p.perturbation.describe() if p.broke else "Held to the limit",
                "Share of range": f"{p.severity:.0%}" if p.broke else "",
                "Prediction": p.top[0].label if p.broke else "",
                "Confidence": f"{p.top[0].prob:.1%}" if p.broke else "",
            }
            for p in result.points
        ],
        hide_index=True,
    )
    if result.broken:
        names = [p.perturbation.describe() for p in result.broken]
        chosen = st.selectbox("Breaking point", names)
        point = result.broken[names.index(chosen)]
        st.button("Show this breaking point", on_click=load_settings, args=(point.perturbation,))


def show_sweep(
    store: CaseStore, image: np.ndarray, classifier: Classifier, true_label: str | None
) -> None:
    st.caption("Tries every setting in the standard grid, one change at a time.")
    token = (pixel_hash(image), str(classifier.identity), true_label)
    if st.button("Run sweep"):
        with st.spinner("Sweeping"):
            st.session_state["sweep"] = (token, sweep(image, classifier, true_label=true_label))
    stored = st.session_state.get("sweep")
    if not stored or stored[0] != token:
        return

    result = stored[1]
    st.write(f"**{len(result.failures)} of {len(result.rows)}** settings failed.")
    st.dataframe(
        [
            {
                "Failed": "Yes" if row.failed else "",
                "Family": row.perturbation.family,
                "Setting": row.perturbation.describe(),
                "Prediction": row.top[0].label,
                "Confidence": f"{row.top[0].prob:.1%}",
            }
            for row in result.rows
        ],
        hide_index=True,
    )
    if result.failures and st.button("Save all failures"):
        for row in result.failures:
            store.save(image, row.perturbation, classifier, true_label=true_label, note="sweep")
        st.toast(f"Saved {len(result.failures)} cases")


def show_cases(store: CaseStore, classifier: Classifier) -> None:
    cases = store.all()
    if not cases:
        st.caption("Failures you save appear here, ready to replay.")
        return

    st.dataframe(
        [
            {
                "Case": case.id,
                "Model": case.model.get("arch"),
                "Settings": case.settings.describe(),
                "Clean": case.clean_top[0]["label"],
                "Perturbed": case.perturbed_top[0]["label"],
                "Saved": case.created_at,
            }
            for case in cases
        ],
        hide_index=True,
    )
    case_id = st.selectbox("Case", [case.id for case in cases])
    load_column, replay_column = st.columns(2)
    load_column.button("Load into controls", on_click=load_case, args=(store, case_id))
    if replay_column.button("Replay with current model"):
        result = store.replay(store.load(case_id), classifier)
        message = (
            f"Stored: {result.stored_top1}. Now: {result.replayed_top1}. "
            f"Largest confidence change: {result.max_prob_delta:.4f}."
        )
        (st.success if result.reproduced else st.warning)(
            ("Reproduced. " if result.reproduced else "Behavior changed. ") + message
        )
        if not result.model_matches:
            st.info("This case was saved with a different model.")


def main() -> None:
    st.set_page_config(page_title="Blindspot", layout="wide")
    st.title("Blindspot")
    st.caption("Find the lighting, blur, and framing that break an image classifier.")

    store = CaseStore(os.environ.get("BLINDSPOT_CASES", DEFAULT_ROOT))
    for key, value in Perturbation().to_dict().items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        arch = st.selectbox("Model", ARCHES)
        checkpoint = st.text_input("Fine-tuned checkpoint (optional)").strip().strip('"') or None
        try:
            classifier = load_classifier(arch, checkpoint)
        except (OSError, RuntimeError) as error:
            st.error(f"Could not load that model: {error}")
            st.stop()
        expected = st.selectbox("True class", [NO_LABEL, *sorted(set(classifier.labels))], key="expected")
        for group, sliders in SLIDERS.items():
            st.subheader(group)
            for key, label, number_format in sliders:
                axis = AXES[key]
                st.slider(label, axis.low, axis.high, step=axis.step, format=number_format, key=key)
        st.button("Reset", on_click=load_settings, args=(Perturbation(),))

    true_label = None if expected == NO_LABEL else expected
    image = choose_image()
    if image is None:
        st.info("Add an image or use the sample photo to start.")
        st.header("Saved cases")
        show_cases(store, classifier)
        return

    settings = Perturbation.from_dict({k: st.session_state[k] for k in Perturbation().to_dict()})
    perturbed = apply(image, settings)
    clean_top, perturbed_top = classifier.predict([image, perturbed])

    if settings.is_identity:
        st.info("Move a slider to perturb the image, or let Blindspot find the breaking points.")
    elif is_failure(clean_top, perturbed_top, true_label):
        st.error(f"Failure: the model now says **{perturbed_top[0].label}**.")
    else:
        st.success(f"Held: the model still says **{perturbed_top[0].label}**.")

    before, after = st.columns(2)
    show_predictions(before, "Original", image, clean_top)
    show_predictions(after, f"Perturbed ({settings.describe()})", perturbed, perturbed_top)

    if st.button("Save this case", disabled=settings.is_identity):
        case = store.save(image, settings, classifier, true_label=true_label)
        st.toast(f"Saved case {case.id}")

    breaks_tab, sweep_tab, cases_tab = st.tabs(["Breaking points", "Sweep", "Saved cases"])
    with breaks_tab:
        show_breakpoints(store, image, classifier, true_label)
    with sweep_tab:
        show_sweep(store, image, classifier, true_label)
    with cases_tab:
        show_cases(store, classifier)


main()
