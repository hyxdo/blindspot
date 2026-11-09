from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import blindspot
from blindspot.cases import CaseStore
from blindspot.model import Classifier

from conftest import make_classifier

APP = str(Path(blindspot.__file__).with_name("app.py"))


@pytest.fixture
def app(tmp_path, monkeypatch, image):
    monkeypatch.setenv("BLINDSPOT_CASES", str(tmp_path / "store"))
    monkeypatch.setattr(Classifier, "pretrained", classmethod(lambda *_a, **_k: make_classifier()))
    st.cache_resource.clear()
    CaseStore._write_image(tmp_path / "input.png", image)
    test = AppTest.from_file(APP, default_timeout=30)
    test.run()
    assert not test.exception
    test.text_input(key="image_path").set_value(str(tmp_path / "input.png")).run()
    assert not test.exception
    return test


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


def test_opens_without_an_image(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINDSPOT_CASES", str(tmp_path))
    monkeypatch.setattr(Classifier, "pretrained", classmethod(lambda *_a, **_k: make_classifier()))
    st.cache_resource.clear()
    test = AppTest.from_file(APP, default_timeout=30).run()
    assert not test.exception
    assert any("Add an image" in info.value for info in test.info)


def test_bad_path_shows_an_error(app):
    app.text_input(key="image_path").set_value("does-not-exist.png").run()
    assert not app.exception
    assert any("Could not read" in error.value for error in app.error)


def test_save_requires_a_change_then_stores_a_case(app, tmp_path):
    assert button(app, "Save this case").disabled
    app.slider(key="exposure").set_value(-3.0).run()
    assert not button(app, "Save this case").disabled
    button(app, "Save this case").click().run()
    assert not app.exception
    cases = CaseStore(tmp_path / "store").all()
    assert len(cases) == 1
    assert cases[0].perturbation["exposure"] == -3.0


def test_loading_a_case_restores_controls_and_replays(app, tmp_path):
    app.slider(key="blur").set_value(0.04).run()
    button(app, "Save this case").click().run()
    button(app, "Reset").click().run()
    assert app.slider(key="blur").value == 0.0

    button(app, "Load into controls").click().run()
    assert not app.exception
    assert app.slider(key="blur").value == 0.04
    assert app.text_input(key="image_path").value.endswith(".png")

    button(app, "Replay with current model").click().run()
    assert not app.exception
    assert any("Reproduced" in success.value for success in app.success)


def test_sweep_can_save_every_failure(app, tmp_path):
    app.selectbox(key="expected").set_value("apple").run()
    button(app, "Run sweep").click().run()
    assert not app.exception
    assert any("settings failed" in text.value for text in app.markdown)
    labels = [b.label for b in app.button]
    if "Save all failures" in labels:
        button(app, "Save all failures").click().run()
        assert not app.exception
        assert CaseStore(tmp_path / "store").all()
