import json

import numpy as np
import pytest

from blindspot.cases import CaseStore, is_failure, pixel_hash, read_image
from blindspot.cli import main
from blindspot.model import Prediction
from blindspot.perturb import Perturbation, apply
from blindspot.sweep import sweep

from conftest import make_classifier

SETTINGS = Perturbation(exposure=-1.5, blur=0.02, rotate=10)


def pred(index: int, label: str) -> list[Prediction]:
    return [Prediction(index, label, 0.9)]


def test_failure_uses_truth_when_known():
    assert is_failure(pred(0, "cat"), pred(0, "cat"), true_label="dog")
    assert not is_failure(pred(1, "dog"), pred(0, "cat"), true_label="cat")


def test_failure_without_truth_means_the_prediction_changed():
    assert is_failure(pred(0, "cat"), pred(1, "dog"))
    assert not is_failure(pred(0, "cat"), pred(0, "cat"))


def test_save_records_everything_needed_to_reproduce(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    case = store.save(image, SETTINGS, classifier, true_label="cat", note="dark")

    assert store.load(case.id) == case
    assert case.settings == SETTINGS
    assert case.model == classifier.identity
    assert {"numpy", "opencv", "torch", "torchvision"} <= set(case.environment)
    assert np.array_equal(store.image(case), image)

    clean, perturbed = classifier.predict([image, apply(image, SETTINGS)])
    assert case.clean_top == [p.to_dict() for p in clean]
    assert case.perturbed_top == [p.to_dict() for p in perturbed]


def test_saving_twice_is_idempotent(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    assert store.save(image, SETTINGS, classifier) == store.save(image, SETTINGS, classifier)
    assert len(store.all()) == 1


def test_different_settings_make_different_cases(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    store.save(image, SETTINGS, classifier)
    store.save(image, Perturbation(gamma=2), classifier)
    assert len(store.all()) == 2


def test_replay_reproduces_with_the_same_model(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    case = store.save(image, SETTINGS, classifier)
    result = store.replay(case, make_classifier())
    assert result.reproduced
    assert result.model_matches
    assert result.max_prob_delta < 1e-6
    assert result.environment_changes == {}


def test_replay_notices_a_different_model(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    case = store.save(image, SETTINGS, classifier)
    result = store.replay(case, make_classifier(seed=99))
    assert not result.reproduced
    assert not result.model_matches


def test_tampered_source_image_is_refused(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    case = store.save(image, SETTINGS, classifier)
    CaseStore._write_image(store.image_dir / case.image_file, 255 - image)
    with pytest.raises(ValueError, match="no longer matches"):
        store.replay(case, classifier)


def test_unknown_case_raises(tmp_path):
    with pytest.raises(KeyError):
        CaseStore(tmp_path).load("missing")


def test_case_files_are_plain_json(tmp_path, image, classifier):
    store = CaseStore(tmp_path)
    case = store.save(image, SETTINGS, classifier)
    data = json.loads((store.case_dir / f"{case.id}.json").read_text(encoding="utf-8"))
    assert data["perturbation"]["exposure"] == -1.5


def test_read_image_handles_non_ascii_paths(tmp_path, image):
    folder = tmp_path / "ảnh"
    folder.mkdir()
    CaseStore._write_image(folder / "x.png", image)
    assert pixel_hash(read_image(folder / "x.png")) == pixel_hash(image)


def test_read_image_rejects_garbage():
    with pytest.raises(ValueError):
        read_image(b"not an image")


def test_sweep_marks_failures_against_the_truth(image, classifier):
    grid = [Perturbation(exposure=-2), Perturbation(gamma=2)]
    result = sweep(image, classifier, grid, true_label="not a real class")
    assert [row.perturbation for row in result.rows] == grid
    assert len(result.failures) == 2


def test_replay_command_returns_zero_with_no_cases(tmp_path, capsys):
    assert main(["replay", "--store", str(tmp_path)]) == 0
    assert "no saved cases" in capsys.readouterr().out
