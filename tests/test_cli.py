import pytest

import blindspot
from blindspot.cases import CaseStore, read_image
from blindspot.cli import main
from blindspot.model import Classifier

from conftest import make_classifier


@pytest.fixture
def photo(tmp_path, image, monkeypatch):
    monkeypatch.setattr(Classifier, "pretrained", classmethod(lambda *_a, **_k: make_classifier()))
    path = tmp_path / "photo.png"
    CaseStore._write_image(path, image)
    return str(path)


def test_sample_photo_ships_with_the_package():
    assert read_image(blindspot.SAMPLE_IMAGE).shape[2] == 3


def test_misspelled_label_is_rejected_with_a_suggestion(photo):
    with pytest.raises(SystemExit, match="Did you mean: apple"):
        main(["sweep", photo, "--label", "appel"])


def test_label_matching_ignores_case(photo, capsys):
    assert main(["sweep", photo, "--label", "APPLE"]) == 0
    assert "settings failed" in capsys.readouterr().out


def test_missing_image_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="could not read"):
        main(["breakpoints", str(tmp_path / "nope.png")])


def test_breakpoints_command_reports_and_saves(photo, tmp_path, capsys):
    store = tmp_path / "cases"
    assert main(["breakpoints", photo, "--save", "--store", str(store)]) == 0
    output = capsys.readouterr().out
    assert "directions broke" in output
    assert len(CaseStore(store).all()) == output.count("saved ")


def test_replay_reports_unknown_cases_clearly(tmp_path):
    with pytest.raises(SystemExit, match="no saved case"):
        main(["replay", "missing", "--store", str(tmp_path)])


def test_replay_flags_a_case_whose_model_is_gone(photo, tmp_path, image, monkeypatch, capsys):
    store = CaseStore(tmp_path / "cases")
    from blindspot.perturb import Perturbation

    store.save(image, Perturbation(gamma=2), make_classifier())

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("checkpoint.pt")

    monkeypatch.setattr(Classifier, "pretrained", classmethod(missing))
    assert main(["replay", "--store", str(store.root)]) == 1
    assert "CANNOT RUN" in capsys.readouterr().out


def test_app_only_listens_on_this_computer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("blindspot.cli.subprocess.call", lambda command, env: calls.append(command) or 0)
    assert main(["app", "--store", str(tmp_path)]) == 0
    command = calls[0]
    assert command[command.index("--server.address") + 1] == "localhost"
