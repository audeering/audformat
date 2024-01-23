import pandas as pd
import pytest

import audformat
from audformat import testing


def test_errors():
    db = testing.create_db()

    with pytest.raises(audformat.errors.BadIdError):
        db["table"] = audformat.Table(split_id="bad")

    with pytest.raises(audformat.errors.BadIdError):
        db["table"] = audformat.Table(media_id="bad")

    with pytest.raises(audformat.errors.BadIdError):
        db["files"]["column"] = audformat.Column(scheme_id="bad")

    with pytest.raises(audformat.errors.BadIdError):
        db["files"]["column"] = audformat.Column(rater_id="bad")

    with pytest.raises(FileNotFoundError):
        audformat.Database.load("./bad/path")

    with pytest.raises(audformat.errors.BadTypeError):
        db.media["media"] = audformat.Rater()

    with pytest.raises(audformat.errors.BadTypeError):
        db.splits["split"] = audformat.Media()

    with pytest.raises(audformat.errors.BadTypeError):
        db.schemes["table"] = audformat.Split()

    with pytest.raises(audformat.errors.BadTypeError):
        db.schemes["scheme"] = audformat.Table()

    with pytest.raises(audformat.errors.BadTypeError):
        db.raters["rater"] = audformat.Scheme()

    with pytest.raises(audformat.errors.BadValueError):
        db.raters["rater"] = audformat.Rater("bad")

    with pytest.raises(audformat.errors.BadValueError):
        db.splits["split"] = audformat.Split("bad")

    with pytest.raises(audformat.errors.BadValueError):
        db.schemes["scheme"] = audformat.Scheme("bad")

    with pytest.raises(audformat.errors.BadValueError):
        audformat.Database("foo", "internal", "bad")

    with pytest.raises(audformat.errors.BadKeyError):
        db["bad"]

    with pytest.raises(audformat.errors.BadKeyError):
        db.media["bad"]

    with pytest.raises(audformat.errors.BadKeyError):
        db.misc_tables["bad"]

    with pytest.raises(audformat.errors.BadKeyError):
        db.raters["bad"]

    with pytest.raises(audformat.errors.BadKeyError):
        db.schemes["bad"]

    with pytest.raises(audformat.errors.BadKeyError):
        db.splits["bad"]

    with pytest.raises(audformat.errors.BadKeyError):
        db.tables["bad"]

    with pytest.raises(audformat.errors.TableExistsError):
        # a miscellaneous table with same ID exists already
        db["misc"] = audformat.Table()

    with pytest.raises(audformat.errors.TableExistsError):
        # a filewise table with same ID exists already
        db["files"] = audformat.MiscTable(pd.Index([], name="idx"))

    with pytest.raises(audformat.errors.TableExistsError):
        # a segmented table with same ID exists already
        db["segments"] = audformat.MiscTable(pd.Index([], name="idx"))
