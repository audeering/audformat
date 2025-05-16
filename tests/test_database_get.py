import numpy as np
import pandas as pd
import pytest

import audeer

import audformat
import audformat.testing


@pytest.fixture(scope="function")
def mono_db(tmpdir):
    r"""Database with ..."""
    name = "mono-db"
    path = audeer.mkdir(tmpdir, name)
    db = audformat.Database(name)

    # --- Splits
    db.splits["train"] = audformat.Split("train")
    db.splits["test"] = audformat.Split("test")

    # --- Schemes
    db.schemes["age"] = audformat.Scheme("int", minimum=0)
    db.schemes["annotated"] = audformat.Scheme("bool")
    db.schemes["height"] = audformat.Scheme("float")
    db.schemes["int"] = audformat.Scheme("int")
    db.schemes["partial"] = audformat.Scheme("str")
    db.schemes["rating"] = audformat.Scheme("int", labels=[0, 1, 2])
    db.schemes["regression"] = audformat.Scheme("float")
    db.schemes["selection"] = audformat.Scheme("int", labels=[0, 1])
    db.schemes["speaker.weight"] = audformat.Scheme(
        "str",
        labels=["low", "normal", "high"],
    )
    db.schemes["status"] = audformat.Scheme("bool")
    db.schemes["status-reversed"] = audformat.Scheme("bool")
    db.schemes["update"] = audformat.Scheme("bool")
    db.schemes["winner"] = audformat.Scheme(
        "str",
        labels={
            "w1": {"year": 1995},
            "w2": {"year": 1996},
            "w3": {"year": 1997},
        },
    )
    db.schemes["weather"] = audformat.Scheme(
        "str",
        labels=["cloudy", "rainy", "sunny"],
    )
    db.schemes["text"] = audformat.Scheme(
        "str",
        labels={"a": "A text", "b": "B text"},
    )
    db.schemes["numbers"] = audformat.Scheme(
        "str",
        labels={"0": 0, "1": 1, "2": 2},
    )
    db.schemes["month"] = audformat.Scheme(
        "int",
        labels={1: "jan", 2: "feb", 3: "mar"},
    )

    # --- Misc tables
    index = pd.Index(["s1", "s2", "s3"], name="speaker", dtype="string")
    db["speaker"] = audformat.MiscTable(index)
    db["speaker"]["age"] = audformat.Column(scheme_id="age")
    db["speaker"]["age"].set([23, np.nan, 59])
    db["speaker"]["gender"] = audformat.Column()
    db["speaker"]["gender"].set(["female", "", "male"])
    db["speaker"]["height-with-10y"] = audformat.Column(scheme_id="height")
    db["speaker"]["height-with-10y"].set([1.12, 1.45, 1.01])
    db["speaker"]["current-height"] = audformat.Column(scheme_id="height")
    db["speaker"]["current-height"].set([1.76, 1.95, 1.80])
    db["speaker"]["weight"] = audformat.Column(scheme_id="speaker.weight")
    db["speaker"]["weight"].set(["normal", "high", "low"])
    db["speaker"]["selection"] = audformat.Column()
    db["speaker"]["selection"].set([1.0, 1.0, 1.0])

    index = pd.Index(["today", "yesterday"], name="day", dtype="string")
    db["weather"] = audformat.MiscTable(index)
    db["weather"]["weather"] = audformat.Column(scheme_id="weather")
    db["weather"]["weather"].set(["cloudy", "sunny"])

    # --- Schemes with misc tables
    db.schemes["speaker"] = audformat.Scheme("str", labels="speaker")

    # --- Filewise tables
    index = audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"])
    db["files"] = audformat.Table(index)
    db["files"]["channel0"] = audformat.Column(scheme_id="speaker")
    db["files"]["channel0"].set(["s1", "s2", "s3"])
    db["files"]["winner"] = audformat.Column(scheme_id="winner")
    db["files"]["winner"].set(["w1", "w1", "w2"])
    db["files"]["perceived-age"] = audformat.Column(scheme_id="age")
    db["files"]["perceived-age"].set([25, 34, 45])
    db["files"]["text"] = audformat.Column(scheme_id="text")
    db["files"]["text"].set(["a", "a", "b"])
    db["files"]["numbers"] = audformat.Column(scheme_id="numbers")
    db["files"]["numbers"].set(["0", "1", "2"])
    db["files"]["month"] = audformat.Column(scheme_id="month")
    db["files"]["month"].set([1, 2, 3])

    index = audformat.filewise_index(["f1.wav"])
    db["files.sub"] = audformat.Table(index)
    db["files.sub"]["speaker"] = audformat.Column(scheme_id="speaker")
    db["files.sub"]["speaker"].set("s1")
    db["files.sub"]["text"] = audformat.Column()
    db["files.sub"]["text"].set("a")
    db["files.sub"]["numbers"] = audformat.Column(scheme_id="int")
    db["files.sub"]["numbers"].set(0)
    db["files.sub"]["partial"] = audformat.Column(scheme_id="partial")
    db["files.sub"]["partial"].set("a")

    index = audformat.filewise_index(["f2.wav", "f1.wav"])
    db["files.reversed"] = audformat.Table(index)
    db["files.reversed"]["status-reversed"] = audformat.Column(
        scheme_id="status-reversed"
    )
    db["files.reversed"]["status-reversed"].set([False, True])

    index = audformat.filewise_index(["f1.wav", "f3.wav"])
    db["other"] = audformat.Table(index)
    db["other"]["sex"] = audformat.Column()
    db["other"]["sex"].set(["female", "male"])
    db["other"]["weight"] = audformat.Column()
    db["other"]["weight"].set([87, 86])
    db["other"]["selection"] = audformat.Column(scheme_id="selection")
    db["other"]["selection"].set([1, 1])

    index = audformat.filewise_index(["f1.wav", "f2.wav"])
    db["rating.train"] = audformat.Table(index, split_id="train")
    db["rating.train"]["rating"] = audformat.Column(scheme_id="rating")
    db["rating.train"]["rating"].set([0, 1])

    index = audformat.filewise_index(["f3.wav"])
    db["rating.test"] = audformat.Table(index, split_id="test")
    db["rating.test"]["rating"] = audformat.Column(scheme_id="rating")
    db["rating.test"]["rating"].set([1])

    index = audformat.filewise_index(["f2.wav"])
    db["status"] = audformat.Table(index)
    db["status"]["status"] = audformat.Column(scheme_id="status")
    db["status"]["status"].set(True)
    db["status"]["update"] = audformat.Column(scheme_id="update")
    db["status"]["update"].set(False)

    index = audformat.filewise_index(["f3.wav"])
    db["last"] = audformat.Table(index)
    db["last"]["annotated"] = audformat.Column(scheme_id="annotated")
    db["last"]["annotated"].set(True)

    # --- Segmented tables
    index = audformat.segmented_index(
        ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
        [0, 0.1, 0.3, 0],
        [0.2, 0.2, 0.5, 0.7],
    )
    db["segments"] = audformat.Table(index)
    db["segments"]["rating"] = audformat.Column(scheme_id="rating")
    db["segments"]["rating"].set([1, 1, 2, 2])
    db["segments"]["winner"] = audformat.Column(scheme_id="winner")
    db["segments"]["winner"].set(["w1", "w1", "w1", "w1"])
    db["segments"]["regression"] = audformat.Column(scheme_id="regression")
    db["segments"]["regression"].set([0.3, 0.2, 0.6, 0.4])

    index = audformat.segmented_index(
        ["f1.wav", "f1.wav"],
        [0, 0.3],
        [0.2, 0.4],
    )
    db["segments.other"] = audformat.Table(index)
    db["segments.other"]["update"] = audformat.Column(scheme_id="update")
    db["segments.other"]["update"].set([True, True])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=1, file_duration="1s")

    return db


@pytest.fixture(scope="function")
def stereo_db(tmpdir):
    r"""Database with stereo files and same scheme for both channels.

    It contains two tables,
    in one the same labels are used for both channels,
    and in the other different labels are used.

    """
    name = "stereo-db"
    path = audeer.mkdir(tmpdir, name)
    db = audformat.Database(name)

    # --- Schemes
    db.schemes["age"] = audformat.Scheme("int", minimum=0)

    # --- Misc tables
    index = pd.Index(["s1", "s2", "s3"], name="speaker", dtype="string")
    db["speaker"] = audformat.MiscTable(index)
    db["speaker"]["age"] = audformat.Column(scheme_id="age")
    db["speaker"]["age"].set([23, np.nan, 59])
    db["speaker"]["gender"] = audformat.Column()
    db["speaker"]["gender"].set(["female", "", "male"])

    # --- Schemes with misc tables
    db.schemes["speaker"] = audformat.Scheme("str", labels="speaker")

    # --- Filewise tables
    index = audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"])
    db["run1"] = audformat.Table(index)
    db["run1"]["channel0"] = audformat.Column(scheme_id="speaker")
    db["run1"]["channel1"] = audformat.Column(scheme_id="speaker")
    db["run1"]["channel0"].set(["s1", "s2", "s3"])
    db["run1"]["channel1"].set(["s3", "s1", "s2"])

    db["run2"] = audformat.Table(index)
    db["run2"]["channel0"] = audformat.Column(scheme_id="speaker")
    db["run2"]["channel1"] = audformat.Column(scheme_id="speaker")
    db["run2"]["channel0"].set(["s1", "s2", "s3"])
    db["run2"]["channel1"].set(["s1", "s2", "s3"])

    db["run3"] = audformat.Table(index)
    db["run3"]["channel0"] = audformat.Column(scheme_id="speaker")
    db["run3"]["channel1"] = audformat.Column(scheme_id="speaker")
    db["run3"]["channel0"].set(["s2", "s1", "s3"])
    db["run3"]["channel1"].set(["s2", "s2", "s3"])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=2, file_duration="1s")

    return db


@pytest.fixture(scope="function")
def overlapping_data_db(tmpdir):
    r"""Database with scheme and scheme labels.

    This databases resulted in an error during development
    as one of the categories is `NaN`,
    see https://github.com/audeering/audformat/pull/399#issuecomment-1794435304

    """
    name = "overlapping_data_db"
    path = audeer.mkdir(tmpdir, name)
    db = audformat.Database(name)

    # --- Schemes
    db.schemes["speaker"] = audformat.Scheme(
        "int",
        labels={0: {"gender": "female"}},
    )
    db.schemes["gender"] = audformat.Scheme("str", labels=["female", "male"])

    # --- Tables
    index = audformat.filewise_index(["f1.wav", "f2.wav"])
    db["files"] = audformat.Table(index)
    db["files"]["speaker"] = audformat.Column(scheme_id="speaker")
    db["files"]["speaker"].set([0, np.nan])
    db["files"]["gender"] = audformat.Column(scheme_id="gender")
    db["files"]["gender"].set(["female", np.nan])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=1, file_duration="1s")

    return db


@pytest.fixture(scope="function")
def scheme_not_assigned_db(tmpdir):
    r"""Database with matching scheme that is not assigned."""
    name = "scheme_not_assigned_db"
    path = audeer.mkdir(tmpdir, name)
    db = audformat.Database(name)

    # --- Schemes
    db.schemes["speaker"] = audformat.Scheme(
        "int",
        labels={0: {"gender": "female"}},
    )
    db.schemes["rating"] = audformat.Scheme("int")

    # --- Tables
    index = audformat.filewise_index(["f1.wav"])
    db["files"] = audformat.Table(index)
    db["files"]["speaker"] = audformat.Column()
    db["files"]["speaker"].set([0])
    db["files"]["rater1"] = audformat.Column()
    db["files"]["rater1"].set([1])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=1, file_duration="1s")

    return db


@pytest.fixture(scope="function")
def wrong_scheme_labels_db(tmpdir):
    r"""Database with scheme labels that do not match."""
    name = "wrong_scheme_labels_db"
    path = audeer.mkdir(tmpdir, name)
    db = audformat.Database(name)

    # --- Schemes
    db.schemes["speaker"] = audformat.Scheme(
        "int",
        labels={
            0: {"gender": "female"},
            1: {"gedner": "male"},  # added a typo here
        },
    )

    # --- Tables
    index = audformat.filewise_index(["f1.wav", "f2.wav"])
    db["files"] = audformat.Table(index)
    db["files"]["speaker"] = audformat.Column(scheme_id="speaker")
    db["files"]["speaker"].set([0, 1])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=1, file_duration="1s")

    return db


@pytest.mark.parametrize(
    "db, scheme, additional_schemes, expected",
    [
        (
            "mono_db",
            "non-existing",
            [],
            pd.DataFrame(
                {
                    "non-existing": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "weather",
            [],
            pd.DataFrame(
                {
                    "weather": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            pd.DataFrame(
                {
                    "gender": ["female", "", "male"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
        (
            "mono_db",
            "sex",
            [],
            pd.DataFrame(
                {
                    "sex": ["female", "male"],
                },
                index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "sex",
            ["gender"],
            pd.concat(
                [
                    pd.Series(
                        ["female", "male"],
                        index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                        dtype="object",
                        name="sex",
                    ),
                    pd.Series(
                        ["female", "male"],
                        index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "gender",
            ["sex", "non-existing"],
            pd.concat(
                [
                    pd.Series(
                        ["female", "", "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                    pd.Series(
                        ["female", np.nan, "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="object",
                        name="sex",
                    ),
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype="object",
                        name="non-existing",
                    ),
                ],
                axis=1,
            ),
        ),
        # Ensure that requesting a non-existing scheme
        # before an existing scheme
        # does return values for existing schemes.
        # https://github.com/audeering/audformat/issues/426
        (
            "mono_db",
            "gender",
            ["non-existing", "sex"],
            pd.concat(
                [
                    pd.Series(
                        ["female", "", "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype="object",
                        name="non-existing",
                    ),
                    pd.Series(
                        ["female", np.nan, "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="object",
                        name="sex",
                    ),
                ],
                axis=1,
            ),
        ),
        # Ensure that requesting a non-existing scheme
        # before an existing scheme
        # does return values for existing schemes.
        # https://github.com/audeering/audformat/issues/426
        (
            "mono_db",
            "gender",
            ["numbers", "non-existing", "sex"],
            pd.concat(
                [
                    pd.Series(
                        ["female", "", "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                    pd.Series(
                        [0, 1, 2],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="Int64",
                        name="numbers",
                    ),
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype="object",
                        name="non-existing",
                    ),
                    pd.Series(
                        ["female", np.nan, "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="object",
                        name="sex",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "winner",
            [],
            pd.DataFrame(
                {
                    "winner": ["w1", "w1", "w2", "w1", "w1", "w1", "w1"],
                },
                index=audformat.utils.union(
                    [
                        audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                    ]
                ),
                dtype=pd.CategoricalDtype(
                    ["w1", "w2", "w3"],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "year",
            [],
            pd.DataFrame(
                {
                    "year": [1995, 1995, 1996, 1995, 1995, 1995, 1995],
                },
                index=audformat.utils.union(
                    [
                        audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                    ]
                ),
                dtype="Int64",
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            pd.DataFrame(
                {
                    "rating": [1, 0, 1, 1, 1, 2, 2],
                },
                index=audformat.utils.union(
                    [
                        audformat.filewise_index(["f3.wav", "f1.wav", "f2.wav"]),
                        audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                    ]
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "regression",
            [],
            pd.DataFrame(
                {
                    "regression": [0.3, 0.2, 0.6, 0.4],
                },
                index=audformat.segmented_index(
                    ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                    [0, 0.1, 0.3, 0],
                    [0.2, 0.2, 0.5, 0.7],
                ),
                dtype="float",
            ),
        ),
        # Segmented index with filewise additional entries
        # https://github.com/audeering/audformat/issues/460
        (
            "mono_db",
            "regression",
            ["speaker"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        ["s1", "s1", "s1", "s2"],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype=pd.CategoricalDtype(
                            ["s1", "s2", "s3"],
                            ordered=False,
                        ),
                        name="speaker",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "regression",
            ["partial"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        ["a", "a", "a", None],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="string",
                        name="partial",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "regression",
            ["status"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        [None, None, None, True],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="boolean",
                        name="status",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "regression",
            ["annotated"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        [None, None, None, None],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="boolean",
                        name="annotated",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "regression",
            ["partial", "status"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        ["a", "a", "a", None],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="string",
                        name="partial",
                    ),
                    pd.Series(
                        [None, None, None, True],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="boolean",
                        name="status",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "regression",
            ["partial", "update"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        ["a", "a", "a", None],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="string",
                        name="partial",
                    ),
                    pd.Series(
                        [True, None, None, None],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="boolean",
                        name="update",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "regression",
            ["status-reversed"],
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="float",
                        name="regression",
                    ),
                    pd.Series(
                        [True, True, True, False],
                        index=audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype="boolean",
                        name="status-reversed",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "selection",
            [],
            pd.DataFrame(
                {
                    "selection": [1, 1, 1],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype=pd.CategoricalDtype(
                    [1, 0],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "numbers",
            [],
            pd.DataFrame(
                {
                    "numbers": [0, 1, 2],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="Int64",
            ),
        ),
        (
            "overlapping_data_db",
            "gender",
            [],
            pd.DataFrame(
                {
                    "gender": ["female", np.nan],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav"]),
                dtype=pd.CategoricalDtype(
                    ["female", "male"],
                    ordered=False,
                ),
            ),
        ),
        (
            "scheme_not_assigned_db",
            "gender",
            [],
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "scheme_not_assigned_db",
            "rating",
            [],
            pd.DataFrame(
                {
                    "rating": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "scheme_not_assigned_db",
            "rater1",
            [],
            pd.DataFrame(
                {
                    "rater1": [1],
                },
                index=audformat.filewise_index(["f1.wav"]),
                dtype="object",
            ),
        ),
        (
            "wrong_scheme_labels_db",
            "gender",
            [],
            pd.DataFrame(
                {
                    "gender": ["female", np.nan],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav"]),
                dtype="string",
            ),
        ),
    ],
)
def test_database_get(request, db, scheme, additional_schemes, expected):
    db = request.getfixturevalue(db)
    df = db.get(scheme, additional_schemes)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "db, scheme, additional_schemes, "
    "original_column_names, aggregate_function, aggregate_strategy, expected",
    [
        # Tests based on `mono_db`,
        # with the following tables, columns
        # matching the scheme `age`.
        #
        # `files`
        # | file   | channel0 (age) | perceived-age |
        # | ------ | -------------- | ------------- |
        # | f1.wav |             23 |            25 |
        # | f2.wav |                |            34 |
        # | f3.wav |             59 |            45 |
        #
        # `files.sub`
        # | file   | speaker (age) |
        # | ------ | ------------- |
        # | f1.wav |            23 |
        #
        (
            # Select first value (based on table/column order)
            #
            # files, age: 23, NaN, 59
            "mono_db",
            "age",
            [],
            False,
            lambda y: y[0],
            "mismatch",
            pd.DataFrame(
                {
                    "age": [23, 34, 59],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="Int64",
            ),
        ),
        (
            # Select second value (based on table/column order)
            #
            # files, perceived age: 25, 34, 45
            "mono_db",
            "age",
            [],
            False,
            lambda y: y[1],
            "mismatch",
            pd.DataFrame(
                {
                    "age": [25, 34, 45],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="Int64",
            ),
        ),
        (
            # Return all columns using column names
            #
            # files, channel0:      23, NaN, 59
            # files, perceived-age: 25, 34, 45
            # files.sub, speaker:   23, NaN, NaN
            "mono_db",
            "age",
            [],
            True,
            None,
            "mismatch",
            pd.DataFrame(
                {
                    "channel0": [23, np.nan, 59],
                    "perceived-age": [25, 34, 45],
                    "speaker": [23, np.nan, np.nan],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="Int64",
            ),
        ),
        # Tests based on `mono_db`,
        # with the following tables, columns
        # matching the schemes `gender` and `sex`
        # (considering scheme mappings and column names).
        #
        # `files`
        # | file   | channel0 (gender) |
        # | ------ | ----------------- |
        # | f1.wav | female            |
        # | f2.wav |                   |
        # | f3.wav | male              |
        #
        # `files.sub`
        # | file   | speaker (gender) |
        # | ------ | ---------------- |
        # | f1.wav | female           |
        #
        # `other`
        # | file   | sex    |
        # | ------ | ------ |
        # | f1.wav | female |
        # | f3.wav | male   |
        #
        (
            # Return all columns using column names
            #
            # files, sex:           female, male
            # files, channel0:      23, 59
            # files, perceived-age: 25, 45
            # files.sub, speaker:   23, NaN
            "mono_db",
            "sex",
            ["age"],
            True,
            None,
            "mismatch",
            pd.concat(
                [
                    pd.Series(
                        ["female", "male"],
                        index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                        dtype="object",
                        name="sex",
                    ),
                    pd.Series(
                        [23, 59],
                        index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                        dtype="Int64",
                        name="channel0",
                    ),
                    pd.Series(
                        [25, 45],
                        index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                        dtype="Int64",
                        name="perceived-age",
                    ),
                    pd.Series(
                        [23, np.nan],
                        index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                        dtype="Int64",
                        name="speaker",
                    ),
                ],
                axis=1,
            ),
        ),
        # Tests based on `stereo_db,
        # with the following tables, columns
        # matching the scheme `gender`.
        # All have [f1.wav, f2.wav, f3.wav] as index
        #
        # | table | column   | values                       |
        # | ----- | -------- | ---------------------------- |
        # | run1  | channel0 | ['female', '',       'male'] |
        # | run1  | channel1 | ['male',   'female', ''    ] |
        # | run2  | channel0 | ['female', '',       'male'] |
        # | run2  | channel1 | ['female', '',       'male'] |
        # | run3  | channel0 | ['',       'female', 'male'] |
        # | run3  | channel1 | ['',       '',       'male'] |
        #
        (
            # maxvote
            #
            # gender: ['female', '', 'male']
            #
            "stereo_db",
            "gender",
            [],
            False,
            lambda y: y.mode()[0],
            "overlap",
            pd.DataFrame(
                {
                    "gender": ["female", "", "male"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
    ],
)
def test_database_get_aggregate_and_modify_function(
    request,
    db,
    scheme,
    additional_schemes,
    original_column_names,
    aggregate_function,
    aggregate_strategy,
    expected,
):
    db = request.getfixturevalue(db)
    df = db.get(
        scheme,
        additional_schemes,
        original_column_names=original_column_names,
        aggregate_function=aggregate_function,
        aggregate_strategy=aggregate_strategy,
    )
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "db, scheme, additional_schemes, tables, splits, expected",
    [
        (
            "mono_db",
            "gender",
            [],
            [],
            None,
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            None,
            [],
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            [],
            [],
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            "non-existing",
            None,
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            None,
            "non-existing",
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            ["sex"],
            None,
            "non-existing",
            pd.concat(
                [
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype="object",
                        name="gender",
                    ),
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype="object",
                        name="sex",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            "non-existing",
            "non-existing",
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            [],
            ["other"],
            ["train"],
            pd.DataFrame(
                {
                    "gender": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            ["speaker"],
            ["files.sub"],
            None,
            pd.concat(
                [
                    pd.Series(
                        ["female"],
                        index=audformat.filewise_index(["f1.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                    pd.Series(
                        ["s1"],
                        index=audformat.filewise_index(["f1.wav"]),
                        dtype=pd.CategoricalDtype(
                            ["s1", "s2", "s3"],
                            ordered=False,
                        ),
                        name="speaker",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "sex",
            [],
            ["other"],
            None,
            pd.DataFrame(
                {
                    "sex": ["female", "male"],
                },
                index=audformat.filewise_index(["f1.wav", "f3.wav"]),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            "segments",
            None,
            pd.DataFrame(
                {
                    "rating": [1, 1, 2, 2],
                },
                index=audformat.segmented_index(
                    ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                    [0, 0.1, 0.3, 0],
                    [0.2, 0.2, 0.5, 0.7],
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            None,
            "train",
            pd.DataFrame(
                {
                    "rating": [0, 1],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav"]),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            ["rating.train", "rating.test"],
            "train",
            pd.DataFrame(
                {
                    "rating": [0, 1],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav"]),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            None,
            ["train", "test"],
            pd.DataFrame(
                {
                    "rating": [1, 0, 1],
                },
                index=audformat.filewise_index(["f3.wav", "f1.wav", "f2.wav"]),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            ["rating.train", "rating.test"],
            None,
            pd.DataFrame(
                {
                    "rating": [0, 1, 1],
                },
                index=audformat.filewise_index(
                    [
                        "f1.wav",
                        "f2.wav",
                        "f3.wav",
                    ]
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            ["rating.test", "rating.train"],
            None,
            pd.DataFrame(
                {
                    "rating": [1, 0, 1],
                },
                index=audformat.filewise_index(["f3.wav", "f1.wav", "f2.wav"]),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "rating",
            [],
            ["rating.test", "rating.train"],
            ["train", "test"],
            pd.DataFrame(
                {
                    "rating": [1, 0, 1],
                },
                index=audformat.filewise_index(["f3.wav", "f1.wav", "f2.wav"]),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            # Select run with identical labels
            #
            # run2, channel0: ['female', '', 'male']
            # run2, channel1: ['female', '', 'male']
            #
            # gender: ['female', '', 'male']
            #
            "stereo_db",
            "gender",
            [],
            ["run2"],
            None,
            pd.DataFrame(
                {
                    "gender": ["female", "", "male"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
        (
            # Limit table to files
            # to avoid error due to non-matching dtypes
            "mono_db",
            "text",
            [],
            ["files"],
            None,
            pd.DataFrame(
                {
                    "text": ["A text", "A text", "B text"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
    ],
)
def test_database_get_limit_search(
    request,
    db,
    scheme,
    additional_schemes,
    tables,
    splits,
    expected,
):
    db = request.getfixturevalue(db)
    df = db.get(
        scheme,
        additional_schemes,
        tables=tables,
        splits=splits,
    )
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "db, scheme, additional_schemes, map, expected",
    [
        (
            "mono_db",
            "month",
            [],
            True,
            pd.DataFrame(
                {
                    "month": ["jan", "feb", "mar"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
        (
            "mono_db",
            "month",
            [],
            False,
            pd.DataFrame(
                {
                    "month": [1, 2, 3],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype=pd.CategoricalDtype(
                    categories=[1, 2, 3],
                    ordered=False,
                ),
            ),
        ),
        (
            "mono_db",
            "month",
            ["gender"],
            True,
            pd.DataFrame(
                {
                    "month": ["jan", "feb", "mar"],
                    "gender": ["female", "", "male"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
        (
            "mono_db",
            "month",
            ["gender"],
            False,
            pd.concat(
                [
                    pd.Series(
                        [1, 2, 3],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype=pd.CategoricalDtype(
                            categories=[1, 2, 3],
                            ordered=False,
                        ),
                        name="month",
                    ),
                    pd.Series(
                        ["female", "", "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                ],
                axis=1,
            ),
        ),
        (
            "mono_db",
            "gender",
            ["month"],
            True,
            pd.DataFrame(
                {
                    "gender": ["female", "", "male"],
                    "month": ["jan", "feb", "mar"],
                },
                index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                dtype="string",
            ),
        ),
        (
            "mono_db",
            "gender",
            ["month"],
            False,
            pd.concat(
                [
                    pd.Series(
                        ["female", "", "male"],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype="string",
                        name="gender",
                    ),
                    pd.Series(
                        [1, 2, 3],
                        index=audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        dtype=pd.CategoricalDtype(
                            categories=[1, 2, 3],
                            ordered=False,
                        ),
                        name="month",
                    ),
                ],
                axis=1,
            ),
        ),
    ],
)
def test_database_get_map(
    request,
    db,
    scheme,
    additional_schemes,
    map,
    expected,
):
    db = request.getfixturevalue(db)
    df = db.get(scheme, additional_schemes, map=map)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "db, scheme, additional_schemes, strict, expected",
    [
        (
            "mono_db",
            "sex",
            [],
            False,
            pd.DataFrame(
                {
                    "sex": ["female", "male"],
                },
                index=audformat.filewise_index(
                    ["f1.wav", "f3.wav"],
                ),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "sex",
            [],
            True,
            pd.DataFrame(
                {
                    "sex": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "year",
            [],
            False,
            pd.DataFrame(
                {
                    "year": [1995, 1995, 1996, 1995, 1995, 1995, 1995],
                },
                index=audformat.utils.union(
                    [
                        audformat.filewise_index(["f1.wav", "f2.wav", "f3.wav"]),
                        audformat.segmented_index(
                            ["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                    ]
                ),
                dtype="Int64",
            ),
        ),
        (
            "mono_db",
            "year",
            [],
            True,
            pd.DataFrame(
                {
                    "year": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "sex",
            ["year"],
            True,
            pd.DataFrame(
                {
                    "sex": [],
                    "year": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
        (
            "mono_db",
            "gender",
            ["sex", "year"],
            True,
            pd.DataFrame(
                {
                    "gender": [],
                    "sex": [],
                    "year": [],
                },
                index=audformat.filewise_index(),
                dtype="object",
            ),
        ),
    ],
)
def test_database_get_strict(
    request,
    db,
    scheme,
    additional_schemes,
    strict,
    expected,
):
    db = request.getfixturevalue(db)
    df = db.get(scheme, additional_schemes, strict=strict)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "db, scheme, additional_schemes, tables, original_column_names, "
    "expected_error, expected_error_msg",
    [
        (
            "mono_db",
            "age",
            [],
            None,
            False,
            ValueError,
            (
                "Found overlapping data in column 'age':\n"
                "        left  right\n"
                "file               \n"
                "f1.wav    23     25\n"
                "f3.wav    59     45"
            ),
        ),
        (
            "mono_db",
            "height",
            [],
            None,
            False,
            ValueError,
            (
                "Found overlapping data in column 'height':\n"
                "        left  right\n"
                "file               \n"
                "f1.wav  1.12   1.76\n"
                "f2.wav  1.45   1.95\n"
                "f3.wav  1.01   1.80"
            ),
        ),
        (
            "mono_db",
            "weight",
            [],
            None,
            False,
            TypeError,
            (
                "Cannot join labels for scheme 'weight' "
                "with different data types: int64, object"
            ),
        ),
        (
            # Fail as both schemes result in same original column name,
            # but different dtypes
            #
            # files, channel0:      female, NaN , male
            # files.sub, speaker:   female, NaN, NaN
            # files, channel0:      23, NaN, 59
            # files, perceived-age: 25, 34, 45
            # files.sub, speaker:   23, NaN, NaN
            #
            "mono_db",
            "gender",
            ["age"],
            None,
            True,
            ValueError,
            (
                "Found two columns with name 'channel0' "
                "but different dtypes:\n"
                "string != Int64."
            ),
        ),
        (
            # Fail as we select runs with different labels
            #
            # run1, channel0: ['female', '', 'male']
            # run1, channel1: ['male', 'female', '']
            #
            "stereo_db",
            "gender",
            [],
            ["run1"],
            False,
            ValueError,
            (
                "Found overlapping data in column 'gender':\n"
                "          left   right\n"
                "file                  \n"
                "f1.wav  female    male\n"
                "f2.wav          female\n"
                "f3.wav    male        "
            ),
        ),
        (
            # Fail a dtype does no longer match after mapping of labels
            #
            "mono_db",
            "text",
            [],
            None,
            False,
            # lambda y: y[0],
            ValueError,
            (
                "Found two columns with name 'text' but different dtypes:\n"
                "string != object."
            ),
        ),
    ],
)
def test_database_get_errors(
    request,
    db,
    scheme,
    additional_schemes,
    tables,
    original_column_names,
    expected_error,
    expected_error_msg,
):
    db = request.getfixturevalue(db)
    with pytest.raises(expected_error, match=expected_error_msg):
        db.get(
            scheme,
            additional_schemes,
            tables=tables,
            original_column_names=original_column_names,
        )
