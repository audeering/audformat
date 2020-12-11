import pytest

import audformat
from audformat import testing


def test_errors():

    db = testing.create_db()

    with pytest.raises(audformat.errors.BadIdError):
        db['table'] = audformat.Table(split_id='bad')

    with pytest.raises(audformat.errors.BadIdError):
        db['table'] = audformat.Table(media_id='bad')

    with pytest.raises(audformat.errors.BadIdError):
        db['files']['column'] = audformat.Column(scheme_id='bad')

    with pytest.raises(audformat.errors.BadIdError):
        db['files']['column'] = audformat.Column(rater_id='bad')

    with pytest.raises(FileNotFoundError):
        audformat.Database.load('./bad/path')

    with pytest.raises(audformat.errors.BadTypeError):
        db.media['media'] = audformat.Rater()

    with pytest.raises(audformat.errors.BadTypeError):
        db.splits['split'] = audformat.Media()

    with pytest.raises(audformat.errors.BadTypeError):
        db.schemes['table'] = audformat.Split()

    with pytest.raises(audformat.errors.BadTypeError):
        db.schemes['scheme'] = audformat.Table()

    with pytest.raises(audformat.errors.BadTypeError):
        db.raters['rater'] = audformat.Scheme()

    with pytest.raises(audformat.errors.BadValueError):
        db.raters['rater'] = audformat.Rater('bad')

    with pytest.raises(audformat.errors.BadValueError):
        db.splits['split'] = audformat.Split('bad')

    with pytest.raises(audformat.errors.BadValueError):
        db.schemes['scheme'] = audformat.Scheme('bad')

    with pytest.raises(audformat.errors.BadValueError):
        audformat.Database('foo', 'internal', 'bad')
