"""Tests for JSON conversion functions (2_convert_json_to_csv_v7.py)."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.import_helpers import import_functions_from_script

_funcs = import_functions_from_script(
    os.path.join(os.path.dirname(__file__), '..', '2_convert_json_to_csv_v7.py'),
    ['is_nested'],
)
is_nested = _funcs['is_nested']


class TestIsNested:
    """Tests for the is_nested() function."""

    def test_flat_dict(self):
        assert is_nested({'a': 1, 'b': 'hello', 'c': True}) is False

    def test_nested_dict_with_inner_dict(self):
        assert is_nested({'a': 1, 'b': {'nested': True}}) is True

    def test_nested_dict_with_inner_list(self):
        assert is_nested({'a': 1, 'b': [1, 2, 3]}) is True

    def test_flat_list_of_primitives(self):
        assert is_nested([1, 2, 3, 'hello']) is False

    def test_list_of_dicts(self):
        assert is_nested([{'a': 1}, {'b': 2}]) is True

    def test_list_of_lists(self):
        assert is_nested([[1, 2], [3, 4]]) is True

    def test_empty_dict(self):
        assert is_nested({}) is False

    def test_empty_list(self):
        assert is_nested([]) is False

    def test_string_input(self):
        assert is_nested('hello') is False

    def test_number_input(self):
        assert is_nested(42) is False

    def test_dict_with_none_values(self):
        assert is_nested({'a': None, 'b': None}) is False

    def test_dict_with_empty_nested_dict(self):
        assert is_nested({'a': {}}) is True

    def test_dict_with_empty_nested_list(self):
        assert is_nested({'a': []}) is True
