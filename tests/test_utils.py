from summer3 import epi
from summer3 import utils


def test_stub():
    assert True


def test_get_unique_keyname():
    c = {}
    for name in ["shoes", "bats", "shoes"]:
        cur_name = utils.get_unique_keyname(name, c)
        c[cur_name] = name

    assert c == {"shoes_0": "shoes", "bats_0": "bats", "shoes_1": "shoes"}
