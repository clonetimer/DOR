import fec


def test_package_version_matches_v24_stable_release():
    assert fec.__version__.startswith("2.4.2")
