
def test_import():
    import heyvi
    assert heyvi.version.is_at_least('0.0.0')
    print('[test_import]: PASSED')


