from drAIn import version

def test_module_version():
    assert version.get_version() != None
