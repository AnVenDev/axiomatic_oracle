from scripts import sample_property


def test_sample_property_format():
    assert isinstance(sample_property.sample_response, dict)
