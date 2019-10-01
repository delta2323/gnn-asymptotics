import pytest

from lib.dataset import citation


@pytest.fixture(params=['citeseer', 'cora'])
def dataset_name(request):
    return request.param


@pytest.fixture(params=[True, False])
def gcn_normalize(request):
    return request.param


@pytest.fixture(params=[True, False])
def noisy(request):
    return request.param


def test_load(dataset_name, gcn_normalize, noisy):
    citation.load(dataset_name, gcn_normalize, noisy)
