"""Small CPU tensors run faster without a large OpenMP thread pool."""
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
