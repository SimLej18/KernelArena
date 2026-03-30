import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def configure_and_report(request):
	dtype_str = request.config.getoption("--dtype")
	if dtype_str == "float64":
		torch.set_default_dtype(torch.float64)
	print(f"\nTorch version : {torch.__version__}")
	print(f"Torch backend : {torch.get_default_device()}")
	print(f"dtype         : {dtype_str}")
