import pytest
import gpflow
import tensorflow as tf


@pytest.fixture(scope="session", autouse=True)
def configure_and_report(request):
	dtype_str = request.config.getoption("--dtype")
	print(f"\nGPFlow version : {gpflow.__version__}")
	print(f"TF version     : {tf.__version__}")
	print(f"dtype          : {dtype_str}")
