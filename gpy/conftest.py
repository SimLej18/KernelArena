import pytest
import GPy


@pytest.fixture(scope="session", autouse=True)
def configure_and_report(request):
	dtype_str = request.config.getoption("--dtype")
	print(f"\nGPy version : {GPy.__version__}")
	print(f"dtype       : float64  (GPy relies on numpy/scipy — --dtype={dtype_str} ignored)")
