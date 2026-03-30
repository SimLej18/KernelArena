import pytest
import sklearn


@pytest.fixture(scope="session", autouse=True)
def configure_and_report(request):
	dtype_str = request.config.getoption("--dtype")
	print(f"\nsklearn version : {sklearn.__version__}")
	print(f"dtype           : {dtype_str}")
