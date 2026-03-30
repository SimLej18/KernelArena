import pytest
import jax
import gpjax


@pytest.fixture(scope="session", autouse=True)
def configure_and_report(request):
	dtype_str = request.config.getoption("--dtype")
	no_jit = request.config.getoption("--no-jit")
	if dtype_str == "float64":
		jax.config.update("jax_enable_x64", True)
	if no_jit:
		jax.config.update("jax_disable_jit", True)
	print(f"\nJAX version   : {jax.__version__}")
	print(f"Backend       : {jax.default_backend()}")
	print(f"dtype         : {dtype_str}")
	print(f"JIT enabled   : {not jax.config.jax_disable_jit}")
	print(f"GPJax version : {gpjax.__version__}")
