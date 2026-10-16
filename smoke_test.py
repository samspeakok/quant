import importlib
import platform
import sys

import numpy
import pandas

import qlib

print(f"python : {sys.version.split()[0]}  ({platform.architecture()[0]})")
print(f"qlib   : {qlib.__version__}")
print(f"location: {qlib.__file__}")
print(f"numpy  : {numpy.__version__}")
print(f"pandas : {pandas.__version__}")
print()

# The Cython extensions are the part most likely to break on a native Windows
# install, so load them first and loudly.
for mod in ["qlib.data._libs.rolling", "qlib.data._libs.expanding"]:
    importlib.import_module(mod)
    print(f"OK  native   {mod}")

deps = ["pyarrow", "lightgbm", "sklearn", "cvxpy", "matplotlib", "pymongo", "mlflow"]
for name in deps:
    m = importlib.import_module(name)
    print(f"OK  dep      {name} {getattr(m, '__version__', '')}")

# Model / strategy / data-layer entry points used by the official examples.
from qlib.contrib.data.handler import Alpha158, Alpha360  # noqa: E402
from qlib.contrib.model.gbdt import LGBModel  # noqa: E402
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy  # noqa: E402
from qlib.backtest import backtest  # noqa: E402
from qlib.workflow import R  # noqa: E402

for cls in [Alpha158, Alpha360, LGBModel, TopkDropoutStrategy]:
    print(f"OK  class    {cls.__module__}.{cls.__name__}")
print("OK  func     backtest / workflow.R")

# End-to-end config-driven construction, which is how every qlib example runs.
from qlib.utils import init_instance_by_config  # noqa: E402

model = init_instance_by_config(
    {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421, "num_leaves": 63},
    }
)
print(f"OK  config   init_instance_by_config -> {type(model).__name__}")

print()
print("Qlib import smoke test passed.")
print("Next: fetch market data, then qlib.init(provider_uri=...) can run.")
