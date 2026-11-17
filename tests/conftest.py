# pytest loads this file before any test module. Crazyflow must be imported before scipy
# (it sets SCIPY_ARRAY_API=1), so importing it here makes the test import order irrelevant.
import crazyflow  # noqa: F401
