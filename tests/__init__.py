"""Test package.

Present so ``from tests.conftest import make_workflow`` resolves. Without it the
imports work only when pytest happens to put the rootdir on ``sys.path``, which
depends on invocation and is exactly the kind of thing that passes locally and
fails in CI.
"""
