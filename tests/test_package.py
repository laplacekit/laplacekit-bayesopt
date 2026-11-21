from importlib import metadata

import laplacekit as lk
import laplacekit_bayesopt


def test_entry_point_loads_the_plugin_record():
    (ep,) = [
        ep
        for ep in metadata.entry_points(group="laplacekit.plugins")
        if ep.name == "laplacekit_bayesopt"
    ]

    assert ep.load() is laplacekit_bayesopt.plugin
    assert ep.load().supports == ">=0.1a0,<0.2"


def test_exports():
    assert laplacekit_bayesopt.__all__ == [
        "LaplaceBNNSurrogate",
        "OnlineMarglik",
        "PosthocMarglik",
        "Trainer",
    ]


def test_plugins_lists_it():
    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-bayesopt"]

    assert info.error is None and info.supported


def test_health_reports_botorch_and_gpytorch():
    checks = {c.name: c for c in laplacekit_bayesopt.health()}

    assert checks["botorch"].status == "ok"
    assert checks["botorch"].detail == metadata.version("botorch")
    assert checks["gpytorch"].status == "ok"
