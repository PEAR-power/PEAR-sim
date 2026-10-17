from pathlib import Path

from pear_sim.contracts import (
    explain_model,
    explanation_text,
    model_hashes,
    physics_model_dict,
    resolved_model_dict,
    sha256_json,
)
from pear_sim.model import load_model


ROOT = Path(__file__).parents[1]


def test_resolved_model_hashes_are_deterministic_and_separate_request() -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")

    first = model_hashes(model)
    second = model_hashes(load_model(ROOT / "examples" / "buck.yaml"))

    assert first == second
    assert first["model_sha256"] == sha256_json(physics_model_dict(model))
    assert resolved_model_dict(model)["hashes"] == first
    assert len(set(first.values())) == 3


def test_m32_preserves_legacy_pwm_physics_and_request_hashes() -> None:
    expected = {
        "boost.yaml": (
            "5b6c644b807478428352d1aa4471a594518a60f1768325126f3e4287618531bd",
            "026b77117a08fdfd8e073ce05b04671532f04aed12bdeef1c9dfc2e5748a8bc6",
        ),
        "buck.yaml": (
            "1423749ba5cce42c67b00a05d50d346fd0c146c5815bf10c6cced19fde30f852",
            "829eb76c92639d2b8ba5d338736c2e9b05700ee276c837c3d3ccfffdf15c4f26",
        ),
        "buck_dcm.yaml": (
            "796f1bd767391e34a69206e793a0c64b585e7dba6010dd48ac4c8454d7495ca6",
            "914f2a9202c04f5eb8a4fc4db0535fe37947a143cd5240623925b36b5588723a",
        ),
        "inverting_buck_boost.yaml": (
            "d458433f48f02ab19aaa0ff132eb183eea1b9d6c20f0d865c9351c419c7dd35f",
            "026b77117a08fdfd8e073ce05b04671532f04aed12bdeef1c9dfc2e5748a8bc6",
        ),
    }

    assert {
        filename: (
            model_hashes(load_model(ROOT / "examples" / filename))["model_sha256"],
            model_hashes(load_model(ROOT / "examples" / filename))["request_sha256"],
        )
        for filename in expected
    } == expected


def test_explanation_identifies_states_outputs_and_limitations() -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    explanation = explain_model(model)

    assert explanation["schema"] == "pear-sim/explanation/v0.1"
    assert explanation["summary"]["state_components"] == ["l1", "c1"]
    assert explanation["summary"]["warm_start"] is True
    assert explanation["request"]["outputs"] == [
        "v(out)",
        "i(l1)",
        "v(q1)",
        "i(q1)",
        "i(d1)",
        "i(vin)",
        "i(load)",
    ]
    assert explanation["conventions"] == {
        "component_voltage": {
            "definition": "v(component)=V(p)-V(n)",
            "positive_terminal": "p",
            "negative_terminal": "n",
        },
        "component_current": {
            "definition": "i(component)>0 means current flows p->n",
            "positive_direction": "p->n",
        },
    }
    initial_conditions = explanation["initial_conditions"]
    assert initial_conditions["strategy"] == "explicit_seeded_values"
    assert initial_conditions["analytic_initialization_claimed"] is False
    assert initial_conditions["summary"] == {
        "state_count": 2,
        "zero_count": 0,
        "nonzero_count": 2,
        "user_declared_count": 2,
        "default_zero_count": 0,
        "unknown_origin_count": 0,
        "warm_start": True,
        "warm_start_components": ["l1", "c1"],
    }
    states = {state["component_id"]: state for state in initial_conditions["states"]}
    assert states["l1"] == {
        "component_id": "l1",
        "component_type": "inductor",
        "state_kind": "current",
        "state_expression": "i(l1)",
        "source_field": "initial_current",
        "value_si": 3.39970055,
        "unit": "A",
        "value_class": "nonzero",
        "seed_strategy": "explicit_numeric_seed",
        "origin": "user_declared",
        "analytic_derivation": "not_claimed",
    }
    assert states["c1"]["state_expression"] == "v(c1)"
    assert states["c1"]["unit"] == "V"
    assert states["c1"]["origin"] == "user_declared"
    assert resolved_model_dict(model)["initial_conditions"] == initial_conditions
    rendered = explanation_text(explanation)
    assert "v(component)=V(p)-V(n)" in rendered
    assert "i(component)>0 means current flows p->n" in rendered
    assert "warm_start=yes" in rendered
    assert "analytic_origin=not_claimed" in rendered
    assert any("sign-off" in item for item in explanation["limitations"])


def test_explicit_zero_and_omitted_initial_condition_have_distinct_provenance(
    tmp_path: Path,
) -> None:
    template = """
schema: pear-sim/v0.1
name: zero_seed
circuit:
  ground: gnd
  components:
    - id: c1
      type: capacitor
      p: out
      n: gnd
      value: 1 uF
{initial_condition}
analysis:
  transient: {{stop: 1 ms, output_step: 10 us}}
outputs: [v(c1)]
""".strip()
    explicit_path = tmp_path / "explicit.yaml"
    omitted_path = tmp_path / "omitted.yaml"
    explicit_path.write_text(
        template.format(initial_condition="      initial_voltage: 0 V") + "\n",
        encoding="utf-8",
    )
    omitted_path.write_text(
        template.format(initial_condition="") + "\n",
        encoding="utf-8",
    )

    explicit = load_model(explicit_path)
    omitted = load_model(omitted_path)
    assert explicit.components[0].initial_state == 0.0
    assert omitted.components[0].initial_state == 0.0
    assert explicit.components[0].initial_state_explicit is True
    assert omitted.components[0].initial_state_explicit is False
    assert model_hashes(explicit)["model_sha256"] == model_hashes(omitted)["model_sha256"]

    explicit_ic = explain_model(explicit)["initial_conditions"]
    omitted_ic = explain_model(omitted)["initial_conditions"]
    assert explicit_ic["states"][0]["origin"] == "user_declared"
    assert omitted_ic["states"][0]["origin"] == "default_zero"
    assert explicit_ic["states"][0]["value_class"] == "zero"
    assert omitted_ic["states"][0]["value_class"] == "zero"
    assert explicit_ic["summary"]["warm_start"] is False
    assert omitted_ic["summary"]["warm_start"] is False
