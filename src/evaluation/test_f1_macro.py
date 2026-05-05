"""
test_f1_macro.py — Tests unitaires pour la correction du bug F1-macro.

Bug résolu : sklearn.f1_score(..., average='macro') sans labels=[0,1]
calculait la moyenne uniquement sur les classes présentes dans y_true.
Quand seule la classe 1 (dark) était présente, f1_macro = 1.0 au lieu de 0.5.

Correction : ajout de labels=[0, 1] dans compute_f1_triage().

Usage :
    /opt/anaconda3/bin/python src/evaluation/test_f1_macro.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.evaluation.metrics import compute_f1_triage


def _make_df(filenames, flags, preds):
    """Helper : crée les DataFrames index + prédictions."""
    df_index = pd.DataFrame({"filename": filenames, "quality_flag": flags})
    df_preds = pd.DataFrame({"filename": filenames, "weather_label": preds})
    return df_index, df_preds


def test_f1_macro_seule_classe_positive():
    """
    Cas critique du bug : y_true ne contient que la classe 1 (dark).
    Sans labels=[0,1], sklearn donnait f1_macro = 1.0 (incorrect).
    Avec labels=[0,1], f1_macro = (1.0 + 0.0) / 2 = 0.5 (correct).
    """
    df_idx, df_pred = _make_df(
        ["a.jpg", "b.jpg", "c.jpg"],
        ["dark",  "dark",  "dark"],   # seule la classe 1 dans y_true
        [1, 1, 1],                     # toutes prédites correctement
    )
    r = compute_f1_triage(df_idx, df_pred)

    assert abs(r["f1_macro"] - 0.5) < 1e-6, (
        f"BUG : f1_macro={r['f1_macro']:.4f} mais attendu=0.5. "
        "Vérifier que labels=[0,1] est bien passé à f1_score()."
    )
    assert r["f1_volcanic"] == 1.0, f"f1_volcanic attendu 1.0, obtenu {r['f1_volcanic']}"
    assert r["f1_weather"]  == 0.0, f"f1_weather attendu 0.0, obtenu {r['f1_weather']}"
    print("✓ test_f1_macro_seule_classe_positive : 0.5 ✓")


def test_f1_macro_deux_classes_parfait():
    """Cas sain : deux classes présentes, toutes les prédictions correctes."""
    df_idx, df_pred = _make_df(
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        ["dark",  "dark",  "cloudy", "cloudy"],
        [1, 1, 0, 0],  # parfait
    )
    r = compute_f1_triage(df_idx, df_pred)

    assert abs(r["f1_macro"] - 1.0) < 1e-6, f"f1_macro attendu 1.0, obtenu {r['f1_macro']}"
    assert r["f1_volcanic"] == 1.0
    assert r["f1_weather"]  == 1.0
    print("✓ test_f1_macro_deux_classes_parfait : 1.0 ✓")


def test_f1_macro_cohérence_interne():
    """
    Propriété fondamentale : f1_macro doit toujours être égal à
    (f1_volcanic + f1_weather) / 2, quelle que soit la distribution.
    """
    cases = [
        # (flags, preds)
        (["dark", "dark", "cloudy"],   [1, 0, 1]),
        (["dark", "cloudy", "cloudy"], [1, 0, 0]),
        (["dark", "dark", "dark"],     [1, 0, 1]),
        (["cloudy", "cloudy"],         [0, 1]),
    ]
    for i, (flags, preds) in enumerate(cases):
        n = len(flags)
        filenames = [f"img{j}.jpg" for j in range(n)]
        df_idx, df_pred = _make_df(filenames, flags, preds)
        r = compute_f1_triage(df_idx, df_pred)
        expected_macro = (r["f1_volcanic"] + r["f1_weather"]) / 2
        assert abs(r["f1_macro"] - expected_macro) < 1e-6, (
            f"Cas {i}: f1_macro={r['f1_macro']:.4f} ≠ (f1_v+f1_w)/2={expected_macro:.4f}"
        )
    print(f"✓ test_f1_macro_cohérence_interne : {len(cases)} cas validés ✓")


def test_f1_macro_seule_classe_negative():
    """
    Symétrique : y_true ne contient que la classe 0 (cloudy).
    f1_macro attendu = (0.0 + 1.0) / 2 = 0.5.
    """
    df_idx, df_pred = _make_df(
        ["a.jpg", "b.jpg"],
        ["cloudy", "cloudy"],
        [0, 0],
    )
    r = compute_f1_triage(df_idx, df_pred)

    assert abs(r["f1_macro"] - 0.5) < 1e-6, (
        f"f1_macro attendu 0.5, obtenu {r['f1_macro']}"
    )
    assert r["f1_volcanic"] == 0.0
    assert r["f1_weather"]  == 1.0
    print("✓ test_f1_macro_seule_classe_negative : 0.5 ✓")


def test_f1_macro_aucune_image():
    """Cas dégénéré : aucune image labellisée → dict vide retourné sans crash."""
    df_idx, df_pred = _make_df(
        ["a.jpg"],
        ["usable"],   # pas cloudy ni dark → filtré
        [0],
    )
    r = compute_f1_triage(df_idx, df_pred)
    assert r == {} or r.get("n_samples", 0) == 0, "Attendu dict vide ou n_samples=0"
    print("✓ test_f1_macro_aucune_image : dict vide ✓")


if __name__ == "__main__":
    print("=" * 55)
    print("TESTS UNITAIRES — F1-macro correction")
    print("=" * 55)
    tests = [
        test_f1_macro_seule_classe_positive,
        test_f1_macro_deux_classes_parfait,
        test_f1_macro_cohérence_interne,
        test_f1_macro_seule_classe_negative,
        test_f1_macro_aucune_image,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} ÉCHEC : {e}")
            failed += 1
    print("=" * 55)
    print(f"Résultat : {passed}/{len(tests)} tests passés", end="")
    if failed:
        print(f" — {failed} ÉCHEC(S)")
        sys.exit(1)
    else:
        print(" ✓")
