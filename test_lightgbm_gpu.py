import numpy as np
import lightgbm as lgb


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20))
    y = rng.integers(0, 2, size=500)

    clf = lgb.LGBMClassifier(
        device="cuda",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    clf.fit(X, y)

    print("LightGBM version:", lgb.__version__)
    print("Params device:", clf.booster_.params.get("device"))
    preds = clf.predict_proba(X[:5])[:, 1]
    print("Sample predictions:", np.round(preds, 4))


if __name__ == "__main__":
    main()
