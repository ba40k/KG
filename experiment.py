import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from DecisionTree import DecisionTree
import Estimators


def generate_data(n_samples=4000, random_state=0):
    """Бинарная классификация с 10 бинарными категориальными признаками."""
    rng = np.random.default_rng(random_state)

    X = pd.DataFrame(
        rng.integers(0, 2, size=(n_samples, 10)),
        columns=[f"cat_{i}" for i in range(10)],
    )

    z = 2 * X.to_numpy() - 1

    score = (
        1.1 * z[:, 0]
        + 0.9 * z[:, 1]
        - 0.8 * z[:, 2]
        + 0.6 * z[:, 3]
        + 1.2 * z[:, 4] * z[:, 5]
        + 0.8 * z[:, 6] * z[:, 7]
    )

    probability = 1 / (1 + np.exp(-score))

    y = pd.Series(
        rng.binomial(1, probability),
        index=X.index,
        name="target",
    )

    return X, y


def run_experiment(
    n_samples=4000,
    depths=(0, 2, 3, 5),
    seeds=range(5),
):
    estimator_classes = [
        Estimators.MajorityEstimator,
        Estimators.KnnEstimator,
        Estimators.LogRegEstimator,
        Estimators.LDAEstimator,
    ]

    rows = []

    for seed in seeds:
        X, y = generate_data(
            n_samples=n_samples,
            random_state=seed,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=y,
        )

        for depth in depths:
            for estimator_class in estimator_classes:
                tree = DecisionTree(
                    max_depth=depth,
                    estimator_class=estimator_class,
                )

                try:
                    tree.fit(X_train, y_train)

                    train_pred = tree.predict(
                        X_train,
                        pd.Series(0, index=X_train.index, dtype=int),
                        None,
                    )

                    test_pred = tree.predict(
                        X_test,
                        pd.Series(0, index=X_test.index, dtype=int),
                        None,
                    )

                    rows.append({
                        "seed": seed,
                        "depth": depth,
                        "estimator": estimator_class.__name__,
                        "train_accuracy": accuracy_score(
                            y_train, train_pred
                        ),
                        "test_accuracy": accuracy_score(
                            y_test, test_pred
                        ),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_test, test_pred
                        ),
                        "error": None,
                    })

                except Exception as exc:
                    # Ошибка одного варианта не прерывает весь эксперимент.
                    rows.append({
                        "seed": seed,
                        "depth": depth,
                        "estimator": estimator_class.__name__,
                        "train_accuracy": np.nan,
                        "test_accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    results = pd.DataFrame(rows)

    summary = (
        results
        .groupby(["depth", "estimator"], sort=True)
        .agg(
            successful_runs=("test_accuracy", "count"),
            total_runs=("seed", "size"),
            train_accuracy=("train_accuracy", "mean"),
            test_accuracy=("test_accuracy", "mean"),
            test_std=("test_accuracy", "std"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )

    return results, summary

def visualize_experiment(
    n_samples=4000,
    max_depth=3,
    random_state=42,
    n_show=80,
):
    X, y = generate_data(
        n_samples=n_samples,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    estimator_classes = [
        Estimators.MajorityEstimator,
        Estimators.KnnEstimator,
        Estimators.LogRegEstimator,
        Estimators.LDAEstimator,
    ]

    predictions = {}
    scores = []

    for estimator_class in estimator_classes:
        name = estimator_class.__name__

        tree = DecisionTree(
            max_depth=max_depth,
            estimator_class=estimator_class,
        )

        tree.fit(X_train, y_train)

        train_pred = tree.predict(
            X_train,
            pd.Series(0, index=X_train.index, dtype=int),
            None,
        )

        test_pred = tree.predict(
            X_test,
            pd.Series(0, index=X_test.index, dtype=int),
            None,
        )

        predictions[name] = test_pred.to_numpy()

        scores.append({
            "estimator": name,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
        })

    scores = pd.DataFrame(scores)

    # ---------------------------------------------------------
    # График 1. Сгенерированные данные.
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 5),
        constrained_layout=True,
    )

    class_counts = (
        y.value_counts()
        .reindex([0, 1], fill_value=0)
    )

    bars = axes[0].bar(
        ["Class 0", "Class 1"],
        class_counts.to_numpy(),
        color=["tab:blue", "tab:orange"],
    )

    axes[0].bar_label(bars)
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Number of objects")

    positions = np.arange(X.shape[1])
    width = 0.38

    for category, offset, color in [
        (0, -width / 2, "tab:blue"),
        (1, width / 2, "tab:orange"),
    ]:
        positive_rates = [
            y.loc[X[feature].eq(category)].mean()
            for feature in X.columns
        ]

        axes[1].bar(
            positions + offset,
            positive_rates,
            width=width,
            label=f"Feature category = {category}",
            color=color,
        )

    axes[1].axhline(
        y.mean(),
        color="black",
        linestyle="--",
        label="Overall class 1 proportion",
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(X.columns, rotation=45)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Observed class 1 proportion")
    axes[1].set_title("Target dependence on each feature separately")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(
        "generated_data.png",
        dpi=160,
        bbox_inches="tight",
    )

    # ---------------------------------------------------------
    # График 2. Истинные ответы и ответы моделей.
    # ---------------------------------------------------------
    n_show = min(n_show, len(X_test))

    # Выбираем одни и те же случайные тестовые объекты для всех моделей.
    rng = np.random.default_rng(random_state)
    selected = rng.choice(
        len(X_test),
        size=n_show,
        replace=False,
    )

    true_values = y_test.to_numpy()[selected]

    # Сначала показываем класс 0, затем класс 1.
    # Это только порядок отображения, не изменение предсказаний.
    order = np.argsort(true_values, kind="stable")
    selected = selected[order]
    true_values = true_values[order]

    names = list(predictions)

    answer_matrix = np.vstack(
        [true_values]
        + [predictions[name][selected] for name in names]
    )

    fig, ax = plt.subplots(
        figsize=(16, 5),
        constrained_layout=True,
    )

    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["tab:blue", "tab:orange"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    image = ax.imshow(
        answer_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(names) + 1))
    ax.set_yticklabels(["True class"] + names)
    ax.set_xlabel("Selected test objects, ordered by true class")
    ax.set_title(
        f"Model predictions: depth={max_depth}; "
        "red crosses mark errors"
    )

    for row, name in enumerate(names, start=1):
        wrong_columns = np.flatnonzero(
            predictions[name][selected] != true_values
        )

        ax.scatter(
            wrong_columns,
            np.full(len(wrong_columns), row),
            marker="x",
            color="red",
            s=30,
            linewidths=1.5,
        )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1],
        pad=0.02,
    )
    colorbar.set_label("Predicted / true class")

    fig.savefig(
        "model_predictions.png",
        dpi=160,
        bbox_inches="tight",
    )

    # ---------------------------------------------------------
    # График 3. Матрицы ошибок на всей тестовой выборке.
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 9),
        constrained_layout=True,
    )

    for ax, name in zip(axes.flat, names):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions[name],
            labels=[0, 1],
            display_labels=["0", "1"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )

        score = accuracy_score(y_test, predictions[name])

        ax.set_title(
            f"{name}\nTest accuracy = {score:.3f}"
        )

    fig.suptitle(
        f"Confusion matrices: depth={max_depth}",
        fontsize=14,
    )

    fig.savefig(
        "confusion_matrices.png",
        dpi=160,
        bbox_inches="tight",
    )

    # ---------------------------------------------------------
    # График 4. Accuracy на обучении и тесте.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(11, 5),
        constrained_layout=True,
    )

    positions = np.arange(len(scores))
    width = 0.35

    train_bars = ax.bar(
        positions - width / 2,
        scores["train_accuracy"],
        width=width,
        label="Train",
    )

    test_bars = ax.bar(
        positions + width / 2,
        scores["test_accuracy"],
        width=width,
        label="Test",
    )

    ax.bar_label(train_bars, fmt="%.3f", padding=3)
    ax.bar_label(test_bars, fmt="%.3f", padding=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(scores["estimator"], rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Train and test accuracy: depth={max_depth}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(
        "model_accuracy.png",
        dpi=160,
        bbox_inches="tight",
    )

    print("\nРезультаты визуализированного запуска:\n")
    print(
        scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    plt.show()

    return scores

if __name__ == "__main__":
    results, summary = run_experiment()

    print("\nСредние результаты по запускам:\n")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nСравнение средней тестовой accuracy:\n")
    comparison = summary.pivot(
        index="depth",
        columns="estimator",
        values="test_accuracy",
    )
    print(
        comparison.to_string(
            float_format=lambda value: f"{value:.4f}",
        )
    )

    errors = results.loc[
        results["error"].notna(),
        ["seed", "depth", "estimator", "error"],
    ]

    if not errors.empty:
        print("\nОшибки отдельных запусков:\n")
        print(errors.to_string(index=False))
    visualize_experiment(
        n_samples=4000,
        max_depth=3,
        random_state=42,
        n_show=80,
    )
    results.to_csv("experiment_results.csv", index=False)
    summary.to_csv("experiment_summary.csv", index=False)