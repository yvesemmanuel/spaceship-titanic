import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_style("whitegrid")
custom_colors: list[str] = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#C73E1D",
    "#6A994E",
    "#577590",
]
sns.set_palette(custom_colors)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


def load_data() -> pd.DataFrame | None:
    """Load the Spaceship Titanic dataset with error handling"""
    try:
        df = pd.read_csv("train.csv")
        print("✅ Dataset loaded successfully")
        return df
    except FileNotFoundError:
        print(
            "❌ Error: train.csv not found. Please ensure the file is in the current directory."
        )
        return None


def data_quality_assessment(df) -> None:
    """Professional data-quality plot – Seaborn edition"""
    print("\n" + "=" * 60)
    print("🔍 DATA QUALITY ASSESSMENT")
    print("=" * 60)

    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = (
        pd.DataFrame({"Feature": missing.index, "Missing %": missing_pct.values})
        .sort_values("Missing %", ascending=False)
        .query("`Missing %` > 0")
    )

    if missing_df.empty:
        print("✅ No missing values!")
        return

    print("\n📋 DATA COMPLETENESS:")
    for _, row in missing_df.iterrows():
        print(f"   • {row.Feature}: {row['Missing %']:.1f}%")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=missing_df,
        x="Feature",
        y="Missing %",
        palette="Reds_r",
        edgecolor="black",
    )
    ax.set_title("Missing Values per Feature", fontsize=16, weight="bold")
    ax.set_ylabel("Missing Percentage (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    ax.axhline(5, ls="--", c="orange", alpha=0.8, label="5 % threshold")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    os.makedirs("eda_plots", exist_ok=True)
    plt.savefig("eda_plots/data_quality.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n💡 {len(missing_df)} features need imputation (>{5}% missing).")


def passenger_demographics_analysis(df) -> None:
    """Demographics – Seaborn edition (FacetGrid + barplot + histplot)"""
    print("\n" + "=" * 60)
    print("👥 PASSENGER DEMOGRAPHICS & BEHAVIOR ANALYSIS")
    print("=" * 60)

    df["Deck"] = df["Cabin"].str.split("/", expand=True)[0]
    df["CabinSide"] = df["Cabin"].str.split("/", expand=True)[2]
    df["GroupId"] = df["PassengerId"].str.split("_", expand=True)[0]
    df["GroupSize"] = df["GroupId"].map(df["GroupId"].value_counts())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(
        "Passenger Demographics & Spatial Distribution",
        fontsize=18,
        weight="bold",
    )

    sns.barplot(
        x=df["HomePlanet"].value_counts().index,
        y=df["HomePlanet"].value_counts().values,
        palette=custom_colors[:3],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Origin Planets")
    axes[0, 0].set_ylabel("Passengers")

    sns.barplot(
        x=df["Deck"].value_counts().sort_index().index,
        y=df["Deck"].value_counts().sort_index().values,
        color="#3498DB",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Deck Occupancy")
    axes[0, 1].set_ylabel("Passengers")

    sns.barplot(
        x=df["CabinSide"].value_counts().index,
        y=df["CabinSide"].value_counts().values,
        palette=["#E74C3C", "#27AE60"],
        ax=axes[0, 2],
    )
    axes[0, 2].set_title("Cabin Side (Port vs Starboard)")
    axes[0, 2].set_ylabel("Passengers")

    sns.histplot(
        data=df,
        x="Age",
        hue="HomePlanet",
        bins=20,
        element="step",
        palette=custom_colors[:3],
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Age Distribution by Home Planet")

    sns.barplot(
        x=df["GroupSize"].value_counts().sort_index().index,
        y=df["GroupSize"].value_counts().sort_index().values,
        color="#9B59B6",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Travel Group Size")
    axes[1, 1].set_xlabel("Group Size")

    sns.histplot(
        data=df,
        x="CryoSleep",
        hue="VIP",
        multiple="fill",
        shrink=0.8,
        palette=["#F39C12", "#8E44AD"],
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("CryoSleep vs VIP (%)")
    axes[1, 2].set_ylabel("Proportion")

    for ax in axes.flat:
        sns.despine(ax=ax)
    plt.savefig(
        "eda_plots/demographics_analysis.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def correlation_heatmap(df) -> None:
    """Correlation heat-map – Seaborn clustermap style"""
    print("\n" + "=" * 60)
    print("🔗 FEATURE CORRELATION & RELATIONSHIP ANALYSIS")
    print("=" * 60)

    numerical = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]
    numerical = [c for c in numerical if c in df.columns]

    corr_df = df[numerical].copy()
    corr_df["Transported"] = df["Transported"].astype(int)
    corr_df["CryoSleep"] = pd.to_numeric(df["CryoSleep"], errors="coerce").astype(
        "Int64"
    )
    corr_df["VIP"] = pd.to_numeric(df["VIP"], errors="coerce").astype("Int64")

    corr = corr_df.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix", fontsize=16, weight="bold")
    sns.despine()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(
        "eda_plots/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    strong = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .abs()
        .stack()
        .reset_index()
    )
    strong.columns = ["var1", "var2", "abs_r"]
    strong = strong[strong.abs_r > 0.3].sort_values("abs_r", ascending=False)

    print("\n🔍 SIGNIFICANT CORRELATIONS (|r| > 0.3):")
    for _, row in strong.iterrows():
        r_val: float = corr.loc[row.var1, row.var2]
        print(
            f"   • {row.var1} ↔ {row.var2}: {r_val:.3f} "
            f"({'strong' if abs(r_val) > 0.7 else 'moderate'} "
            f"{'positive' if r_val > 0 else 'negative'})"
        )

    if strong.empty:
        print("   • No significant correlations (|r| > 0.3).")


def main():
    """Main analysis function for stakeholder presentation"""
    df = load_data()
    if df is None:
        return

    data_quality_assessment(df)
    passenger_demographics_analysis(df)
    correlation_heatmap(df)


if __name__ == "__main__":
    main()
