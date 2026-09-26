import pandas as pd
from scipy.stats import shapiro, friedmanchisquare, wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import pingouin as pg


# Dummy dataframe, swap out for your real ones, may have to merge multiple if results are stored under separate CSVs
df = pd.DataFrame({
    "User": [f"U{i}" for i in range(1, 21)],

    "XYPlot": [
        17.2, 18.1, 16.5, 17.8, 16.9,
        18.4, 17.5, 16.8, 18.0, 17.1,
        16.7, 18.2, 17.6, 16.4, 17.9,
        18.3, 17.0, 16.6, 17.4, 18.1
    ],

    "XYPlot_AV": [
        11.4, 12.0, 11.1, 11.7, 10.9,
        12.3, 11.5, 11.2, 12.1, 11.0,
        10.8, 12.2, 11.6, 10.7, 11.9,
        12.4, 11.3, 10.9, 11.4, 12.0
    ],

    "XYPlot_DV": [
        13.2, 13.8, 12.9, 13.5, 13.1,
        14.0, 13.4, 12.8, 13.7, 13.0,
        12.7, 13.9, 13.3, 12.6, 13.6,
        14.1, 13.1, 12.9, 13.4, 13.8
    ],

    "vxvy": [
        8.2, 8.7, 8.5, 8.0, 8.4,
        8.9, 8.3, 8.1, 8.6, 8.2,
        7.9, 8.8, 8.4, 8.0, 8.5,
        8.7, 8.1, 8.3, 8.6, 8.2
    ],
})


# Representations
representations = list(df.columns[1:])


# Shapiro-Wilk normality test, use the within-user pairwise differences because the observations are paired/repeated measures.
print('SHAPIRO-WILK NORMALITY TESTS')

normality_results = []

for rep1, rep2 in combinations(representations, 2):

    # Get differences in EER
    differences = df[rep1] - df[rep2]

    # Do Shapiro-Wilk test for normality
    W, p = shapiro(differences)

    # Append the normality results
    normality_results.append({
        "Comparison": f"{rep1} - {rep2}",
        "W": W,
        "p": p,
        "Normal": p >= 0.05
    })

    print(
        f"{rep1} - {rep2}: "
        f"W = {W:.3f}, "
        f"p = {p:.4f}, "
        f"normal = {'NORMAL' if p >= 0.05 else 'NOT NORMAL'}"
    )


# Select analysis based on Shapiro-Wilk
# If ANY pairwise difference is non-normal, use the nonparametric approach, Friedman + Wilcoxon signed-rank.
# Otherwise, use the parametric approach, repeated-measures ANOVA + paired t-tests.

all_normal = all(result["Normal"] for result in normality_results)

print("\nOVERALL NORMALITY ASSESSMENT")
print(
    "All pairwise differences are approximately normal."
    if all_normal
    else
    "At least one pairwise difference is non-normal."
)


# NONPARAMETRIC ANALYSIS
if not all_normal:

    # Friedman test, used as the nonparametric equivalent of a repeated-measures ANOVA.
    print("\nFRIEDMAN TEST")

    statistic, p = friedmanchisquare(
        *[df[rep] for rep in representations]
    )

    print(
        f"Friedman ({len(representations) - 1}) = "
        f"{statistic:.3f}, "
        f"p = {p:.4g}"
    )


    # Pairwise Wilcoxon signed-rank tests, used to determine which specific representations differ.
    print("\nPAIRWISE WILCOXON SIGNED-RANK TESTS")

    results = []

    for rep1, rep2 in combinations(representations, 2):

        # Compare the same users across the two representations.
        W, p_unc = wilcoxon(
            df[rep1],
            df[rep2],
            alternative="two-sided"
        )

        # Store the test results for later multiple-comparison correction.
        results.append({
            "A": rep1,
            "B": rep2,
            "W": W,
            "p_unc": p_unc
        })


    pairwise = pd.DataFrame(results)


    # Holm correction for multiple comparisons, as making multiple comparisons increases the probability of a false positive.
    reject, p_corr, _, _ = multipletests(
        pairwise["p_unc"],
        alpha=0.05,
        method="holm"
    )

    pairwise["p_corr"] = p_corr
    pairwise["Significant"] = reject


    # Print pairwise results
    print("\nPairwise Wilcoxon tests (Holm corrected):")

    for _, r in pairwise.iterrows():

        print(
            f"  {r['A']} vs {r['B']}: "
            f"W = {r['W']:.1f}, "
            f"p = {r['p_unc']:.4g}, "
            f"Holm p = {r['p_corr']:.4g}, "
            f"significant = {'YES' if r['Significant'] else 'NOT SIGNIFICANT'}"
        )


    # Determine which representations are significantly different from ALL other representations.
    # A representation is marked YES only if all three of its pairwise comparisons are significant.
    print("\nSIGNIFICANTLY DIFFERENT FROM ALL OTHER REPRESENTATIONS?")

    for representation in representations:

        comparisons = pairwise[
            (pairwise["A"] == representation) |
            (pairwise["B"] == representation)
        ]

        all_significant = comparisons["Significant"].all()

        print(
            f"  {representation}: "
            f"{'SIGNIFICIANT' if all_significant else 'NOT SIGNIFICANT'}"
        )


# PARAMETRIC ANALYSIS
else:

    # Repeated-measures ANOVA, used to test whether there is an overall difference between representations.
    print("\nREPEATED-MEASURES ANOVA")

    # Convert wide data into long format for the repeated-measures ANOVA.
    long_df = df.melt(
        id_vars="User",
        var_name="Representation",
        value_name="EER"
    )
    
    anova = pg.rm_anova(
        data=long_df,
        dv="EER",
        within="Representation",
        subject="User",
        detailed=True
    )

    row = anova[anova["Source"] == "Representation"].iloc[0]
    error_df = anova.loc[anova["Source"] == "Error", "DF"].iloc[0]

    print(
        f"Repeated-measures ANOVA: "
        f"F({int(row['DF'])}, {int(error_df)}) = "
        f"{row['F']:.2f}, "
        f"p = {row['p_unc']:.4g}"
    )


    # Pairwise paired t-tests, used to determine which specific representations differ.
    print("\nPAIRWISE PAIRED T-TESTS")

    results = []

    for rep1, rep2 in combinations(representations, 2):

        # Compare the same users across the two representations.
        t_stat, p_unc = ttest_rel(
            df[rep1],
            df[rep2]
        )

        # Store the test results for later multiple-comparison correction.
        results.append({
            "A": rep1,
            "B": rep2,
            "t": t_stat,
            "p_unc": p_unc
        })


    pairwise = pd.DataFrame(results)

    # Holm correction for multiple comparisons, as making multiple comparisons increases the probability of a false positive.
    reject, p_corr, _, _ = multipletests(
        pairwise["p_unc"],
        alpha=0.05,
        method="holm"
    )

    pairwise["p_corr"] = p_corr
    pairwise["Significant"] = reject


    # Print pairwise results
    print("\nPairwise paired t-tests (Holm corrected):")

    for _, r in pairwise.iterrows():

        print(
            f"  {r['A']} vs {r['B']}: "
            f"t = {r['t']:.3f}, "
            f"p = {r['p_unc']:.4g}, "
            f"Holm p = {r['p_corr']:.4g}, "
            f"significant = {'YES' if r['Significant'] else 'NO'}"
        )


    # Determine which representations are significantly different from ALL other representations. 
    # A representation is marked YES only if all three of its pairwise comparisons are significant.
    print("\nSIGNIFICANTLY DIFFERENT FROM ALL OTHER REPRESENTATIONS?")

    for representation in representations:

        comparisons = pairwise[
            (pairwise["A"] == representation) |
            (pairwise["B"] == representation)
        ]

        all_significant = comparisons["Significant"].all()

        print(f"{representation}: {'SIGNIFICANT' if all_significant else 'NOT SIGNIFICANT'}")