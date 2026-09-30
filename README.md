## Pipeline Pseudocode

```text

Input:
    D = {(X_i, y_i)}              # Student dataset
    y ∈ {No_Disorder, Insomnia, Sleep_Apnea}
    M = {CatBoost, GB, SVM, RF, DT, KNN, LR, XGB}
    α = 0.05                      # Significance level

Output:
    F*                            # Final Stacking Model
    P(y | X)                      # Risk prediction
    E(X)                          # Model explanation


────────────────────────────────────────────────────────────
1. DATA_COLLECTION
────────────────────────────────────────────────────────────

D ← Collect_Data(
        source = {Google_Forms, Face_to_Face},
        universities = {
            HSTU, RU, CU, JnU, KU, MBSTU, DU
        }
     )

X ← D.features
y ← D.target


────────────────────────────────────────────────────────────
2. EDA
────────────────────────────────────────────────────────────

for each feature x_j ∈ X:

    μ_j  ← mean(x_j)
    σ_j  ← std(x_j)
    Sk_j ← skewness(x_j)
    Ku_j ← kurtosis(x_j)

    Analyze_Distribution(x_j)

OR ← Odds_Ratio(X, y)

for each x_j ∈ X:

    VIF_j ← 1 / (1 - R_j²)

    if VIF_j → high:
        flag_multicollinearity(x_j)


────────────────────────────────────────────────────────────
3. DATA_PREPROCESSING
────────────────────────────────────────────────────────────

D ← Remove_Duplicates(D)

for each sample x_i:

    z_ij ← (x_ij - μ_j) / σ_j

    if |z_ij| > 3:
        Remove(x_i)

Handle_Missing_Values(D)
Remove_Noise(D)


# Feature_Encoding

X_encoded ← Encode(
                  X,
                  method = {
                      Label_Encoding,
                      Custom_Mapping
                  }
              )

# Example:
# Sleep Quality:
# Very Poor → 0
# Poor      → 1
# Average   → 2
# Good      → 3
# Excellent → 4

# Sleep Duration:
# <4h       → 1
# 4–5h      → 2
# 5–6h      → 3
# 6–7h      → 4
# 7–8h      → 5
# ≥8h       → 6


────────────────────────────────────────────────────────────
4. TRAIN–TEST_SPLIT
────────────────────────────────────────────────────────────

(X_train, X_test,
 y_train, y_test) ← Stratified_Split(
                        X_encoded,
                        y,
                        test_size = 0.20
                    )

D_train = 80%
D_test  = 20%


────────────────────────────────────────────────────────────
5. BASELINE_MODEL_DEVELOPMENT
────────────────────────────────────────────────────────────

for model m ∈ M:

    m.fit(X_train, y_train)

    ŷ_m ← m.predict(X_test)

    P_m ← m.predict_proba(X_test)

    Acc_m  ← Accuracy(y_test, ŷ_m)
    Pre_m  ← Precision(y_test, ŷ_m)
    Rec_m  ← Recall(y_test, ŷ_m)
    F1_m   ← F1(y_test, ŷ_m)
    AUC_m  ← ROC_AUC(y_test, P_m)

Store(
    m,
    Acc_m,
    Pre_m,
    Rec_m,
    F1_m,
    AUC_m
)


────────────────────────────────────────────────────────────
6. TUNED_USING_BAYESIAN_OPTIMIZATION
────────────────────────────────────────────────────────────

for m ∈ M:

    Define search space Θ_m

    θ* ← Bayesian_Optimization(
              objective = CV_Score(m, θ),
              search_space = Θ_m
          )

    m* ← Train(m, θ*)

    M_opt ← M_opt ∪ {m*}


Objective:

    θ* = argmax_θ  CV_F1(m, θ)

where

    CV_F1 = (1/K) Σ(k=1→K) F1_k

    K = 5


────────────────────────────────────────────────────────────
7. TOP-3_STACKING_ENSEMBLE
────────────────────────────────────────────────────────────

Select top 3 optimized models:

    B = {SVM, CatBoost, LR}

for each base model b ∈ B:

    Z_b ← Cross_Validated_Prediction(
               b,
               X_train,
               y_train,
               K = 5
           )

Z ← [Z_SVM || Z_CatBoost || Z_LR]

# Meta Learner

F* ← RidgeClassifier(
          CV = 5,
          random_state = 42
      )

F*.fit(Z, y_train)


────────────────────────────────────────────────────────────
8. FINAL_PREDICTION
────────────────────────────────────────────────────────────

for new sample x:

    z_1 ← SVM(x)
    z_2 ← CatBoost(x)
    z_3 ← LR(x)

    z ← [z_1 || z_2 || z_3]

    ŷ ← F*(z)

Return ŷ


────────────────────────────────────────────────────────────
9. VALIDATION_&_ROBUSTNESS_TEST
────────────────────────────────────────────────────────────

# Stratified_5-Fold_CV

for k = 1 → 5:

    (D_train^k, D_val^k) ← Stratified_Fold(D_train, k)

    F_k ← Train(F*, D_train^k)

    score_k ← Evaluate(F_k, D_val^k)

CV_mean ← (1/5) Σ score_k


# ROC-AUC

AUC_macro ← (1/C) Σ(c=1→C) AUC_c


# Bootstrap_Confidence_Interval

for b = 1 → B:

    D_b ← Bootstrap_Sample(D_test)

    S_b ← Metric(F*, D_b)

CI_95% ← Percentile(S, [2.5%, 97.5%])


# Stress_Test

for r ∈ {
        20:80, 30:70, 40:60,
        50:50, 60:40, 70:30,
        80:20, 90:10
    }:

    D_r ← Split_By_Ratio(D_train, r)

    F_r ← Train(F*, D_r)

    S_r ← Evaluate(F_r, D_test)


# Ablation_Study

for feature_set ∈ {
        Top_9,
        Top_7,
        Top_5
    }:

    F_a ← Train(F*, feature_set)

    S_a ← Evaluate(F_a, D_test)


────────────────────────────────────────────────────────────
10. STATISTICAL_COMPARISON
────────────────────────────────────────────────────────────

Compare(F*, M_opt) using:

    McNemar_Test()
    5×2_CV_Paired_t_Test()
    Cochran_Q_Test()

if p < α:

    Difference ← Statistically_Significant

else:

    Difference ← Not_Significant


────────────────────────────────────────────────────────────
11. EXTERNAL_VALIDATION
────────────────────────────────────────────────────────────

D_ext1 ← Load_External_Data("Mendeley")
D_ext2 ← Load_External_Data("Kaggle")

for D_ext ∈ {D_ext1, D_ext2}:

    X_ext, y_ext ← Preprocess(D_ext)

    ŷ_ext ← F*.predict(X_ext)

    External_Score ← Evaluate(
                         y_ext,
                         ŷ_ext
                     )

Compare(
    Internal_Performance,
    External_Score
)


────────────────────────────────────────────────────────────
12. EXPLAINABLE_AI (XAI)
────────────────────────────────────────────────────────────

# SHAP

φ_j ← SHAP(F*, X)

Global_Importance_j ← mean(|φ_j|)

Rank features by:

    Importance_j = E(|φ_j|)


Generate:

    SHAP_Global_Plot()
    SHAP_Decision_Plot()
    SHAP_Force_Plot()


# LIME

for sample x_i:

    E_i ← LIME(
              model = F*,
              sample = x_i
          )

    Explain(
        local_features = E_i
    )


────────────────────────────────────────────────────────────
13. RISK_FACTOR_ANALYSIS
────────────────────────────────────────────────────────────

R ← Rank(
        features,
        importance = {SHAP, LIME}
    )

Identify:

    Key_Risk_Factors ← Top(R)

Risk_Strata ← {
    Low,
    Moderate,
    High
}


────────────────────────────────────────────────────────────
14. DSS
────────────────────────────────────────────────────────────

User_Input → Preprocessing_Pipeline

X_user ← Transform(User_Input)

Z_user ← [
    SVM(X_user),
    CatBoost(X_user),
    LR(X_user)
]

Risk_Class ← F*(Z_user)

Explanation ← {
    SHAP(X_user),
    LIME(X_user)
}

Risk_Level ← Classify(Risk_Class)

Recommendation ← Generate_Recommendation(
                     Risk_Level,
                     Key_Risk_Factors
                 )

Return {
    Prediction,
    Risk_Level,
    Explanation,
    Recommendation
}


────────────────────────────────────────────────────────────
15. FINAL_OUTPUT
────────────────────────────────────────────────────────────

Return:

    F*                  # Optimized Stacking Model
    Performance         # Accuracy, Precision, Recall, F1, AUC
    CI_95%              # Bootstrap_Confidence_Interval
    Statistical_Test    # p-values
    External_Validation
    Risk_Factors
    XAI_Explanation
    Real_Time_DSS
