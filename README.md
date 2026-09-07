# Credit Risk Prediction with Explainable AI

Predicting loan default on the Home Credit dataset, with SHAP-based adverse
action notices and a fairness audit that caught a protected attribute driving
real decisions. The original model listed an applicant's **gender as a reason to
decline their loan** — an explanation that would be unlawful to send under the UK
Equality Act and the US Equal Credit Opportunity Act. This project catches that,
removes it, and shows what removal does and does not fix.

Built around honest evaluation: it benchmarks gradient boosting against a
transparent baseline, converts model outputs into the legally-required decline
reasons a lender must give, and audits those decisions for bias — then acts on
the audit rather than just reporting it. Uses the same SHAP explainability
backbone as my [sepsis-early-warning-lstm project](https://github.com/TochiOkafor),
applied to a fintech domain.

## Why This Matters

Credit scoring is one of the most heavily regulated applications of machine
learning. In the UK, the FCA requires that credit decisions be explainable to
consumers, and the Equality Act 2010 prohibits using sex as a direct input to a
lending decision. In the US, the Equal Credit Opportunity Act requires lenders to
give applicants specific reasons for a decline. A model that is slightly more
accurate but uses a protected characteristic is not a better model — it is an
illegal one. This project treats explainability and fairness as the design brief,
not a closing paragraph.

## Project Overview

Three models are compared on ~307,000 real loan applications. The champion
(LightGBM) is explained at both the global and individual level with SHAP, and
each declined applicant's explanation is turned into an adverse action notice —
the ranked reasons a lender would legally have to provide. A fairness audit by
gender and age then drives a concrete remediation: removing gender from the model
and measuring the accuracy and fairness deltas.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
— ~307,000 loan applications, 120 features covering demographics, employment,
credit history, and three external credit scores. Binary target: `TARGET = 1`
means the applicant defaulted.

The data is heavily imbalanced — the default rate is **8.1%** — which shapes every
modelling choice. Exploratory analysis showed the three external credit scores
(`EXT_SOURCE_1/2/3`) carrying the strongest correlation with default, and default
rates varying by contract type (cash loans default more than revolving loans).
The imbalance means accuracy is a useless metric here; the project uses AUC-ROC
and Average Precision instead.

## Methodology

- **Preprocessing** — median imputation for numerical features, label encoding
  for categoricals. Gender and age were held aside as sensitive attributes for
  auditing rather than dropped, so the fairness analysis runs on attributes the
  model can be tested against.
- **Class imbalance** — handled with `class_weight='balanced'` (LightGBM,
  Logistic Regression) and `scale_pos_weight` (XGBoost), which preserves
  interpretable probabilities better than resampling.
- **Models** — Logistic Regression as a transparent baseline, then XGBoost and
  LightGBM. This mirrors the "benchmark deep learning against classical
  baselines" discipline from the sepsis project: the interpretable model sets the
  bar the complex models have to clear.
- **Explainability** — SHAP `TreeExplainer` for global importance, dependence
  plots, and per-applicant waterfall plots.
- **Fairness** — `fairlearn` metrics (selection rate, false positive rate, false
  negative rate) by gender and age band, followed by remediation.

## Results

| Model | AUC-ROC | Average Precision |
|---|---|---|
| Logistic Regression | 0.7467 | 0.2264 |
| XGBoost | 0.7589 | 0.2496 |
| **LightGBM (champion)** | **0.7613** | **0.2509** |

LightGBM won on the full feature set. At a decision threshold of 0.5 it catches
about two-thirds of true defaulters (recall 0.66 on the default class) while
flagging a large share of good borrowers (precision 0.17). That trade-off is
deliberate: in lending a false negative (approving a defaulter) is a direct loss,
while a false positive (declining a good borrower) is forgone revenue, and where
the threshold sits is a business decision rather than a modelling one.

## Explainability

The three external credit scores dominate every explanation, followed by loan and
goods amounts. Dependence plots show a clean monotonic relationship — a higher
external score pushes the prediction toward "repay" — which is the reassuring
sign that the model leans on creditworthiness signals rather than demographics.

Each decline is converted into an adverse action notice. Example from the final
gender-free model:

```
Applicant #1 (predicted default probability: 92.50%)
Top reasons for decline:
  - EXT_SOURCE_3: 0.03 (SHAP contribution: +1.6005)
  - EXT_SOURCE_2: 0.28 (SHAP contribution: +0.3994)
  - AMT_GOODS_PRICE: 157500.00 (SHAP contribution: +0.3254)
  - DAYS_EMPLOYED: -263.00 (SHAP contribution: +0.1528)
  - EXT_SOURCE_1: 0.51 (SHAP contribution: +0.0937)
```

Every reason listed is a lawful basis for a credit decision. That was not true of
the original model.

## Fairness Evaluation

This is the section most credit-risk portfolio projects skip, and it is where the
project earns its keep.

### Gender: a protected attribute in the decline notices

Auditing the original model by sex (excluding the `XNA` group, ~4 records, too
few to be meaningful):

| Group | Decline rate | False positive rate | False negative rate |
|---|---|---|---|
| Female | 0.248 | 0.223 | 0.414 |
| Male | 0.407 | 0.368 | 0.252 |

Men were declined far more often than women (decline-rate ratio F/M = 0.61, below
the 0.80 four-fifths threshold), and good male borrowers were wrongly declined
more often. The cause was direct: `CODE_GENDER` was the sixth most important
feature by SHAP and appeared **inside an applicant's decline notice as a reason**.
That explanation is unlawful to send.

Gender was removed and the champion retrained. The cost was negligible and the
fairness gain was large:

| LightGBM | AUC-ROC | Avg Precision | Demographic Parity Diff | Equalised Odds Diff |
|---|---|---|---|---|
| With `CODE_GENDER` | 0.7613 | 0.2509 | 0.1586 | 0.1617 |
| Without `CODE_GENDER` | 0.7583 | 0.2487 | 0.0975 | 0.0983 |

Removing the protected attribute cost **0.003 AUC** and cut both fairness gaps by
about **39%**. The gender-free model is the final deliverable, and its decline
notices are lawful to send.

**The residual is the real lesson.** The gap shrank but did not close — a 0.0975
parity difference remains with gender fully removed from training. That residual
is carried by proxy variables correlated with gender (income, car-ownership age,
employment length, occupation). Deleting the protected column is necessary but
not sufficient, because the model reconstructs part of the signal from proxies.
Closing it further needs proxy-aware methods, which is the natural next step.

### Age: a different problem, handled differently

| Age band | Decline rate | False positive rate | False negative rate |
|---|---|---|---|
| Under 30 | 0.492 | 0.454 | 0.209 |
| 30–45 | 0.349 | 0.314 | 0.291 |
| 45–60 | 0.228 | 0.204 | 0.447 |Tools & Libraries
| 60+ | 0.124 | 0.112 | 0.644 |

The age gradient is steeper than gender (four-fifths ratio 0.25), but age is not
gender: it is legitimately predictive of credit risk, and the false-negative
pattern shows why removing it would be wrong. The model rarely misses a risky
young applicant (FNR 0.209) but often misses a risky older one (FNR 0.644).
Dropping age would blind the model to genuine risk. The defensible response is to
keep it, monitor the disparity, and be explicit about the tension. Remove gender,
monitor age — the two cases need different answers, and treating them the same
would be the mistake.

## Key Findings

- LightGBM reached 0.7613 AUC, edging XGBoost (0.7589) and clearly beating the
  Logistic Regression baseline (0.7467).
- External credit scores drive predictions; the model does not depend on
  demographics for its core signal.
- A protected attribute (gender) was influencing decisions and appearing in
  decline notices. Removing it cost 0.003 AUC and cut fairness disparity ~39%.
- Removal alone left a residual disparity carried by proxies — the central,
  often-missed lesson of fair lending.
- Age disparity is larger but partly legitimate; it is monitored rather than
  removed.

## Limitations

- **Application data only.** The bureau, previous-application, and instalment
  tables were not joined in. Adding them is the largest available accuracy gain
  and is how Kaggle leaders reach ~0.80.
- **Proxy bias remains** after gender removal, as above.
- **Single threshold.** Decisions use 0.5; a cost-sensitive threshold reflecting
  the real asymmetry between a default loss and a lost customer would be more
  realistic.
- **Age fairness is monitored, not resolved.**

## Future Improvements

- Join the bureau and previous-application tables for a meaningful AUC gain.
- Apply proxy-aware fairness mitigation (in-training constraints or per-group
  threshold adjustment) and re-measure the residual disparity.
- Add probability calibration and a cost-sensitive decision threshold.

## View Notebook

[nbviewer link — add after upload]

## How to Run

```bash
pip install -r requirements.txt
```

Download the Home Credit data from Kaggle, then run
`notebooks/credit_risk_analysis.ipynb` top to bottom. Model selection runs on the
full feature set; the compliance retrain drops `CODE_GENDER` and regenerates the
SHAP explanations and fairness metrics.

## Tools & Libraries

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Fairlearn ·
Matplotlib · Seaborn

## Author

Tochukwu (Tee) Okafor — [GitHub](https://github.com/TochiOkafor) ·
[LinkedIn](https://linkedin.com/in/contacttochukwuedith)
MRes Artificial Intelligence, University of Wolverhampton.
