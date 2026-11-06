# The Label Is Not the Disease: analysis code

Analysis code for

> Moonseong Jeong, Prateek Anand, Simon A. Lee, Sungjoo Han, Zhengtong Liu, and Sriram Sankararaman. *The Label Is Not the Disease: Recognition Delay in Diagnosis Codes and Its Cost to EHR Case Finding.* Proceedings of the Machine Learning for Health Symposium, Proceedings of Machine Learning Research 346, 2026. Paper: [link to be added](https://PLACEHOLDER).

```bibtex
@inproceedings{Jeong2026Label,
  author    = {Jeong, Moonseong and Anand, Prateek and Lee, Simon A. and Han, Sungjoo and Liu, Zhengtong and Sankararaman, Sriram},
  title     = {The Label Is Not the Disease: Recognition Delay in Diagnosis Codes and Its Cost to {EHR} Case Finding},
  booktitle = {Proceedings of the Machine Learning for Health Symposium},
  series    = {Proceedings of Machine Learning Research},
  volume    = {346},
  year      = {2026}
}
```

The study uses UK Biobank participants whose blood was assayed at enrollment for HbA1c, creatinine, cystatin C, and hemoglobin. It measures how often disease-range values (diabetes, chronic kidney disease, anemia) had no diagnosis or treatment record, how slowly they were recognized, and how case-finding models trained on recorded diagnoses compare with models trained on the enrollment assays.

This repository contains the analysis scripts and two code lists. It contains no participant data, predictions, fitted models, or results. UK Biobank data are available to approved researchers through UK Biobank (https://www.ukbiobank.ac.uk); the paper used application 33127.

## Requirements

Python 3.11 with the package versions in `requirements.txt` (NumPy 1.24.3, pandas 2.0.3, PyArrow 11.0.0, SciPy 1.11.1, scikit-learn 1.3.0, statsmodels 0.14.0, matplotlib 3.7.2, joblib 1.2.0, Jinja2 3.1.2), and R with the `survival` and `splines` packages (the paper used R 4.5.1 and `survival` 3.8.9).

```bash
pip install -r requirements.txt
```

## Inputs

The scripts read input tables from a data directory and write everything else to a separate work directory. Both must be outside this repository:

```bash
export ANALYSIS_DATA_ROOT=/path/to/inputs
export ANALYSIS_WORK_ROOT=/path/to/work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The data directory holds the following files, prepared from an approved UK Biobank application. Field numbers refer to the UK Biobank showcase, and all tables must come from the same withdrawal-filtered release.

| File | Contents |
| --- | --- |
| `cohort_baseline.parquet` | One row per participant with an enrollment date: `eid`, `baseline_date` (field 53, instance 0), `sex_field31` (field 31; 0 female, 1 male), and `age_years`, computed as the days from the 15th of the birth month (fields 34 and 52) to enrollment divided by 365.2425. |
| `registrations.tsv` | Primary-care registrations (`gp_registrations`): `eid`, `data_provider`, `reg_date`, `deduct_date`, with dates as day/month/year and a missing deduction date for an open registration. |
| `registry.tsv.gz` | Tab-separated, with columns named `f.FIELD.INSTANCE.ARRAY`: `f.eid`; fields 31, 34, 52; field 53, instances 0 and 1; field 40000, instance 0; field 20002, instance 0 (all arrays); and, for every first-occurrence field in `scripts/definitions.py`, the date field and the source field that follows it. |
| `assays.tsv` | `f.eid` and fields 30750 (HbA1c), 30700 (creatinine), 30720 (cystatin C), and 30020 (hemoglobin), instances 0 and 1, and field 30040 (mean corpuscular volume), instance 0. |
| `baseline.tsv.gz` | `f.eid`, `f.53.0.0`, `f.54.0.0`, `f.189.0.0`, `f.21000.0.0` (enrollment date, assessment center, Townsend index, ethnicity). |
| `extra_baseline.tsv` | `f.eid` and, at instance 0, the arrays of field 20003 (medications), field 2443 (doctor-diagnosed diabetes), and fields 6153 and 6177 (medications). |
| `menopause_pregnancy.tsv` | `f.eid`, `f.2724.0.0` (menopause), `f.3140.0.0` (pregnancy). |
| `events.parquet` | One row per primary-care or hospital record (`gp_clinical`, `gp_scripts`, `hesin_diag`, `hesin_oper`): `eid`, `event_date`, `source_table`, `code_system` (for example `READ2`, `CTV3`, `BNF`, `DMD`, `ICD10`, `OPCS4`), `code_raw`, `code_normalized`, `drug_name`, `value1_num` (the numeric value of `value1`), and `provider_or_country` (`England_Vision`, `Scotland`, `England_TPP`, or `Wales` for primary care). |
| `coverage.parquet` | Record coverage per participant and source: `eid`, `source` (`gp` or `hes`), `coverage_end` (for `gp`, the end of the participant's primary-care records), and `censor_primary` (for `hes`, the end of hospital records in the participant's country). |
| `prior_testing.parquet` | `eid` and booleans `selected__hba1c__365d`, `selected__creatinine__365d`, `selected__haemoglobin__365d`: true when the participant's primary-care record, from their own GP data supplier, contains a code for that test in `code_lists/prior_testing_codes.tsv` 1 to 365 days before enrollment. |
| `feature_exclusions.tsv` | A copy of `code_lists/feature_exclusions.tsv`. |
| `medication_encoding.txt` | The `encoding.ukb` file distributed with the UK Biobank conversion tools, which contains data-coding 4; it translates field 20003 codes to drug names. |
| `lookups/` | Sheets of the UK Biobank primary-care code lookups (`all_lkps_maps`), each saved as gzipped tab-separated text: `read_v2_lkp.tsv.gz` and `read_ctv3_lkp.tsv.gz` (`read_code`, `term_description`, and `term_code` or `description_type`), `read_v2_drugs_lkp.tsv.gz` (`read_code`, `term_description`), and `read_v2_icd10.tsv.gz` and `read_ctv3_icd10.tsv.gz` (`read_code`, `icd10_code`). |

The code lists in `code_lists/` are not participant data. `prior_testing_codes.tsv` gives the Read 2 and CTV3 test codes that define prior-year testing for each analyte, and `feature_exclusions.tsv` gives the codes for the laboratory tests measured at UK Biobank enrollment and for their panels, which are removed before the features are built.

## Running the analyses

Run the scripts in this order from the repository root:

```bash
# cohort, folds, features, and analysis frame
python scripts/build_cohort.py
python scripts/build_frame.py
python scripts/add_diagnosis_codes.py
python scripts/build_features.py
python scripts/encode_features.py \
  --feature-cache "$ANALYSIS_WORK_ROOT/inputs/raw_features.parquet" \
  --feature-metadata "$ANALYSIS_WORK_ROOT/inputs/feature_schema.json" \
  --assay-matrix "$ANALYSIS_WORK_ROOT/inputs/participants.parquet" \
  --output-directory "$ANALYSIS_WORK_ROOT/features" \
  --audit-output "$ANALYSIS_WORK_ROOT/inputs/encoding_audit.json"

# analyses
python scripts/describe_recognition.py
python scripts/compare_groups.py
python scripts/fit_models.py --jobs 12
python scripts/evaluate_models.py --jobs 3
python scripts/sensitivity_inputs.py
python scripts/sensitivity_models.py --jobs 12
python scripts/sensitivity_describe.py
python scripts/sensitivity_evaluate.py

# figures and tables
python scripts/render_design.py
python scripts/render_paper.py
```

Summaries are written to `$ANALYSIS_WORK_ROOT/summaries`, and the figures and LaTeX tables of the paper to `$ANALYSIS_WORK_ROOT/rendered`. With 12 parallel jobs, model fitting took about 45 minutes and used about 30 GB of memory, and the whole analysis took about an hour; `--jobs` sets the number of parallel processes. The scripts refuse to overwrite earlier outputs of the preparation steps, so use a new work directory for a new run.

| Script | What it does | Paper |
| --- | --- | --- |
| `build_cohort.py` | Selects participants registered with a single GP data supplier for at least the year before enrollment and assigns five participant-level folds, balanced by supplier. | Section 2.1 |
| `build_frame.py` | Builds the enrollment assays, disease-range values ($D$), prior record ($R$), recognition dates, follow-up, prior-year testing, and primary-care values after enrollment. | Sections 2.1, 2.2; Appendix A |
| `add_diagnosis_codes.py` | Adds primary-care diagnosis codes to the prior record and to recognition, using code lists built from the UK Biobank lookups. | Section 2.2; Appendix A |
| `build_features.py`, `encode_features.py` | Build the 627 pre-enrollment features and encode and standardize them within each training fold. | Section 2.4; Appendix B |
| `describe_recognition.py` | Recording at enrollment, cumulative recognition, follow-up testing after enrollment, the enrollment value at first recognition, and persistence at the repeat assessment. | Table 2; Figure 2A, C; Tables 3, 10, 14, 15 |
| `compare_groups.py` | Five-year recognition by group, paired differences, severity by group, participants at risk, and Fine–Gray models (`severity_models.R`). | Figure 2B; Tables 4, 5, 6, 9 |
| `fit_models.py` | Fits the recorded, combined, measured, and incident-label models, the 115-column models, the recording propensity, and the positive–unlabeled (PU) correction. | Sections 2.4, 2.5; Appendices B, C |
| `evaluate_models.py` | Case-finding yield, paired bootstrap comparisons, AUROC against both targets, recall by group, and propensity validation. | Figures 2D, 3, 4, 5; Tables 7, 13, 16, 19, 23–26 |
| `sensitivity_inputs.py` | Anemia definitions without hemoglobinopathy codes, second abnormal values, and mean corpuscular volume. | Tables 8, 11, 27 |
| `sensitivity_models.py` | Stricter-threshold targets and the models on 67 columns. | Tables 21, 22 |
| `sensitivity_describe.py` | Recognition at stricter thresholds, by severity, after abnormal primary-care values, by anemia subgroup, without hemoglobinopathy evidence, and by GP data supplier. | Tables 8, 11, 12, 20, 21, 27 |
| `sensitivity_evaluate.py` | Operating points, net benefit, stricter thresholds, feature sets, and selection within GP data supplier. | Tables 17, 18, 20, 21, 22 |
| `render_design.py`, `render_paper.py` | Draw Figure 1 and render Figures 2–5 and the tables. | |

Supporting modules: `common.py` (paths, labels, censoring weights, top-fraction selection), `definitions.py` (UK Biobank fields, self-report codes, and drug lists), `survival.py` (Aalen–Johansen estimator), `event_features.py`, `features.py`, and `preprocessing.py` (features), and `figstyle.py` (figure style).

## Definitions and methods

The paper and its appendix give the full definitions. In brief:

- **Disease range ($D$).** HbA1c of 48 mmol/mol or more; creatinine eGFR (2021 CKD-EPI) and cystatin C eGFR (2012 CKD-EPI) both below 60 mL/min/1.73 m²; hemoglobin below 12 g/dL in women or 13 g/dL in men.
- **Prior record ($R$).** Self-reported diagnoses and medications at enrollment, UK Biobank first-occurrence dates, primary-care diagnosis codes, and primary-care prescriptions of glucose-lowering drugs or iron, on or before enrollment. Prescriptions and medications are matched by drug name against the lists in `definitions.py`; British National Formulary codes are not used.
- **Recognition.** The first diagnosis code from primary-care, hospital, or death records, or the first relevant primary-care prescription, after enrollment. Follow-up ends at the earlier of the end of the primary-care extract and the end of hospital records, and death is a competing event.
- **Primary-care values.** HbA1c recorded as a percentage is converted to mmol/mol and rounded to an integer, and hemoglobin of 40 or more is read as g/L. Analyses of primary-care values after enrollment use practices in England and Wales, because Scottish results often store an operator in `value1` and the number in `value2`.
- **Features.** 67 columns for demographics, GP data supplier, assessment center, calendar time, and registration length; 48 for healthcare use in the 30, 90, 180, and 365 days before enrollment; and 512 code-history columns that count primary-care and hospital codes in the year before enrollment, assigned to 256 columns per source by a fixed hash. Codes in `code_lists/feature_exclusions.tsv` are removed before any feature is computed.
- **Models.** $\ell_2$-regularized logistic regression ($C=0.1$) and histogram gradient boosting (at most 200 iterations with scikit-learn's default early stopping, learning rate 0.05, 15 leaves, at least 50 samples per leaf, $\ell_2$ penalty 1), trained in five folds. The recorded label is $R$ and the combined label is $R$ or $D$, both in all training participants; the measured label is $D$ and the incident label is recognition within 1, 2, or 5 years, both in training participants with $R=0$, the latter with inverse-probability-of-censoring weights.
- **Evaluation.** In held-out participants with $R=0$, the highest-scoring fraction of each fold is tested, with ties broken by participant identifier, and cases found per 1,000 tested are compared with 1,000 paired bootstrap samples that reselect the top fraction in each sample.

The fold seed, the hash prefix, and the random seeds are fixed in the code (`build_cohort.py`, `event_features.py`, `preprocessing.py`, `fit_models.py`, and the bootstrap functions). Changing them changes the folds, the code-history columns, or the bootstrap samples.

## Verification

We ran this code, with the package versions above, on the UK Biobank data used in the paper and compared each stage with the paper's own files. It reproduced the cohort, the folds, and the 627 feature columns of all five folds exactly, and the analysis frame exactly in every column. From these, the analysis scripts reproduced every summary behind the paper, all data tables (Tables 2–27) character for character, and all five figures pixel for pixel. Results can differ slightly with other package versions or a later UK Biobank release.

## Privacy

All outputs, including summaries, stay in the work directory, which should be kept restricted. Summary tables without participant identifiers can still disclose information through small cells; the scripts suppress counts below 5 in some outputs but do not implement a full disclosure review, so check any output before sharing it.

## License

MIT; see `LICENSE`.
