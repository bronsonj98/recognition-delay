"""Phenotype rules for the three conditions in the manuscript.

These are method constants and published coding identifiers, never participant
values or extracted lookup tables. Supply required dictionaries independently.
"""

CFG = {'first_occurrence_sources': {'recognition_default': [20, 21, 30, 31, 40, 41]},  # death, primary care, hospital
 'self_report_illness_codes': {'diabetes': [1220, 1222, 1223],
                               'ckd': [1192, 1193, 1194, 1519, 1520, 1607, 1608, 1609],
                               'anaemia': [1446, 1330, 1331, 1332]},
 'conditions': {'diabetes': {'first_occurrence_fields': [130706,
                                                         130708,
                                                         130710,
                                                         130712,
                                                         130714]},
                'ckd': {'first_occurrence_fields': [132032]},
                'anaemia': {'first_occurrence_fields': [130622,
                                                        130624,
                                                        130626,
                                                        130628,
                                                        130630,
                                                        130632,
                                                        130634,
                                                        130636,
                                                        130638,
                                                        130640,
                                                        130642,
                                                        130644,
                                                        130646,
                                                        130648]}}}

# Drug lists (drug classes v3.1). A drug name counts when it matches 'include' and not 'exclude'. Names are the lower-cased
# prescription drug name (Read 2 drug codes mapped through the UK Biobank lookup) or the decoded self-reported medication text
# (field 20003, coding 4). British National Formulary codes are not used.
DRUG_CLASSES = {'antidiabetic': {'exclude': 'lancet|test strip|strips|sharps|disposal unit|needle|syringe|safe-clip|clipping|meter\\b|re-usable '
                             'pen|reusable pen|injection device|novopen|humapen|autopen|clikstar|saxenda|wegovy|guar '
                             'gum|glucose|hypodermic|reusable|insulin pen\\b|nova 3\\b',
                  'include': '\\b(metformin|gliclazide|glimepiride|glipizide|glibenclamide|tolbutamide|chlorpropamide|tolazamide|repaglinide|nateglinide|pioglitazone|rosiglitazone|sitagliptin|vildagliptin|saxagliptin|linagliptin|alogliptin|gliptin|dapagliflozin|canagliflozin|empagliflozin|ertugliflozin|gliflozin|exenatide|liraglutide|lixisenatide|dulaglutide|semaglutide|albiglutide|acarbose|insulin|isophane|glucophage|sukkarto|diagemet|metabet|bolamyn|glucient|victoza|bydureon|byetta|lyxumia|trulicity|ozempic|rybelsus|xultophy|suliqua|eperzan|diamicron|diaglyk|amaryl|daonil|euglucon|diabetamide|glibenese|minodiab|diabinese|tolanase|rastinon|novonorm|prandin|starlix|actos|competact|avandia|avandamet|januvia|janumet|galvus|eucreas|onglyza|komboglyze|trajenta|jentadueto|vipidia|vipdomet|incresync|forxiga|xigduo|invokana|vokanamet|jardiance|synjardy|glyxambi|qtern|steglatro|glucobay|humulin|novorapid|novomix|lantus|levemir|humalog|insulatard|hypurin|apidra|tresiba|toujeo|actrapid|mixtard|insuman|monotard|ultratard|velosulin|protaphane|actraphane|initard|penmix|abasaglar|semglee|fiasp|lyumjev|admelog|ryzodeg|optisulin|lente|semitard|humaject|gliquidone|glurenorm|glibornuride|glybenclamide|glyburide|troglitazone|glucamet|oral '
                             'hypoglycaemic|efficib|velmetia|ristaben|tesavel|xelevia|jalra|xiliarx|icandra|zomarist|glubrava|glustin|tandemact|edistride|ebymect|segluromet|steglujan|trazec|exubera)'},
 'iron': {'exclude': 'iron overload|desferrioxamine|deferasirox|deferiprone|multivitamin|bc-500|-vit\\b|solution '
                     '0\\.025|folic|folate|ferrocap-f|pregaday|fefol|galfer fa|galfer '
                     'f\\.a|ferfolic|meterfolic|folex|folicin|lexpec|pregnavite',
          'include': '\\b(ferrous|ferric|iron\\b|sodium feredetate|polysaccharide-iron|slow[- '
                     ']?fe|feospan|fersaday|fersamal|galfer|sytron|ferrograd|ferrocontin|fefol|niferex|venofer|ferinject|cosmofer|monofer|diafer|jectofer|feraccru|pregaday|plesmet|spatone|ironorm|fe-cap|ferromyn|ironedetate|ferrocap)'}}
