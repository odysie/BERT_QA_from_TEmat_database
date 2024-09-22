# materials database to question-answering dataset

Algorithm to transform a thermoelectric materials database to a question-answering (QA) dataset for language model fine tuning.

## Usage

The TE_database_to_QA module in the code folder can be run with the default arguments to generate a QA dataset from the thermoelectric materials database in the TE_database folder. Intended for language model fine tuning.
```
python TE_database_to_QA.py
```
Similarly for the BRAT_ann_to_QA module to generate a QA dataset from the [BRAT](https://brat.nlplab.org/) annotations. Intended for language model evaluation.
```
python BRAT_ann_to_QA.py
```

## Acknowledgements

This project was financially supported by the <u>Science and Technology Facilities Council (STFC)</u>, the <u>Royal Academy of Engineering</u> (RCSRF1819\7\10) and the Engineering and Physical Sciences Research Council (EPSRC) for PhD funding (EP/R513180/1 (2020–2021) and EP/T517847/1 (2021–2024)). The Argonne Leadership Computing Facility, which is a <u>DOE Office of Science Facility</u>, is also acknowledged for use of its research resources, under contract No. DEAC02-06CH11357.

## Citation (PENDING UPDATE)

```
 @article{sierepeklis_cole_2022,
 title={A thermoelectric materials database auto-generated from the scientific literature using chemdataextractor},
 journal={Nature News},
 publisher={Nature Publishing Group},
 author={Sierepeklis, Odysseas and Cole, Jacqueline M.},
 year={2022},
 month={Oct}} 
}
```

(PENDING UPDATE)
[![DOI](https://zenodo.org/badge/DOI/10.1038/s41597-022-01752-1.svg)](https://doi.org/10.1038/s41597-022-01752-1)