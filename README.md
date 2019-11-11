# PtLnc-BXE

This computational tool is designed for plant long non-coding RNA prediction, and namely PtLnc-BXE ( 'Prediction of plant lncRNAs using a Bagging-XGBoost-ensemble method with multiple features').

Datasets and tools we used are available at: https://pan.baidu.com/s/1M3zAve936BBbReoaFL8ZEQ

## Setup

1.To use this tool, clone this repository on your machine by:
```
git clone https://github.com/BioMedicalBigDataMiningLab/PtLnc-BXE.git
```

2.Download datasets and tools  
2.1 Download '/data' and '/feamodule.zip' from given url  
2.2 Unzip feamodule.zip to /PtLnc-BXE/src/feamodule/  
2.3 Unzip /PtLnc-BXE/src/feamodule/blast.zip to /PtLnc-BXE/src/feamodule/blast

### Prerequisites

To use this tool you will need:

1. linux 

2. python3
    - numpy
    - pandas
    - biopython
    - scikit-learn
    - xgboost
    - deap
    - CPAT

3. python2
    - numpy
    - regex
    - biopython
    - scikit-learn
    - scipy

4. nodejs

## Example

Run:
```
python3 PtLnc-BXE.py -i input.fa -m a -o result
```

Model selection `-m` :  
a : Arabidopsis_thaliana  
c : Chlamydomonas_reinhardtii  
h : Hordeum_vulgare  
o : Oryza_sativa_Japonica_Group  
p : Physcomitrella_patens  
s : Solanum_tuberosum

## Output files

`all_features.csv`: 175 features extracted from input transcripts.  
`result`: the classes of predicted input trainscipts.


