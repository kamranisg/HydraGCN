### Feature matrix
`features_all_cols.npy`
* Size is ~ 90,000 x 45
* Dataset contains demographic information, laboratory results, questionnaire, and physical examination data.


### Target labels

`targets.npy` 
* ~ 90,000 x 1 (not yet one-hot encoded)
* Three class labels (normal, pre-diabetes, and diabetes) which are based on their measured fasting glucose levels.
    0 == normal
    1 == pre-diabetes
    2 == diabetes

##### For more information please check:
CentersforDiseaseControlandPrevention(CDC),“Nationalhealthand nutrition examination survey (NHANES),” 
https://www.cdc.gov/nchs/ nhanes/index.htm, accessed: 2020-03-11.