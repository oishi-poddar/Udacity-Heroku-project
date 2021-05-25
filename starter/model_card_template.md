# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model used is Linear Regression.

## Intended Use
The model is intended to be used by users to find out their salary range by using various inputs based on occupations, race etc.
## Factors
The relevant factors affecting model performance is the environemt- it requires a python environment with several packages installed.
## Metrics
The metrics for model performance used are precision, recall and fbeta score.
## Evaluation Data
The training date is split into train and test data and the test data has been used to evalaute model performance.
## Training Data
The training data used is from the file 'clean_census_data'.
The various features are :
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


## Quantitative Analyses
Upon evaluating performance of the model, it was found that the model has a 
precision of 0.7150442477876107 , 
recall of  0.2573248407643312 and a F-beta score 0.3784543325526933
## Ethical Considerations
Should keep race, sex biases in mind.
## Caveats and Recommendations
Can perform hyperparameter tuning and other model comparisons for better performance.