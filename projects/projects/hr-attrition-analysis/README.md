# HR Attrition Analysis

## Overview
This project analyzes the IBM HR Analytics Employee Attrition dataset to identify key patterns and risk factors that lead to employee turnover. The goal of this project is to explain actionable insights and strategies to improve employee retention based upon data-driven strategies.

## Table of Contents
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Tools and Technologies](#tools-and-technologies)
- [Conclusion](#conclusion)

## Dataset
- **Source**: [IBM HR Analytics Employee Attrition & Performance dataset](https://www.ibm.com/analytics/hr-analytics-employee-attrition)
- **Size**: 1,470 records, 35 features
- **Features**: Job Role, Monthly Income, Age, Overtime, etc.

## Methodology
1. **Data Cleaning**: Addressed missing values and ensured data types were appropriate.
2. **Exploratory Data Analysis (EDA)**: Detailed visualizations to showcase relationships between variables.
3. **Modeling**: Integrated classification algorithms to predict attrition.
4. **Evaluation**: Measured model performance using F1-score, accuracy, precision and recall.
  
## Key Findings
- Certain positions like Sales Representatives and Laboratory Technician had higher attrition rates than other positions.
- Employees working overtime had higher attrition rates.
- Those with lower monthly income correlated to higher attrition.
- Younger employees with fewer years at the company correlated to a higher attrition likelihood.

### Visualizations

#### Employee Attrition Count

This bar chart shows the imbalance in employee attrition — significantly more employees stayed than left.

![Employee Attrition Count](https://github.com/sirinagoolbis/sirinagoolbis.github.io/blob/master/projects/projects/hr-attrition-analysis/images%3Aemployee_attrition_count.png)

## Tools and Technologies
- Python
- Pandas
- Seaborn
- Matplotlib
- Jupyter Notebook

## Conclusion
This project explains factors that correlate to a higher attrition rate for employees, which includes job role, overtime status, and income level. These findings aid HR teams in creating proactive retention strategies for such at-risk employee groups.

## Future Improvements
- Integrate SHAP/LIME for model explainability.
- Implemented a dashboard for visualization of trends.
- For better predictive power, test ensemble models (Random Forest, XGBoost).
