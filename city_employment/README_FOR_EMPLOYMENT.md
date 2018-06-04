# Employment model for IIEG at Jalisco
## Author: Ing. Raul Romero Barragan 

### Summary

The procedure involving this code represents efforts to estimate 
with high accuracy employment 
levels on several cities inside the state of Jalisco. 

Main code gets the time series (provided a DB) and checks different regression 
models (randomForest, lasso, ridge, xgboost) with a simple and independent hyperparameter tunning each.

### Installation 
*Recommended* installation on a virtualenv. 
On folder `city_employment` install `setup.py` with `pip install -e .`. All the required packages 
will be then installed.

### Previous considerations
 
 > Go to `utils.py` (line 130) and `utils_yearly.py` (line 129) and replace with your own information 
 (DB info).
 
> There's a config file under the name `city_model.config`. This file has all params model will check 
 in order to find the best model.
Main modifier of this config is called `CITY`. It must be a list of lower-underscored cities 
 (contained in DB).
 
 

### Running code
 Go to `/city_employment` and run `python monthly_city_estimator -h` for help.
 
 To run **montlhy estimator**: `python monthly_city_estimator all city_model.config -v`
 To run **yearly estimator**: `python monthly_city_estimator all city_model.config -v -y`
 
 Code will then save a .json file under the name `configuration` (`configuration_yearly` 
 if the case) that contains estimations for a given set of cities on a given specific month.
 
 
 ### Considerations
 
 > This code was developed as a final project of Financial Engineering
  PAP (Professional Application Project), so it has a lot of improvements that **could 
  (and must be) implemented.**
 > Proposed params on config file are far from the optimal set, but are the ones that showed a 
 good performance for the main objective.
 > Jalisco is a state with a great potential, so it deserves the best tools in order to design
 the best policies for the region. 