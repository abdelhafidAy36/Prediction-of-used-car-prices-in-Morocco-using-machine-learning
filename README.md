# Prediction of used car prices in Morocco using ML

Predicting the price of a used car has been a very interesting area of research, as it requires notable effort and knowledge on the part of the domain expert. A considerable number of distinct attributes are examined for a reliable and accurate prediction. To simplify this process, we decided to use machine learning techniques to build an accurate prediction model of used car prices in Morocco.

To accomplish this, we collected data from a local website sells used cars in Morocco using web scraper that was written in Python programming language. Then, preparation and pre-processing techniques were applied to this data in order to obtain clean data
to build our model. Then we have experimented and evaluated many machine learning algorithms. Among all the experiments and after hyperparameter tuning, the CatBoost model got the best score with 90,86% (R2 Score), followed by LightGBM with 89,72% and XGBoost with 89,51%. 

And finally, the final prediction model has been integrated into a web application for ease of use, so that public users and also entrepreneurs can use this model to make predictions on used cars, in addition to other services provided by the web application.
