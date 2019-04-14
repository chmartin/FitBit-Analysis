This is a github repository with my analysis of my FitBit personal data.

The goal is to use TensorFlow and my personal data to classify 'office days' (weekday) vs 'recreation days' (weekend). Everyone has a different activity level, and my hope is that the model is general enough for another user to apply to their data as well. If my phone application can learn different categories of days it can better recomend actions for me to be healthier.

Let's take an example: How should my tracker respod if I am working on a cool TensorFlow project and am at the office though a weekend? If my tracker can identify this, later in the month it can reccommend to me to be more active because I did not have my normal activity levels earlier in the month. 

Currently, the app uses 'active minutes' is used to set goals for active days per week, and individual goals can be set for each different categort the tracker monitors on a daily, weekly, and monthly level. Can my tracker can use the totality of that information it classify days so a user with an abnormal work schedule monitor 'office days' (weekday) vs 'recreation days' (weekend).

My intuition was that my weekend days would have more total steps on average as I run errands. However, my office days would be more consistant because weekend activities vary greatly.
I also thought sleep longer on weekends than on weekdays. 
Those weekend days may have also higher fraction of time in sedentary state, while I relax and do some mental reset.

### How to get your data

Really easy! Here is a link with the instructions:

[Download Your Fitbit Data](https://help.fitbit.com/articles/en_US/Help_article/1133)

### What is here

I started with the jupyter notebook for notes and data exploration, however I would not recommend this for others to use. It is my personal notes and often does not use the best algorythm to do a task:

NN_weekday_classification_notes.ipynb

The best option for understanding the project is here:

nn_weekend_classif_present.ipynb

I think it contains a good summary of the project and the tests that I did.
You can see the data exploration, preparation, and NN fomulation testing. 
(I did not play with a ton of hyperparameters, but a few.)

Last, and the deliverable for outside users:

FitBit_Weekday_Classifier.py

This is a python script anyone can run on thier personal fitbit data to train, and output the classification of their personal data.
The output is a CSV with the input data and the NN classifier output!

The output of the Classifier can be seen in my Tableau Public Area:
[link](https://public.tableau.com/profile/christopher.martin1729#!/vizhome/FitBit-Analysis/Dashboard1)


nn_weekend_classif_output.ipynb
