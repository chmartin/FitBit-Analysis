# script to fit and predict weekend classifier
import pandas as pd
import numpy as np
import datetime
import sys, getopt
import glob
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Global variables

#data minimums
min_steps = 1500 #steps
min_sleep = 200 #minutes

#model
units_layer1 = 7 #layer 1 units
epochs = 1000 #maximum training epochs
patience = 50 #early stop patience

def load_json(dir,type):
# Load all JSON files of 'type' from 'dir', return as pandas.DataFrame
    df = pd.DataFrame()
    for file in glob.glob(str(dir+"/"+type+"-*.json")):
        print("Reading JSON: ",file)
        df_file = pd.read_json(file,orient='records')
        df = df.append(df_file)
    return df
    
def add_weekend_flag(df):
# Add weekend flag to pandas.Dataframe, input df must have 'date_np' 
#   column containing np.datetime64 date, return pandas.DataFrame
    df['day_of_week'] = df['date_np'].dt.dayofweek
    df['weekend'] = df['day_of_week'].apply(lambda x: x > 4)
    df = df.drop('day_of_week',axis=1)
    return df
    
def prep_steps(df_steps):
# Prep steps df for merge

    df_steps = df_steps.set_index('dateTime')
    
    # group by day to get sum of steps per day
    max_steps_by_day = pd.DataFrame(df_steps.groupby(df_steps.index.date).sum())
    max_steps_by_day['sum'] = max_steps_by_day['value']
    max_steps_by_day = max_steps_by_day.drop('value',axis=1)
    
    #cut days without 1500 steps
    max_steps_by_day = max_steps_by_day[max_steps_by_day['sum'] > min_steps]
    
    #convert date to numpy datetime64 steps df
    max_steps_by_day['date_np'] = max_steps_by_day.index.to_series().apply(lambda x: np.datetime64(str(x)))
    max_steps_by_day = max_steps_by_day.reset_index()
    max_steps_by_day = max_steps_by_day.drop('index',axis=1)
    
    #add weekendflag to dataframe
    max_steps_by_day = add_weekend_flag(max_steps_by_day)
    
    return max_steps_by_day
    
def prep_sleep(df_sleep):
# Prep sleep df for merge

    #convert date to numpy datetime64 steps df
    df_sleep['date_np'] = df_sleep['dateOfSleep'].apply(lambda x: np.datetime64(str(x)))
    
    #sum all sleeps for that day together, in case of naps, etc...
    sum_sleep = (df_sleep.groupby(df_sleep['date_np']).sum()).reset_index()
    
    #drop variables with less interest
    sum_sleep = sum_sleep.drop(['infoCode','logId','timeInBed','duration','efficiency','minutesToFallAsleep','minutesAfterWakeup'],axis=1)
    
    #cut days without 200 minutes sleep
    sum_sleep = sum_sleep[sum_sleep['minutesAsleep'] > min_sleep]
    
    #add weekendflag to dataframe
    sum_sleep = add_weekend_flag(sum_sleep)
    
    return sum_sleep
    
def prep_active_minutes(df,type):
#Prep active minutes df for merge

    df_min = df
    #Rename columns
    df_min[str(type+'_min')] = df_min['value']
    df_min['date_np'] = df_min['dateTime']
    df_min = df_min.drop(['value','dateTime'],axis=1)
    
    #add weekendflag to dataframe
    df_min = add_weekend_flag(df_min)
    
    return df_min
    
def create_baseline(input_dim):
    # create model
    # based on: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    model = keras.Sequential()
    model.add(layers.Dense(units_layer1 , input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
class PrintDot(keras.callbacks.Callback):
#Print dot class for training.
#From: https://www.tensorflow.org/tutorials/keras/basic_regression
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def main(argv):
    
    try:
        opts, args = getopt.getopt(argv,"hd:",["dir="])
    except getopt.GetoptError:
        print('FitBit_Weekend_Classifier.py -d <directory_of_json_files>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('FitBit_Weekend_Classifier.py -d <directory_of_json_files>')
            sys.exit()
        elif opt in ("-d", "--dir"):
            dir = str(arg)
    
    #Load data
    df_steps = load_json(dir,"steps")
    df_sleep = load_json(dir,"sleep")
    df_sed = load_json(dir,"sedentary_minutes")
    df_la = load_json(dir,"lightly_active_minutes")
    df_ma = load_json(dir,"moderately_active_minutes")
    df_va = load_json(dir,"very_active_minutes")
    
    #clean and prepare data
    my_df_steps = prep_steps(df_steps)
    my_df_sleep = prep_sleep(df_sleep)
    my_df_sed = prep_active_minutes(df_sed,"sed")
    my_df_la = prep_active_minutes(df_la,"ls")
    my_df_ma = prep_active_minutes(df_ma,"ma")
    my_df_va = prep_active_minutes(df_va,"va")
    
    #Merge dfs
    merged = my_df_sleep.merge(my_df_sed,on=['weekend','date_np'],how='inner')
    merged = merged.merge(my_df_va,on=['weekend','date_np'],how='inner')
    merged = merged.merge(my_df_ma,on=['weekend','date_np'],how='inner')
    merged = merged.merge(my_df_la,on=['weekend','date_np'],how='inner')
    merged = merged.merge(my_df_steps,on=['weekend','date_np'],how='inner')
    
    #Prepare for training, use 20% of data for test
    X_train, X_test, y_train, y_test = train_test_split(merged.drop(['weekend','date_np'],axis=1), merged['weekend'].astype(int), test_size=0.2, stratify=merged['weekend'].astype(int),random_state=42)
    
    #Use QuantileTransformer on inputs
    rng = np.random.RandomState(304)
    scaler = QuantileTransformer(output_distribution='normal',random_state=rng)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    input_keys = len(X_train_scaled[0])
    
    #Create Model
    model = create_baseline(input_keys)
    print(model.summary())

    #Train Model
    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=patience)
    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_split = 0.2, verbose=0, batch_size=5, callbacks=[early_stop, PrintDot()])

    #Output Test classification report to confirm expected performance
    pred = model.predict(X_test_scaled)
    print('\n')
    print("Test Classification:\n",classification_report((pred > .5).astype(int),y_test))
    print('\n')
    print("Test Confusion:\n",confusion_matrix((pred > .5).astype(int),y_test))
    
    #Fit full dataset, for output
    X_full = merged.drop(['weekend','date_np'],axis=1)
    X_full_scaled = scaler.transform(X_full)
    y_full = merged['weekend'].astype(int)
    pred = model.predict(X_full_scaled)
    print('\n')
    print("Full Classification:\n",classification_report((pred > .5).astype(int),y_full))
    print('\n')
    print("Full Confusion:\n",confusion_matrix((pred > .5).astype(int),y_full))
    merged['pred'] = pd.DataFrame(pred > 0.5)
    merged.to_csv('My_NN_Classifier.csv')
    
    
if __name__ == "__main__":
   main(sys.argv[1:])