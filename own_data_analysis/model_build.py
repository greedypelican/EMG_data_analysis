import numpy as np
import cupy as cp
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.metrics import accuracy_score, classification_report
from sktime.transformations.panel.rocket import MiniRocket
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from joblib import dump



class Filters:
    def __init__(self, raw, box, ifftn, beta):
        self.raw = raw
        self.box = box
        self.ifftn = ifftn
        self.beta = beta

    def maximum(self): 
        # Maximum Filter
        raw_maximum = ndimage.maximum_filter(self.raw,self.box,mode='nearest')
    
        return raw_maximum.real

    def minimum(self): 
        # Maximum Filter
        raw_minimum = ndimage.minimum_filter(self.raw, self.box, mode='nearest')
    
        return raw_minimum.real

    def denoise_fft(self, data): 
        # Reconstruct the signal
        fft_signal = np.fft.fft(data)
        fft_signal[self.ifftn:len(fft_signal)//2]=0
        fft_signal[len(fft_signal)//2:-self.ifftn]=0
        reconstructed_signal = np.fft.ifft(fft_signal)
    
        return reconstructed_signal.real

    def smooth(self, data): 
        # Kaiser Window Smoothing
        window_len = 11 
        s = np.r_[data[window_len-1:0:-1], data, data[-1:-window_len:-1]]
        w = np.kaiser(window_len,self.beta)
        y = np.convolve(w/w.sum(),s,mode='valid')
        smoothed_signal = y[5:len(y)-5]
    
        return smoothed_signal.real

    def filter(self):
        # Apply the filters to the data
        filtered_data = self.smooth(self.denoise_fft(self.maximum()-self.minimum()))

        return filtered_data



class PrepareData:
    def __init__(self):
        self.filtered_list = []
        self.timed_list = []
        self.root_path = 'own_dataset/'
        
    def import_data(self):
        # import the data from the csv files
        for i in range(4):
            self.timed_list.append([0, self.root_path + f'data4/rest/rest{i}.csv'])
            self.timed_list.append([1, self.root_path + f'data4/rock/rock{i}.csv'])
            self.timed_list.append([2, self.root_path + f'data4/scissors/scissors{i}.csv'])
            self.timed_list.append([3, self.root_path + f'data4/right/right{i}.csv'])
            self.timed_list.append([4, self.root_path + f'data4/left/left{i}.csv'])
            self.timed_list.append([5, self.root_path + f'data4/fire/fire{i}.csv'])
            self.timed_list.append([6, self.root_path + f'data4/paper/paper{i}.csv'])
            self.timed_list.append([7, self.root_path + f'data4/paper_left/paper_left{i}.csv'])
            self.timed_list.append([8, self.root_path + f'data4/paper_right/paper_right{i}.csv'])
            
        for i in range(1):
            self.timed_list.append([0, self.root_path + f'data5/rest{i}.csv'])
            self.timed_list.append([1, self.root_path + f'data5/rock{i}.csv'])
            self.timed_list.append([5, self.root_path + f'data5/fire{i}.csv'])
            self.timed_list.append([6, self.root_path + f'data5/paper{i}.csv'])
        
        print("data imported\n")
        return self.timed_list

    def preprocess_data(self):
        # preprocess the data by converting to TimeSeries and applying filters
        timed_data = self.import_data()
        for num, data in timed_data:
            # read the data into DataFrame and convert the time column to datetime
            df = pd.read_csv(data, header=None, names=['time', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'])
            df['time'] = pd.to_datetime(df['time'])
            time_series = []
            sec = 12
            
            while sec < 80: 
                # convert the data into 0.4 sec time series
                start_time = df['time'].iloc[0] + pd.Timedelta(seconds=sec)
                end_time = df['time'].iloc[0] + pd.Timedelta(seconds=sec+0.4)
                df_filtered = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                time_series.append([num, df_filtered])
                sec+=0.1
                
            for label, filtered_df in time_series: 
                # apply filters to the data and append to the data_list
                depths_list = []
                
                for i in range(6):
                    column = f'a{i}'
                    raw = filtered_df[column]
                    filters = Filters(raw, box=5, ifftn=10, beta=14)
                    depths = filters.filter()
                    depths_list.append(depths)
                    
                self.filtered_list.append(np.append(label, np.hstack(depths_list)))
        
        print("data filtered\n")
        return self.filtered_list

    def split_data(self):
        # split the data into training and testing data
        filtered_data = self.preprocess_data()
        df_2d = pd.DataFrame(filtered_data)
        df_2d_cleaned = df_2d.dropna(axis=1, how='any')

        X = df_2d_cleaned.iloc[:, 1:].values
        y = df_2d_cleaned.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #print("shape of X_train: ",X_train.shape, ", shape of y_train: ",y_train.shape)
        #print("shape of X_test: ",X_test.shape, ", shape of y_test: ",y_test.shape)
    
        print("train test splitted\n")
        return X_train, X_test, y_train, y_test

    def extract_features(self, num_kernels=400):
        # extract features from the data using MiniRocket
        X_train, X_test, y_train, y_test = self.split_data()

        # convert the data to nested format
        X_train_nested = from_2d_array_to_nested(X_train)
        X_test_nested = from_2d_array_to_nested(X_test)

        # convert the labels to 2D array
        y_train_newaxis = y_train[:, np.newaxis]
        y_test_newaxis = y_test[:, np.newaxis]

        # applt the MiniRocket transformation to the data
        rocket = MiniRocket(num_kernels)
        rocket.fit(X_train_nested)
        X_train_feature_extracted = rocket.transform(X_train_nested)
        X_test_feature_extracted = rocket.transform(X_test_nested)
        dump(rocket, 'mini_rocket.joblib')

        print("feature extracted\n")
        return X_train_feature_extracted, X_test_feature_extracted, y_train_newaxis, y_test_newaxis

    def prepare_data(self):
        # prepare the data for the model
        X_train, X_test, y_train, y_test = self.extract_features()

        print("data prepared\n")
        return X_train, X_test, y_train, y_test



class ClassifierModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = xgb.XGBClassifier(device='cuda')

    def tune_model(self):
        # tune the model using GridSearchCV
        param_grid = {
            'tree_method': ['hist'],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': [0.5, 0.7, 1],
            'min_child_weight': [1, 3, 5]
        }
        grid_model = xgb.XGBClassifier(device='cuda')
        grid_search = GridSearchCV(grid_model, param_grid, scoring='accuracy', n_jobs=-1, cv=5)
        grid_result = grid_search.fit(cp.array(self.X_train), self.y_train)
        best_params = grid_result.best_params_
        best_score = grid_result.best_score_

        print("hyperparameter tuned\n")
        return best_params, best_score

    def build_model(self):
        # build the model using the tuned parameters
        self.model.set_params(**self.tune_model()[0])
        self.model.fit(cp.array(self.X_train), self.y_train)
        dump(self.model, 'emg_classifier.joblib')
        
        print("model built\n")
        return

    def evaluate_model(self):
        # evaluate the model using the test data
        y_pred = self.model.predict(self.X_test)

        print("model evaluated\n")
        print("accuracy score : {}\n".format(accuracy_score(self.y_test, y_pred)))
        print("\nreport :    \n" + classification_report(self.y_test, y_pred))
        return


def main(args=None):
    p = PrepareData()
    X_train, X_test, y_train, y_test = p.prepare_data()

    clf = ClassifierModel(X_train, X_test, y_train, y_test)
    clf.build_model()
    clf.evaluate_model()

if __name__ == '__main__':
    main()