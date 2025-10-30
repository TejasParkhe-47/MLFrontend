from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import math
import pandas as pd
import numpy as np

class Generate_JSON:
    feature_names = 0
    model = 0
    data = 0
    dic = {}
    tree_pos = -1
    output_flow = {}
    z = 0
    x = 0

    def __init__(self, data, n_estimators=1, max_depth=None, random_state=42, test_size=0.3):
        self.data = pd.read_csv(data)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.test_size = test_size
        self.feature_names = self.data.columns.tolist()
        for i in range(self.n_estimators):
            self.output_flow[i+1] = list()

    def majority_vote(self,y):
        return max(set(y), key=y.count)

    class DecisionTree:
        feature_names = 0

        def __init__(self, outer_instance, features, max_depth=None):
            self.outer = outer_instance
            self.max_depth = max_depth
            self.tree = None
            self.feature_names = features
            self.dic = self.outer.dic

        def gini_impurity(self,y):
            classes = set(y)
            impurity = 1.0
            for cls in classes:
                p = y.count(cls) / len(y)
                impurity -= p ** 2
            return round(impurity,3)

        def majority_vote(self,y):
            return max(set(y), key=y.count)

        def fit(self, X, y):
            self.tree = self.build_tree(X, y, depth=0)

        def best_split(self, X, y):
            best_feature, best_threshold, best_gain, lefty, righty, currentimpurity = 0,0,0,0,0,0
            current_impurity = self.gini_impurity(y)
            n_features = len(X[0])

            for feature in range(n_features):
                values = sorted(set(row[feature] for row in X))

                for i in range(len(values) - 1):
                    threshold = (values[i] + values[i + 1])/2
                    left_y = [y[j] for j in range(len(y)) if X[j][feature] <= threshold]
                    right_y = [y[j] for j in range(len(y)) if X[j][feature] > threshold]
                    if not left_y or not right_y:
                        continue

                    p_left = round(len(left_y) / len(y),3)
                    p_right = round(len(right_y) / len(y),3)
                    gain = round(current_impurity - (p_left * self.gini_impurity(left_y) + p_right * self.gini_impurity(right_y)),3)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
                        lefty = left_y
                        righty = right_y
                        currentimpurity = current_impurity

            return best_feature, best_threshold, best_gain, lefty, righty, currentimpurity

        def check(self,node):
            if isinstance(node,tuple):
                return self.feature_names[node[0]]
            else:
                return node

        def build_tree(self, X, y, depth):
            if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                return self.majority_vote(y)

            feature, threshold, gain, lefty, righty, currentimpurity = self.best_split(X, y)
            if gain == 0:
                return self.majority_vote(y)

            left_indices = [i for i in range(len(X)) if X[i][feature] <= threshold]
            right_indices = [i for i in range(len(X)) if X[i][feature] > threshold]

            left_subtree = self.build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1)
            right_subtree = self.build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1)

            self.dic[self.outer.z] = [self.outer.tree_pos,depth,self.feature_names[feature],threshold,dict([("Class 0",lefty.count(0)),("Class 1",lefty.count(1))]),dict([("Class 0",righty.count(0)),("Class 1",righty.count(1))]),self.gini_impurity(lefty),self.gini_impurity(righty),currentimpurity,gain,self.check(left_subtree),self.check(right_subtree)]
            self.outer.z += 1

            return (feature, threshold, left_subtree, right_subtree)

        def predict_one(self, x, node):
            if not isinstance(node, tuple):
                return node
            feature, threshold, left, right = node
            if x[feature] <= threshold:
                try:
                    self.outer.output_flow.get(self.outer.x).append([self.outer.feature_names[feature],'<=',threshold,self.check(node[2])])
                except:
                    pass
                return self.predict_one(x, left)
            else:
                try:
                    self.outer.output_flow.get(self.outer.x).append([self.outer.feature_names[feature],'>',threshold,self.check(node[3])])
                except:
                    pass        
                return self.predict_one(x, right)

        def predict(self, X):
            return [self.predict_one(x, self.tree) for x in X]

    class RandomForestClassifier:
        def __init__(self, outer_instance, features, n_estimators=5, max_depth=None, random_state=None):
            self.outer = outer_instance
            self.n_estimators = n_estimators
            self.outer = outer_instance
            self.max_depth = max_depth
            self.random_state = random_state
            self.trees = []
            self.features = features
            random.seed(random_state)

        def fit(self, X, y):
            self.trees = []
            for x in range(self.n_estimators):
                # bootstrap sample
                self.outer.tree_pos = x+1
                print("Training Tree ",self.outer.tree_pos," ......")
                indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
                X_sample = [X[i] for i in indices]
                y_sample = [y.iloc[i] for i in indices]

                tree = self.outer.DecisionTree(self.outer,self.features,max_depth=self.max_depth)
                tree.fit(X_sample, y_sample)
                print(f"Tree {self.outer.tree_pos} training completed.","\n") 
                self.trees.append(tree)

        def predict(self, X):
            predictions = []
            for x in X:
                tree_preds = []
                for tree in self.trees:
                    self.outer.x+=1
                    tree_preds.append(tree.predict_one(x, tree.tree))
                predictions.append(self.outer.majority_vote(tree_preds))
            return predictions

    def store_data(self):
        return pd.read_csv(self.data)

    def resample_split_data(self):
        X = self.data.drop('Response', axis = 1)
        X = X.values
        y = self.data['Response']

        ros = RandomOverSampler(random_state=0)
        X, y = ros.fit_resample(X,y)
        return train_test_split(X, y,test_size=0.2,random_state=1)
    
    def train_model(self,X,y):
        rf = self.RandomForestClassifier(self,self.feature_names,self.n_estimators, max_depth=10, random_state=42)
        rf.fit(X,y)
        self.model = rf

    def to_json(self,X,y):
        print("Model getting trained......")
        self.train_model(X,y)
        print("Model trained Successfully.\n")

        structure = pd.DataFrame(self.dic).T
        structure.columns = ['Tree','Level','Feature','Threshold','Value Distribution in Left Node','Value Distribution in Right Node','Gini Impurity in Left Node','Gini Impurity in Right Node','Current Impurity','Information Gain','Left Subtree','Right Subtree']

        structure.sort_values(by=['Tree','Level'], ascending=[True,True], inplace=True)

        structure[['Threshold','Gini Impurity in Left Node','Gini Impurity in Right Node','Current Impurity','Information Gain']] = structure.drop(columns=['Tree','Level','Feature','Left Subtree','Right Subtree','Value Distribution in Left Node','Value Distribution in Right Node']).astype(float)
        structure[['Tree','Level']] = structure[['Tree','Level']].astype('int')

        structure.to_csv('Output/structure.csv',index=False)
        print("Saved the dataframe as CSV File")
        structure.to_json('Output/structure.json',index=False)
        print("Saved the dataframe as JSON File, named as 'structure.json'\n")
    
    def predict_stats(self,X_test,y_test):
        y_pred = pd.DataFrame(self.model.predict(X_test))
        print("\nAccuracy Score : ",round(accuracy_score(y_test.values.tolist(),y_pred)*100.0,3),"%")
        print("\nClassification Report : ")
        print(classification_report(y_test,y_pred))