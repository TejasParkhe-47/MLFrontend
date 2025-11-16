from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import math
import pandas as pd
import numpy as np
import os
import json

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

    def convert_flat_forest_to_reactflow(self, input_path, output_path="Output/reactflow_converted.json"):
        """
        Converts flat tree JSON (like {"Tree 1": {"0": [...], "1": [...]}}) into
        React Flow compatible format with nodes and edges.
        """
        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path}")
            return

        with open(input_path, "r") as f:
            forest_data = json.load(f)

        forest_output = {}

        def build_reactflow(tree_data, tree_id):
            nodes, edges = [], []

            for i, (node_id, details) in enumerate(tree_data.items()):
                feature, op, threshold, next_node = details

                # Create main node
                nodes.append({
                    "id": f"{tree_id}-{node_id}",
                    "data": {"label": f"{feature} {op} {threshold}"},
                    "position": {"x": (i % 4) * 320, "y": (i // 4) * 200},
                    "style": {
                        "background": "#e0e7ff",
                        "border": "2px solid #4f46e5",
                        "borderRadius": 12,
                        "padding": 10,
                        "fontWeight": 600,
                        "textAlign": "center",
                        "minWidth": 140
                    }
                })

                # Edge logic
                if isinstance(next_node, (int, float)):
                    # Leaf node
                    leaf_id = f"{tree_id}-{node_id}-leaf"
                    nodes.append({
                        "id": leaf_id,
                        "data": {"label": f"🌿 Leaf → {int(next_node)}"},
                        "position": {"x": (i % 4) * 320 + 180, "y": (i // 4) * 200 + 80},
                        "style": {
                            "background": "#dcfce7",
                            "border": "2px solid #16a34a",
                            "borderRadius": 12,
                            "padding": 10,
                            "fontWeight": 600,
                        },
                    })
                    edges.append({
                        "id": f"e-{tree_id}-{node_id}-{leaf_id}",
                        "source": f"{tree_id}-{node_id}",
                        "target": leaf_id,
                        "label": "Yes" if op == "<=" else "No",
                        "animated": True,
                        "labelBgPadding": [6, 2],
                        "labelBgBorderRadius": 4,
                        "labelBgStyle": {"fill": "#fef9c3", "stroke": "#ca8a04"},
                        "style": {"stroke": "#16a34a"},
                    })

                elif isinstance(next_node, str):
                    # Next split node
                    edges.append({
                        "id": f"e-{tree_id}-{node_id}-{tree_id}-{next_node}",
                        "source": f"{tree_id}-{node_id}",
                        "target": f"{tree_id}-{next_node}",
                        "label": "Yes" if op == "<=" else "No",
                        "animated": True,
                        "labelBgPadding": [6, 2],
                        "labelBgBorderRadius": 4,
                        "labelBgStyle": {"fill": "#e0e7ff", "stroke": "#4f46e5"},
                        "style": {"stroke": "#4f46e5"},
                        
                    })

            return {"nodes": nodes, "edges": edges}

        for tree_name, tree_data in forest_data.items():
            forest_output[tree_name] = build_reactflow(tree_data, tree_name)

        os.makedirs("Output", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(forest_output, f, indent=4)

        print(f"✅ ReactFlow-compatible forest saved to {output_path}")


    class DecisionTree:
        feature_names = 0

        def __init__(self, outer_instance, features, max_depth=None):
            self.outer = outer_instance
            self.max_depth = max_depth
            self.tree = None
            self.feature_names = features
            self.dic = self.outer.dic

        def export_tree_structure(self, node=None, parent=None, tree_id=1, level=0, pos_x=0, nodes=None, edges=None, branch_label=None):
            """
            Recursively generate a structured decision tree suitable for React Flow.
            - node: the subtree root (tuple) or leaf (int/label)
            - parent: id of parent node
            - tree_id: which tree index
            - level: depth level (root=0)
            - pos_x: horizontal position for this node
            - nodes, edges: lists to append
            - offset: horizontal spacing baseline
            Returns (nodes, edges)
            """
            if nodes is None:
                nodes, edges = [], []

            # if node param not provided, use stored tree
            if node is None:
                node = self.tree

            base_offset = 800/(level+1)

            # Leaf node
            if not isinstance(node, tuple):
                node_id = f"{tree_id}-{len(nodes)}"
                nodes.append({
                    "id": node_id,
                    "data": {"label": f"🌿 Leaf → {int(node)}"},
                    "position": {"x": pos_x, "y": level * 280},
                    "style": {
                        "background": "#dcfce7",
                        "border": "2px solid #16a34a",
                        "borderRadius": 12,
                        "padding": 10,
                        "fontWeight": 600,
                        "textAlign": "center",
                        "midwidth":120
                    }
                })
                if parent:
                         edges.append({
                "id": f"e-{parent}-{node_id}",
                "source": parent,
                "target": node_id,
                "label": branch_label or "",
                "animated": True,
                "labelBgPadding": [6, 2],
                "labelBgBorderRadius": 4,
                "labelBgStyle": {"fill": "#fef9c3", "stroke": "#ca8a04"},
                "style": {"stroke": "#16a34a"}
            })
                return nodes, edges

            # Split node
            feature, threshold, left_child, right_child = node
            node_id = f"{tree_id}-{len(nodes)}"
            label = f"{self.feature_names[feature]} ≤ {round(threshold, 2)}"
            # build label using feature name if available
            
              
            nodes.append({
        "id": node_id,
        "data": {"label": label},
        "position": {"x": pos_x, "y": level * 220},
        "style": {
            "background": "#e0e7ff",
            "border": "2px solid #4f46e5",
            "borderRadius": 12,
            "padding": 10,
            "fontWeight": 600,
            "textAlign": "center",
            "minWidth": 160
        }
    })

            if parent:
                edges.append({
            "id": f"e-{parent}-{node_id}",
            "source": parent,
            "target": node_id,
            "label": branch_label or "",
            "animated": True,
            "labelBgPadding": [6, 2],
            "labelBgBorderRadius": 4,
            "labelBgStyle": {"fill": "#e0e7ff", "stroke": "#4f46e5"},
            "style": {"stroke": "#4f46e5"}
        })

            # compute horizontal offsets for children (decrease spread as depth grows)
            # ensure offset division never zero
            
            left_offset = pos_x - base_offset
            right_offset = pos_x + base_offset

            # recurse left and right, building nodes/edges in same lists
            nodes, edges = self.export_tree_structure(
                left_child, node_id, tree_id, level + 1, left_offset, nodes, edges, branch_label="Yes"
                )
            nodes, edges = self.export_tree_structure(
                right_child, node_id, tree_id, level + 1, right_offset, nodes, edges, branch_label="No"
            )

            return nodes, edges
        

        

        
        
        def build_tree_with_highlighted_path(self, x, tree_id=1):
            """
    Build tree structure and highlight path in one pass - guaranteed to match
    """
            nodes = []
            edges = []
            path_nodes = set()
            path_edges = set()
    
            def build_recursive(node, parent=None, level=0, pos_x=0, node_counter=[0], branch_label=None):
        # Generate node ID
                node_id = f"{tree_id}-{node_counter[0]}"
                node_counter[0] += 1
        
        # Add to path if this is the current node being processed
                current_node_in_path = node_id
        
        # Leaf node
                if not isinstance(node, tuple):
                    nodes.append({
                "id": node_id,
                "data": {"label": f"🌿 Class → {int(node)}"},
                "position": {"x": pos_x, "y": level * 200},
                "type": "output"
            })
                    if parent:
                        edges.append({
                    "id": f"e-{parent}-{node_id}",
                    "source": parent,
                    "target": node_id,
                    "label": branch_label or "",
                    "type": "smoothstep"
                })
                    return node_id, int(node)
        
        # Split node
                feature, threshold, left_child, right_child = node
                label = f"{self.feature_names[feature]} ≤ {round(threshold, 3)}"
        
                nodes.append({
            "id": node_id,
            "data": {"label": label},
            "position": {"x": pos_x, "y": level * 200},
            "type": "default"
        })
        
                if parent:
                    edges.append({
                "id": f"e-{parent}-{node_id}",
                "source": parent,
                "target": node_id,
                "label": branch_label or "",
                "type": "smoothstep"
            })
        
        # Determine which path to take based on input data
                base_offset = 800/(level+1)
                if x[feature] <= threshold:
            # Take left path
                    path_nodes.add(node_id)
                    if parent:
                        path_edges.add(f"e-{parent}-{node_id}")
                    child_id, pred = build_recursive(left_child, node_id, level + 1, pos_x - base_offset, node_counter, "Yes")
                else:
            # Take right path  
                    path_nodes.add(node_id)
                    if parent:
                        path_edges.add(f"e-{parent}-{node_id}")
                    child_id, pred = build_recursive(right_child, node_id, level + 1, pos_x + base_offset, node_counter, "No")
        
                return node_id, pred
    
    # Build the tree and get prediction
            root_id, prediction = build_recursive(self.tree)
            path_nodes.add(root_id)  # Root is always in path
    
    # Apply highlighting
            for node in nodes:
                if node["id"] in path_nodes:
                    node["style"] = {
                "background": "#dcfce7",
                "border": "4px solid #10b981",
                "borderRadius": "12px",
                "padding": "10px",
                "fontWeight": "bold",
                "color": "#166534",
                "boxShadow": "0 0 15px #22c55e",
                "textAlign": "center",
                "minWidth": "140px"
            }
    
            for edge in edges:
                if edge["id"] in path_edges:
                    edge["style"] = {
                "stroke": "#10b981",
                "strokeWidth": 4,
                "strokeDasharray": "5,5"
            }
            edge["animated"] = True
            edge["labelStyle"] = {"fill": "#10b981", "fontWeight": "bold"}
    
            return {"nodes": nodes, "edges": edges, "prediction": prediction}
        

        def export_tree_with_prediction_path(self, x, tree_id=1):
            """
    Generate complete tree structure with highlighted prediction path
    """
    # First build the complete tree structure
            nodes, edges = self.export_tree_structure2(node=self.tree, tree_id=tree_id, pos_x=0, level=0)
    
    # Trace the prediction path using the same node ID generation
            path_keys = self.trace_prediction_path(x, node=self.tree, node_id_prefix=f"{tree_id}-")
    
            print(f"Tree {tree_id} - Prediction path: {path_keys}")
            print(f"Tree {tree_id} - All node IDs: {[n['id'] for n in nodes]}")
    
    # Verify the path exists in the tree
            missing_nodes = set(path_keys) - set(n['id'] for n in nodes)
            if missing_nodes:
                print(f"WARNING: Missing nodes in tree: {missing_nodes}")
    
    # Highlight nodes on the prediction path
            for node in nodes:
                if node["id"] in path_keys:
                    node["style"] = {
                "background": "#dcfce7",
                "border": "4px solid #10b981",
                "borderRadius": "12px",
                "padding": "10px",
                "fontWeight": "bold",
                "color": "#166534",
                "boxShadow": "0 0 15px #22c55e",
                "textAlign": "center",
                "minWidth": "140px"
            }
                else:
                    node["style"] = {
                "background": "#eef2ff",
                "border": "2px solid #4f46e5",
                "borderRadius": "12px",
                "padding": "10px",
                "fontWeight": "600",
                "color": "#374151",
                "textAlign": "center",
                "minWidth": "140px"
            }
    
    # Highlight edges on the prediction path
            path_pairs = set(zip(path_keys, path_keys[1:]))
    
            for edge in edges:
                if (edge["source"], edge["target"]) in path_pairs:
                    edge["style"] = {
                "stroke": "#10b981",
                "strokeWidth": 4,
                "strokeDasharray": "5,5"
            }
                    edge["animated"] = True
                    edge["labelStyle"] = {"fill": "#10b981", "fontWeight": "bold"}
                else:
                    edge["style"] = {
                "stroke": "#4f46e5",
                "strokeWidth": 2
            }
                    edge["animated"] = False
    
            return {"nodes": nodes, "edges": edges}


            


        def export_tree_structure2(self, node=None, parent=None, tree_id=1, level=0, pos_x=0, nodes=None, edges=None, branch_label=None, node_counter=None):
            """
    Recursively generate a structured decision tree suitable for React Flow.
    """
            if nodes is None:
                nodes, edges = [], []
            if node_counter is None:
                node_counter = [0]  # Use list to maintain reference across recursive calls

            if node is None:
                node = self.tree

    # Increased horizontal spacing for better visualization
            base_offset = 800/(level+1)

    # Generate node ID using the shared counter
            node_id = f"{tree_id}-{node_counter[0]}"
            node_counter[0] += 1

    # Leaf node
            if not isinstance(node, tuple):
                nodes.append({
            "id": node_id,
            "data": {"label": f"🌿 Class → {int(node)}"},
            "position": {"x": pos_x, "y": level * 200},
            "type": "output"
        })
                if parent:
                    edges.append({
                "id": f"e-{parent}-{node_id}",
                "source": parent,
                "target": node_id,
                "label": branch_label or "",
                "type": "smoothstep"
            })
                return nodes, edges

    # Split node
            feature, threshold, left_child, right_child = node
            label = f"{self.feature_names[feature]} ≤ {round(threshold, 3)}"
    
            nodes.append({
        "id": node_id,
        "data": {"label": label},
        "position": {"x": pos_x, "y": level * 200},
        "type": "default"
    })

            if parent:
                edges.append({
            "id": f"e-{parent}-{node_id}",
            "source": parent,
            "target": node_id,
            "label": branch_label or "",
            "type": "smoothstep"
        })

    # Compute horizontal offsets for children with increased spacing
            left_offset = pos_x - base_offset
            right_offset = pos_x + base_offset

    # Recurse left and right with the same counter
            nodes, edges = self.export_tree_structure2(
        left_child, node_id, tree_id, level + 1, left_offset, nodes, edges, branch_label="Yes", node_counter=node_counter
    )
            nodes, edges = self.export_tree_structure2(
        right_child, node_id, tree_id, level + 1, right_offset, nodes, edges, branch_label="No", node_counter=node_counter
    )

            return nodes, edges
        
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
            # build_tree returns a nested tuple/leaf structure and we store it
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
            
        def trace_prediction_path(self, x, node=None, path=None, node_id_prefix="", visited_nodes=None):
            """
    Trace the actual prediction path through the tree and return the correct node IDs.
    This method now traverses the tree in the same order as export_tree_structure2.
    """
            if path is None:
                path = []
            if visited_nodes is None:
                visited_nodes = [0]  # Counter to match export_tree_structure2

            if node is None:
                node = self.tree

    # Generate node ID using the same logic as export_tree_structure2
            current_node_id = f"{node_id_prefix}{visited_nodes[0]}"
            visited_nodes[0] += 1
            path.append(current_node_id)

    # Leaf node - end of path
            if not isinstance(node, tuple):
                return path

            feature, threshold, left, right = node

    # Follow the actual decision path based on input data
            if x[feature] <= threshold:
                return self.trace_prediction_path(x, left, path, node_id_prefix, visited_nodes)
            else:
                return self.trace_prediction_path(x, right, path, node_id_prefix, visited_nodes)




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
                if isinstance(X, pd.DataFrame):
                    X_sample = X.iloc[indices].values
                else:
                    X_sample = [X[i] for i in indices]

                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y_sample = y.iloc[indices].tolist()  # <-- convert to Python list
                else:
                    y_sample = [y[i] for i in indices]

                tree = self.outer.DecisionTree(self.outer,self.features,max_depth=self.max_depth)
                tree.fit(X_sample, y_sample)
                print(f"Tree {self.outer.tree_pos} training completed.","\n") 
                self.trees.append(tree)

            # After training all trees -> export each trained tree structure for ReactFlow
            forest_structures = {}
            for i, tree in enumerate(self.trees, start=1):
                if hasattr(tree, "tree") and tree.tree is not None:
                    nodes, edges = tree.export_tree_structure2(node=tree.tree, tree_id=i)
                    forest_structures[i] = {"nodes": nodes, "edges": edges}
                else:
                    forest_structures[i] = {"nodes": [], "edges": []}

            os.makedirs("Output", exist_ok=True)
            with open("Output/reactflow_forest.json", "w") as f:
                json.dump(forest_structures, f, indent=4)

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
        return train_test_split(X, y,test_size=0.4,random_state=1)
    
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

        os.makedirs("Output", exist_ok=True)
        structure.to_csv('Output/structure.csv',index=False)
        print("Saved the dataframe as CSV File")
        structure.to_json('Output/structure.json',index=False)
        print("Saved the dataframe as JSON File, named as 'structure.json'\n")
    
    def predict_stats(self,X_test,y_test):
        y_pred = pd.DataFrame(self.model.predict(X_test))
        print("\nAccuracy Score : ",round(accuracy_score(y_test.values.tolist(),y_pred)*100.0,3),"%")
        print("\nClassification Report : ")
        print(classification_report(y_test,y_pred))
