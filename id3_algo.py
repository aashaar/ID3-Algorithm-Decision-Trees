# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:02:04 2018

@authors: 
    Aashaar Panchalan
	Priyadarshini Vasudevan

"""

import pandas as pd
from math import log
from collections import defaultdict
import copy
#path = "E:\\UTD\\prob2_dataset.csv"
#path="E:\\UTD\\2nd Sem\\Machine Learning CS 6375\\Assignments\\02 Decision Tree Planning\\data_sets1\\training_set.csv"
#dataset = pd.read_csv(path)
class node_struct(object):
    def __init__(self):
        self.data = None
        self.left_child = None
        self.right_child = None
        self.positive_count = None
        self.negative_count = None
        self.flag = 'f'
#to define dict of attributes
def get_attributes(dataset):
        columns = {}
        column_list = dataset.columns
        j = 0
        for i in column_list:
            j = j+1
            columns[j] = i
        return columns
    
#to calculate entropy
def entropy_calculate(dataset):
        total_entries = len(dataset)
        class_label = {}
        class_values = dataset['Class']
        class_label = defaultdict(int)
        for i in class_values:
           if i not in class_values.keys():
               class_label[i] = 1
           else:
               class_label[i] += 1
        entropy = 0
        for j in class_label:
            prob = class_label[j]/total_entries
            entropy = entropy + ((-1)*prob*log(prob,2))
        #print("Entropy= ",entropy)
        return entropy
    
#to split the data that satisifes the best attribute and its value
def split_dataset(dataset,attribute,value):
    reduced_dataset = []
    attribute = attribute -1
    for row in dataset.itertuples(index=False, name='Pandas'):
        if(row[attribute] == value):
            instance = row
            reduced_dataset.append(instance)
    return pd.DataFrame.from_dict(reduced_dataset)

#selecting the best attribute from the given dataset
def select_best_attribute(dataset,number_columns):
    #attributes = get_attributes(dataset)
    #print("initial key= ",number_columns)
    attributes = number_columns
    parent_entropy = entropy_calculate(dataset)
    #print("Parent Entropy= ",parent_entropy)
    best_information_gain = 0
    information_gain = 0
    best_attribute_index = -1
    values = [0,1]
    max_index_value = max(attributes)
    key = attributes.keys()    
    #print("key= ",key)
    #print("Max(key)= ", max(key))
    for i in key:
        if(i != max(key)):
            #print("i in for loop for Key= ",i)
            attribute_entropy = 0
            for value in values:
                #print("value= ",value)
                
                reduced_dataset_df_res = split_dataset(dataset,i,value)
                #print("reduced_dataset_df_res==> ")
                #print(reduced_dataset_df_res)
                if(reduced_dataset_df_res.notnull().values.any()):
                    reduced_entropy = entropy_calculate(reduced_dataset_df_res)
                    reduced_probability = (len(reduced_dataset_df_res)/len(dataset)) * reduced_entropy
                    attribute_entropy += reduced_probability
            information_gain = parent_entropy - attribute_entropy
            #print ("IG= ",information_gain)
            #print("Best IG = ",best_information_gain)
            #print("Best Attribute Index= ",best_attribute_index)
            if(information_gain > best_information_gain):
                best_information_gain = information_gain
                best_attribute_index = i
    #print("===================")
    #print("FINAL Best IG= ",best_information_gain)
    #print("Best Attribute Index= ",best_attribute_index)
    return best_attribute_index

#recurssive call function for nodes that are NOT pure
def decide_class(classlabels):
    negative_count = 0
    positive_count = 0
    for i in classlabels:
        if(i == 0):
            negative_count += 1
        else:
            positive_count += 1
    if positive_count >= negative_count: 
        #we choose to include positive_count = negative_count along with positive_count > negative_count
        return 1
    else:
        return 0 # for positive_count < negative_count
    
# main function
def decision_tree(dataset,number_columns):
    class_label_list = dataset["Class"]
    if(sum(class_label_list) == len(class_label_list)):
        return 1
    if(sum(class_label_list) == 0):
        return 0
    if(len(number_columns) == 1):
        return decide_class(class_label_list)
    #print("flags")
    best_attribute_index = select_best_attribute(dataset,number_columns)
    #print("flag2")
    if(best_attribute_index == -1): # Error in data OR if two or more rows with exact same values for all attributes have different class values
        return decide_class(class_label_list)
    else:
        best_attribute_name = number_columns[best_attribute_index]
        #print(best_attribute_name)
        del(number_columns[best_attribute_index])
        decision_tree_final  = {best_attribute_name:{}}
        values = [0,1]
        for value in values:
            splitdata = split_dataset(dataset,best_attribute_index,value)
            #print("Splitdata==> ")
            #print(splitdata)
            #print("columns===>")
            #print(number_columns)
            sublabels = {}
            for i in number_columns:
                sublabels[i] = number_columns[i]
            decision_tree_final[best_attribute_name][value] = decision_tree(splitdata,sublabels)
    return decision_tree_final

def create_tree(dataset,number_columns):
    class_label_list = dataset["Class"]
    if(sum(class_label_list) == len(class_label_list)):
        return 1
    if(sum(class_label_list) == 0):
        return 0
    if(len(number_columns) == 1):
        return decide_class(class_label_list)
    #print("flags")
    best_attribute_index = select_best_attribute(dataset,number_columns)
    #print("flag2")
    if(best_attribute_index == -1): # Error in data OR if two or more rows with exact same values for all attributes have different class values
        return decide_class(class_label_list)
    else:
        best_attribute_name = number_columns[best_attribute_index]
        #print(best_attribute_name)
        del(number_columns[best_attribute_index])
        node=node_struct() # initiate node structure
        node.data = best_attribute_name
        count_0 = 0
        count_1 = 0
        for i in range(len(class_label_list)):
            a = class_label_list[i]
            if(a==0):
                count_0 = count_0+1
            else:
                count_1= count_1+1
        node.positive_count = count_1
        node.negative_count = count_0
        values = [0,1]
        for value in values:
            splitdata = split_dataset(dataset,best_attribute_index,value)
            #print("Splitdata==> ")
            #print(splitdata)
            #print("columns===>")
            #print(number_columns)
            sublabels = {}
            for i in number_columns:
                sublabels[i] = number_columns[i]
            if (value ==0):
                node.left_child = create_tree(splitdata,sublabels)
            else:
                node.right_child = create_tree(splitdata,sublabels)
                
        #print("Node values-->")
        #print(node.data)
        #print(node.left_child)
        #print(node.right_child)
        #print(node.positive_count)
        #print(node.negative_count)
        #print("==========")
            
    return node

def tree_to_dict(node):
    if(node.data == 1 or node.data == 0):
        return node.data
    
    if(type(node) ==int):
        return node
    else:
        dict = {node.data:{}}
        values = [0,1]
        for value in values:
            if(value ==0):
                if(type(node.left_child) ==int):
                    dict[node.data][value] = node.left_child
                else:
                    dict[node.data][value] = tree_to_dict(node.left_child)
                    tree_to_dict(node.left_child)
            else:
                if(type(node.right_child) ==int):
                    dict[node.data][value] = node.right_child
                else:
                    dict[node.data][value] = tree_to_dict(node.right_child)
                    #tree_to_dict(node.left_child)
    return dict

#   c = get_attributes(dataset)
#   tree = decision_tree(dataset,c)

def Validate(val_dataset,tree,dataset):
    class_pred_labels = []
    for row in val_dataset.itertuples(index=False, name='Pandas'):
        #print("row")
        #print(row)
        res = predict(row,tree,dataset)
        #print("predicted op")
        #print(res)
        class_pred_labels.append(res)
    return class_pred_labels

def predict(row,tree,dataset):
    columns = get_attributes(dataset)
    
    if(tree !=0 and tree != 1):
        for i in tree:
            tree_attribute_value = tree[i]
            #print(i)
            #print(tree_attribute_value)
            index_row = get_attributes_index(i,columns)
            #print(index_row)
            row_attribute_value = row[index_row-1]
            #print("index_row")
            #print(row_attribute_value)
            val = tree_attribute_value[row_attribute_value]
            if(val == 0 or val == 1):
                return val
            else:
                return predict(row,val,dataset)
    else:
        return tree
        
#get attribute index
def get_attributes_index(i,columns):
        for j in columns:
            if(columns[j] == i):
                return j
        return -1
    
# function to print decision tree  
def printTree(tree, d = 0):
    if (tree == None or len(tree) == 0):
    #if (tree == None):
        print("   " * d, "-")
    else:
        if(type(tree) == int):
            return
        for key, value in tree.items() :
            if(type(key) == int or type(value) == int):              
                printTree(value,d)
            else:               
                if (isinstance(value, dict)):
                    for key1,value1 in tree[key].items():
                        if(type(value1)== int):
                            print ("|   " * d +str(key)+" = "+str(key1) +" : "+ str(value1))
                        else:   
                            print ("|   " * d +str(key)+" = "+str(key1))
                            printTree(value1, d+1)
                else:
                    print("|   " * d + str(key) + str(' = ') + str(value))
    



def calculate_accuracy(tree,val_path,dataset):
    val_dataset = pd.read_csv(val_path)
    no_of_attributes = len(get_attributes(val_dataset))-1
    no_of_instances  = len(val_dataset)
    predicted_values = Validate(val_dataset,tree,dataset)
    instances_correct = 0
    val_dataset_class = val_dataset['Class']
    for i in range(len(predicted_values)):
        if(val_dataset_class[i] == predicted_values[i]):
            instances_correct += 1
    accuracy = (instances_correct/no_of_instances)*100
    nodes_function = number_of_nodes(tree,dataset,0,0)
    no_of_nodes = nodes_function[0] + nodes_function[1] + 1
    no_of_leafs = nodes_function[1]
    #print("Number of instances  =  " , no_of_instances)
    #print("Number of attributes =  " , no_of_attributes)
    #print("Number of nodes      =  " , no_of_nodes)
    #print("Number of leafs      =  " , no_of_leafs)
    #print("Accuracy             =  " ,  accuracy)
    return no_of_instances,no_of_attributes,no_of_nodes,no_of_leafs,accuracy
        

def number_of_nodes(tree,dataset,nodes,leaf):
    columns = get_attributes(dataset)    
    for i in tree:
        tree_attribute_value = tree[i]
        #print(tree_attribute_value)
        val_index = [0,1]
        for j in val_index:   
            #print(i)
            #print(j)
            val = tree_attribute_value[j]
            m = [0,0]
            if(val == 0 or val == 1):
                leaf = leaf + 1
            else:
                nodes = nodes + 1
                m = number_of_nodes(val,dataset,nodes,leaf)
                nodes = m[0]
                leaf =  m[1]
            #print("node " , nodes)
            #print("leaf " , leaf)
        return nodes,leaf
    
def print_tree(tree):
    if tree == None: return
    if(type(tree) == int):
        print(tree)
    else:
        print(tree.data)
        print_tree(tree.left_child)
        print_tree(tree.right_child)
        
        
def prune(prune_tree,validation_dataset_path,dataset,decision_tree_root):  
    act_tree_dict=tree_to_dict(prune_tree)
    act_accuraccy = calculate_accuracy(act_tree_dict,validation_dataset_path,dataset) 
    updated = 'true'
    res_node = node_tobe_selected(prune_tree)
    if(res_node.data == decision_tree_root):
        return 'stop'
        
    temp_res_node = copy.deepcopy(res_node)
    res_node.flag = 't'
    res_node.left_child = None
    res_node.right_child = None
    pos_count = res_node.positive_count
    neg_count = res_node.negative_count
    if(pos_count >= neg_count):
        res_node.data = 1
    else:
        res_node.data = 0
    pruned_tree_dict=tree_to_dict(prune_tree)   
    pruned_accuracy = calculate_accuracy(pruned_tree_dict,validation_dataset_path,dataset)   
    if((pruned_accuracy[4] < act_accuraccy[4]) or (pruned_accuracy[4] == act_accuraccy[4])):
        res_node.data = copy.deepcopy(temp_res_node.data)
        res_node.left_child = copy.deepcopy(temp_res_node.left_child)
        res_node.right_child =  copy.deepcopy(temp_res_node.right_child)
        updated = 'false'
    return updated


def node_tobe_selected(tree_prune):
    if(type(tree_prune.left_child) == int and type(tree_prune.right_child) == int):
        return tree_prune
        
    if(type(tree_prune.left_child) == int):
        if((tree_prune.right_child.data != 1 and tree_prune.right_child.data != 0) and (tree_prune.right_child.flag != 't')):
            return node_tobe_selected(tree_prune.right_child) 
        else:
            return tree_prune
        
    if(type(tree_prune.right_child) == int):
        if((tree_prune.left_child.data != 1 and tree_prune.left_child.data != 0) and (tree_prune.left_child.flag != 't')):
            return node_tobe_selected(tree_prune.left_child)
        else:
            return tree_prune
        
    if((tree_prune.left_child != None) and (tree_prune.right_child != None)):
        if((tree_prune.left_child.data == 1 or tree_prune.left_child.data == 0) and (tree_prune.right_child.data == 1 or tree_prune.right_child.data == 0)):
            return tree_prune          
        if((tree_prune.left_child.flag == 't') and (tree_prune.right_child.flag == 't')):
            return tree_prune
            
    if(tree_prune.left_child != None):
        if((tree_prune.left_child.data == 1 or tree_prune.left_child.data == 0) or (tree_prune.left_child.flag == 't')):
            return node_tobe_selected(tree_prune.right_child)
            
    if(tree_prune.right_child != None):
        if((tree_prune.right_child.data == 1 or tree_prune.right_child.data == 0) or (tree_prune.right_child.flag == 't')):
            return node_tobe_selected(tree_prune.left_child)
            

    node_res = node_tobe_selected(tree_prune.left_child)
    if(node_res.data == tree_prune):
         node_res = node_tobe_selected(tree_prune.right_child)
    return node_res
    

def main():
    training_dataset_path = input("Please enter the path of the training dataset : ")
    validation_dataset_path = input("Please enter the path of the validation dataset : ")
    testing_dataset_path = input("Please enter the path of the testing dataset : ")
    pruning_factor = float(input("Please enter the pruning factor : "))
    
    #training_dataset_path="E:\\UTD\\prob2_dataset.csv"
    #validation_dataset_path ="E:\\UTD\\prob2_dataset.csv"
    #testing_dataset_path = "E:\\UTD\\prob2_dataset.csv"
    #pruning_factor = 1
    
    #training_dataset_path="E:\\UTD\\2nd Sem\\Machine Learning CS 6375\\Assignments\\02 Decision Tree Planning\\data_sets1\\training_set.csv"
    #validation_dataset_path = "E:\\UTD\\2nd Sem\\Machine Learning CS 6375\\Assignments\\02 Decision Tree Planning\\data_sets1\\validation_set.csv"
    #testing_dataset_path = "E:\\UTD\\2nd Sem\\Machine Learning CS 6375\\Assignments\\02 Decision Tree Planning\\data_sets1\\test_set.csv"
    #pruning_factor = 0.2
    
    print("Computing . . .")
    dataset = pd.read_csv(training_dataset_path)
    c = get_attributes(dataset)    
    tree_dict = decision_tree(dataset,c)
    print("*************** Pre-pruned Tree***************")
    printTree(tree_dict)
    print("*************** Pre-pruning accuracy***************")
    #cal accuracy for training dataset    
    training_output = calculate_accuracy(tree_dict,training_dataset_path,dataset)
    no_of_instances_tr = training_output[0]
    no_of_attributes_tr = training_output[1]
    no_of_nodes_tr= training_output[2]
    no_of_leafs_tr = training_output[3]
    accuracy_tr =  training_output[4]
    #cal accuracy for validation dataset
    validation_output = calculate_accuracy(tree_dict,validation_dataset_path,dataset)
    no_of_instances_val = validation_output[0]
    no_of_attributes_val = validation_output[1]
    no_of_nodes_val= validation_output[2]
    no_of_leafs_val = validation_output[3]
    accuracy_val =  validation_output[4]
    #cal accuracy for testing dataset
    testing_output = calculate_accuracy(tree_dict,testing_dataset_path,dataset)
    no_of_instances_test = testing_output[0]
    no_of_attributes_test = testing_output[1]
    no_of_nodes_test = testing_output[2]
    no_of_leafs_test = testing_output[3]
    accuracy_test =  testing_output[4]
    
    print("Number or training instances =",no_of_instances_tr)
    print("Number of training attributes =",no_of_attributes_tr)
    print("Total number of nodes in the tree =",no_of_nodes_tr)
    print("Number of leaf nodes in the tree =",no_of_leafs_tr)
    print("Accuracy of the model on training dataset =",accuracy_tr)
    print("")
    print("Number or validation instances =",no_of_instances_val)
    print("Number of validation attributes =",no_of_attributes_val)
    print("Accuracy of the model on validation dataset before pruning =",accuracy_val)
    print("")
    print("Number or testing instances =",no_of_instances_test)
    print("Number of testing attributes =",no_of_attributes_test)
    print("Accuracy of the model on testing dataset =",accuracy_test)    
    print("==============================================")
    dataset = pd.read_csv(training_dataset_path)
    d = get_attributes(dataset)
    tree_dt = create_tree(dataset,d)
    decision_tree_root = tree_dt.data
    count_prune_node = int(pruning_factor*no_of_nodes_tr)
    prune_tree = copy.deepcopy(tree_dt)
    j = count_prune_node
    i = 0
    while (i < j):
        updated_res = prune(prune_tree,validation_dataset_path,dataset,decision_tree_root)
        if(updated_res == 'stop'):
            pruned_tree_dict=tree_to_dict(prune_tree)
            break
        if(updated_res == 'false'):
            j += 1
        pruned_tree_dict=tree_to_dict(prune_tree)
        i += 1
        
   

    print("*************** Post-pruned Tree***************")
    printTree(pruned_tree_dict)
    print("*************** Post-pruning accuracy***************")
    #cal accuracy for training dataset    
    training_output = calculate_accuracy(pruned_tree_dict,training_dataset_path,dataset)
    no_of_instances_tr = training_output[0]
    no_of_attributes_tr = training_output[1]
    no_of_nodes_tr= training_output[2]
    no_of_leafs_tr = training_output[3]
    accuracy_tr =  training_output[4]
    #cal accuracy for validation dataset
    validation_output = calculate_accuracy(pruned_tree_dict,validation_dataset_path,dataset)
    no_of_instances_val = validation_output[0]
    no_of_attributes_val = validation_output[1]
    no_of_nodes_val= validation_output[2]
    no_of_leafs_val = validation_output[3]
    accuracy_val =  validation_output[4]
    #cal accuracy for testing dataset
    testing_output = calculate_accuracy(pruned_tree_dict,testing_dataset_path,dataset)
    no_of_instances_test = testing_output[0]
    no_of_attributes_test = testing_output[1]
    no_of_nodes_test = testing_output[2]
    no_of_leafs_test = testing_output[3]
    accuracy_test =  testing_output[4]
    
    print("Number or training instances =",no_of_instances_tr)
    print("Number of training attributes =",no_of_attributes_tr)
    print("Total number of nodes in the tree =",no_of_nodes_tr)
    print("Number of leaf nodes in the tree =",no_of_leafs_tr)
    print("Accuracy of the model on training dataset =",accuracy_tr)
    print("")
    print("Number or validation instances =",no_of_instances_val)
    print("Number of validation attributes =",no_of_attributes_val)
    print("Accuracy of the model on validation dataset after pruning =",accuracy_val)
    print("")
    print("Number or testing instances =",no_of_instances_test)
    print("Number of testing attributes =",no_of_attributes_test)
    print("Accuracy of the model on testing dataset =",accuracy_test)

if __name__ == '__main__':
   main()
   
