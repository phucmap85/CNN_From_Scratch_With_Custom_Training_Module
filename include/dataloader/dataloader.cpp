#include "dataloader.h"

Tensor *one_hot(const ll label, const ll num_classes) {
    Tensor *one_hot_tensor = new Tensor(1, 1, num_classes, 1);
    for(ll i=0;i<num_classes;i++) {
        one_hot_tensor->arr[0][0][0][i] = (i == label) ? 1.0 : 0.0;
    }
    return one_hot_tensor;
}

void load_dataset(const ll maxLength, const std::str path, std::vector<Tensor*> &X, std::vector<Tensor*> &Y, std::vector<std::str> &classes) {
    std::ifstream trainLabels((path + "/train.csv").c_str());

    if(!trainLabels.is_open()) {
        std::cerr<<"Error opening labels file";
        exit(1);
    }

    std::set<std::str> uniqueClasses;
    std::vector<std::str> Y_str_temp;
    std::vector<ll> Y_temp;

    std::str s;
    getline(trainLabels, s);
    while(getline(trainLabels, s)) {
        // -1 is no limit
        if(maxLength > 0 && sz(X) >= maxLength) break;
        
    	std::str filename = s.substr(0, s.find(",")) + (s.find(".png") == std::str::npos ? ".png" : "");
        std::str label = s.substr(s.find(",") + 1);

        Tensor *img = new Tensor(path + "/" + filename);

        X.push_back(img);
        Y_str_temp.push_back(label);
        uniqueClasses.insert(label);
    }

    for(auto &label : Y_str_temp) {
        Y_temp.push_back(std::distance(uniqueClasses.begin(), uniqueClasses.find(label)));
    }

    for(auto &cls : uniqueClasses) classes.push_back(cls);

    for(ll i=0;i<sz(Y_temp);i++) Y.push_back(one_hot(Y_temp[i], sz(classes)));

    trainLabels.close();
}

void undersampling(std::vector<Tensor*> &X, std::vector<Tensor*> &Y, const std::vector<std::str> &classes) {
    // Count samples per class
    std::vector<ll> classCount(sz(classes), 0);
    
    // Find class for each sample
    for(auto y : Y) {
        for(ll i = 0; i < sz(classes); i++) {
            if(y->arr[0][0][0][i] == 1.0) {
                classCount[i]++;
                break;
            }
        }
    }
    
    // Find minimum class count
    ll minCount = 1e9;
    for(ll count : classCount) {
        if(count < minCount) minCount = count;
    }
    
    // Create balanced dataset
    std::vector<Tensor*> X_new, Y_new;
    std::vector<ll> classTaken(sz(classes), 0);
    
    for(ll i = 0; i < sz(X); i++) {
        for(ll j = 0; j < sz(classes); j++) {
            if(Y[i]->arr[0][0][0][j] == 1.0) {
                if(classTaken[j] < minCount) {
                    X_new.push_back(X[i]);
                    Y_new.push_back(Y[i]);
                    classTaken[j]++;
                }
                break;
            }
        }
    }

    X = X_new;
    Y = Y_new;

    std::cout<<"Undersampling completed. Dataset size: "<<sz(X)<<std::endl;
}


void train_test_split(const std::vector<Tensor*> &X, const std::vector<Tensor*> &Y, 
                            std::vector<Tensor*> &X_train, std::vector<Tensor*> &Y_train, 
                            std::vector<Tensor*> &X_test, std::vector<Tensor*> &Y_test, 
                            const double test_ratio) {
    ll train_size = sz(X) * (1 - test_ratio);
    
    for(ll i=0;i<train_size;i++) {
        X_train.push_back(X[i]);
        Y_train.push_back(Y[i]);
    }

    for(ll i=train_size;i<sz(X);i++) {
        X_test.push_back(X[i]);
        Y_test.push_back(Y[i]);
    }
}

void shuffle_dataset(std::vector<Tensor*> &X, std::vector<Tensor*> &Y) {
    std::vector<ll> indices(sz(X));
    for(ll i=0;i<sz(X);i++) indices[i] = i;

    std::random_shuffle(indices.begin(), indices.end());

    std::vector<Tensor*> X_shuffled, Y_shuffled;
    for(ll i=0;i<sz(X);i++) {
        X_shuffled.push_back(X[indices[i]]);
        Y_shuffled.push_back(Y[indices[i]]);
    }

    X = X_shuffled;
    Y = Y_shuffled;
}