#include "dataloader.h"

void load_dataset(const ll maxLength, const std::str path, std::vector<Tensor *> &X, std::vector<ll> &Y, std::vector<std::str> &classes) {
    std::ifstream trainLabels((path + "/trainLabels.csv").c_str());

    if(!trainLabels.is_open()) {
        std::cout<<"Error opening trainLabels.csv.";
        exit(1);
    }

    std::str s;
    ll cntClasses = 0;
    std::map<std::str, ll> hashMap;

    getline(trainLabels, s);
    while(getline(trainLabels, s) && sz(X) < maxLength) {
    	std::str filename = s.substr(0, s.find(",")) + ".png";
        std::str label = s.substr(sz(filename) - 3);

        if(!hashMap[label]) {
            hashMap[label] = ++cntClasses;
            classes.push_back(label);
        }

        Tensor *img = new Tensor(path + "/" + filename);

        X.push_back(img);
        Y.push_back(hashMap[label] - 1);
        
        // std::cout<<filename<<" "<<label<<" "<<sz(X)<<" "<<sz(Y)<<" "<<sz(classes)<<"\n";
    }

    trainLabels.close();
}