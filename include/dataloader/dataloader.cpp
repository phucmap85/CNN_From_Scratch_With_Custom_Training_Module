#include "dataloader.h"

std::str to_str(ll a) {
    std::str s = "";

    if(a == 0) return "0";

    while(a > 0) {
        s = (char) ((a % 10) + '0') + s;
        a /= 10;
    }

    return s;
}

void load_dataset(const ll maxLength, const std::str path, std::vector<Image*> &X, std::vector<ll> &Y, std::vector<std::str> &classes) {
    std::ifstream trainLabels((path + "/trainLabels.csv").c_str());

    if(!trainLabels.is_open()) {
        std::cout << "Error opening trainLabels.csv.";
        exit(1);
    }

    std::str s;
    ll id = 0, temp = 0;
    std::map<std::str, ll> hashMap;
    Image *img;

    getline(trainLabels, s);
    while(getline(trainLabels, s) && id < maxLength) {
        std::str label = s.substr(s.find(",") + 1);

        // std::cout<<id + 1<<" "<<label<<" "<<sz(X)<<" "<<sz(Y)<<" "<<sz(classes)<<"\n";

        if(!hashMap[label]) {
            hashMap[label] = ++temp;
            classes.push_back(label);
        }

        img = new Image(path + "/" + to_str(++id) + ".png");

        X.push_back(img);
        Y.push_back(hashMap[label] - 1);
    }

    trainLabels.close();
}
