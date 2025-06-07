#ifndef DATALOADER_H
#define DATALODAER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include "../nn/nn.h"

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

void load_dataset(const ll maxLength, const std::str path, std::vector<Tensor*> &X, std::vector<Tensor*> &Y, std::vector<std::str> &classes);
void undersampling(std::vector<Tensor*> &X, std::vector<Tensor*> &Y, const std::vector<std::str> &classes);
void shuffle_dataset(std::vector<Tensor*> &X, std::vector<Tensor*> &Y);
void train_test_split(const std::vector<Tensor*> &X, const std::vector<Tensor*> &Y, 
                            std::vector<Tensor*> &X_train, std::vector<Tensor*> &Y_train, 
                            std::vector<Tensor*> &X_test, std::vector<Tensor*> &Y_test, 
                            const double train_ratio);

#endif // DATALOADER_H