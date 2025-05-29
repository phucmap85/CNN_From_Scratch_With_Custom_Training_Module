#ifndef DATALOADER_H
#define DATALODAER_H

#include <bits/stdc++.h>
#include "../nn/nn.h"
#include "../utils/utils.h"

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

void load_dataset(const ll maxLength, const std::str path, std::vector<Tensor *> &X, std::vector<ll> &Y, std::vector<std::str> &classes);

#endif // DATALOADER_H