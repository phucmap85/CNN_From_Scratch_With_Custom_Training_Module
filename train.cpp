#include <bits/stdc++.h>
#include "include/nn/nn.h"
#include "include/lodepng/lodepng.h"
#include "include/dataloader/dataloader.h"
using namespace std;

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

// Input and Output
vector<Tensor*> X;
vector<ll> Y;
vector<str> classes;

signed main() {
	Model *model = new Model();
	model->read_model_config("model.conf");
	model->summary();

	load_dataset(100, "cifar", X, Y, classes);

	model->forward(X[0]);

	model->layers[sz(model->layers) - 1]->dense->y->print();
	model->layers[sz(model->layers) - 1]->dense->a->print();
}