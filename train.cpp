#include <bits/stdc++.h>
#include "include/nn/nn.h"
#include "include/utils/utils.h"
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

// db softmax(db x, ll layer, ll node) {
//     db total_exp = 0;
//     for(ll cnode=0;cnode<nodes_per_layer[layer];cnode++) {
//         total_exp += exp(Y[layer][cnode]);
//     }
//     return exp(x) / total_exp;
// }

signed main() {
	srand(time(NULL));

	Model *sequential = new Model();
	sequential->read_model_config("model.conf");

	sequential->summary();

	load_dataset(100, "cifar", X, Y, classes);

	convolution(X[0], sequential->layers[0]->conv);

	pooling(sequential->layers[0]->conv->a, sequential->layers[1]->pooling);

	convolution(sequential->layers[1]->pooling->a, sequential->layers[2]->conv);
}
