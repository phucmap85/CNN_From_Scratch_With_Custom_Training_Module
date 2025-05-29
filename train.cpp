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

// Model configuration
ll epochs;
ll batch_size;
db rate = 1;
db lr;
db best;
str loss_function;
vector<Layer*> sequential;

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

// db activation_fn(db x, ll layer, ll node) {
//     if(activation_per_layer[layer] == "relu") return relu(x);
//     else if(activation_per_layer[layer] == "sigmoid") return sigmoid(x);
//     else if(activation_per_layer[layer] == "tanh") return tanh(x);
//     else if(activation_per_layer[layer] == "softmax") return softmax(x, layer, node);
//     else return linear(x);
// }
// db activation_fn_dx(db x, ll layer, ll node) {
//     if(activation_per_layer[layer] == "relu") return relu_dx(x);
//     else if(activation_per_layer[layer] == "sigmoid") return sigmoid_dx(x);
//     else if(activation_per_layer[layer] == "tanh") return tanh_dx(x);
//     else return linear_dx(x);
// }

void read_model_config(const string filename) {
	str s;
	ifstream ModelConfiguration(filename.c_str());
	if(!ModelConfiguration.is_open()) {
		cout<<"Error opening file "<<filename<<endl;
		return;
	}

	// cout<<"Model Configuration:"<<endl;
	// cout<<"======================"<<endl;

	while(getline(ModelConfiguration, s)) {
		str command = s.substr(0, s.find(" "));

		if(command == "conv") {
			ll filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, input_layer = 0;
			char _activation[15], trash[15];

			sscanf(s.c_str(), "%s %d (%d, %d) (%d, %d) (%d, %d) %s %d", trash, &filter, &kernel_h, &kernel_w, 
				&stride_h, &stride_w, &pad_h, &pad_w, _activation, &input_layer);

			str activation = str(_activation);

			Layer *conv = new Layer(filter, kernel_h, kernel_w, stride_h, stride_w, 
				pad_h, pad_w, activation, input_layer);

			// cout<<"Convolution Layer:"<<endl;
			// cout<<"  Filters: "<<conv->conv->filter<<endl;
			// cout<<"  Kernel Size: ("<<conv->conv->kernel_h<<", "<<conv->conv->kernel_w<<", "<<conv->conv->kernel_d<<")"<<endl;
			// cout<<"  Stride: ("<<conv->conv->stride_h<<", "<<conv->conv->stride_w<<")"<<endl;
			// cout<<"  Padding: ("<<conv->conv->pad_h<<", "<<conv->conv->pad_w<<")"<<endl;
			// cout<<"  Activation: "<<conv->conv->activation<<endl;

			sequential.push_back(conv);
		}
		else if(command == "maxpool" || command == "avgpool") {
			ll pool_h, pool_w;
			char _type[15];

			sscanf(s.c_str(), "%s (%d, %d)", _type, &pool_h, &pool_w);

			str type = str(_type);

			Layer *pooling = new Layer(pool_h, pool_w, type);

			// cout<<"Pooling Layer:"<<endl;
			// cout<<"  Pooling Size: ("<<pooling->pooling->pool_h<<", "<<pooling->pooling->pool_w<<")"<<endl;
			// cout<<"  Pooling Type: "<<command<<endl;

			sequential.push_back(pooling);
		}
		else if(command == "dense") {
			ll nodes; 
			char _activation[15], trash[15];

			sscanf(s.c_str(), "%s %d %s", trash, &nodes, _activation);

			str activation = str(_activation);

			Layer *dense = new Layer(nodes, activation);

			// cout<<"Dense Layer:"<<endl;
			// cout<<"  Nodes: "<<nodes<<endl;
			// cout<<"  Activation: "<<activation<<endl;

			sequential.push_back(dense);
		}
		else if(command == "flatten") {
			Layer *flatten = new Layer();

			// cout<<"Flatten Layer:"<<endl;

			sequential.push_back(flatten);
		}
		else if(command == "compile") {
			char _loss_function[15], trash[15];

			sscanf(s.c_str(), "%s %s %d %d %lf", trash, _loss_function, &epochs, &batch_size, &lr);

			loss_function = str(_loss_function);

			// cout<<"Compile:"<<endl;
			// cout<<"  Loss Function: "<<loss_function<<endl;
			// cout<<"  Epochs: "<<epochs<<endl;
			// cout<<"  Batch Size: "<<batch_size<<endl;
			// cout<<"  Learning Rate: "<<lr<<endl;
		}
	}

	ModelConfiguration.close();
}

signed main() {
	srand(time(NULL));
	cout<<fixed<<setprecision(9);

	read_model_config("model.conf");

	initialize_weights(sequential);

	str path = "cifar";

	load_dataset(100, path, X, Y, classes);

	convolution(X[0], sequential[0]->conv);

	// print 5 images and their labels
	// for(ll i=0;i<5 && i<sz(X);i++) {
	// 	cout<<"Image "<<i+1<<":\n";

	// 	X[i]->export_to_file("test_image/output_" + to_str(i) + ".png");

	// 	for(ll c=0;c<X[i]->depth;c++) {
	// 		cout<<"Channel "<<c<<":\n";
	// 		for(ll h=0;h<X[i]->height;h++) {
	// 			for(ll w=0;w<X[i]->width;w++) {
	// 				cout<<X[i]->arr[c][h][w]<<" ";
	// 			}
	// 			cout<<"\n";
	// 		}
	// 		cout<<"\n";
	// 	}

	// 	cout<<"Label: "<<classes[Y[i]]<<"\n\n";
	// }
}
