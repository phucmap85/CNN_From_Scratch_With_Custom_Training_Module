#include <cstdlib>
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
vector<Tensor*> Y;
vector<str> classes;

vector<Tensor*> X_train, X_test;
vector<Tensor*> Y_train, Y_test;

db accuracy(vector<ll> &y_true, vector<ll> &y_pred) {
	if(sz(y_true) != sz(y_pred)) {
		std::cerr<<"Error: y_true and y_pred must have the same size."<<std::endl;
		exit(1);
	}

	ll correct = 0;
	for(ll i=0;i<sz(y_true);i++) correct += (y_true[i] == y_pred[i]);

	return (db) ((db) correct / (db) sz(y_true));
}

signed main() {
	// Load dataset
	load_dataset(50000, "cifar", X, Y, classes);

	train_test_split(X, Y, X_train, Y_train, X_test, Y_test, 0.2);

	undersampling(X_train, Y_train, classes);
	undersampling(X_test, Y_test, classes);

	cout<<"Training set size: "<<sz(X_train)<<endl;
	cout<<"Testing set size: "<<sz(X_test)<<endl;

	// Create the model
	Model *model = new Model();
	model->read_model_config("model.conf");
	model->summary();
	if(!model->load_model_from_file("best.nn")) {
		cout<<"No pre-trained model found, starting from scratch."<<endl;
	} else {
		cout<<"Loaded pre-trained model successfully."<<endl;
	}

	// Train the model
	db best = 0;
	for (ll epoch = 0; epoch < model->epochs; epoch++) {
		shuffle_dataset(X_train, Y_train);
		
		db train_total_loss = 0;
		
		cout << "Epoch " << epoch+1 << "/" << model->epochs << ":\n";
		
		// Training loop with batches
		for (ll batch_start = 0; batch_start < sz(X_train); batch_start += model->batch_size) {
			// Determine actual batch size (might be smaller at the end)
			ll actual_batch_size = min(model->batch_size, sz(X_train) - batch_start);
			
			// Accumulate gradients over the batch
			model->reset_backward_per_batch();
			db batch_loss = 0;
			
			for (ll i = 0; i < actual_batch_size; i++) {
				ll idx = batch_start + i;
				
				// Forward pass
				model->reset_forward();
				model->forward(X_train[idx]);
				
				// Compute loss and backward pass (accumulate gradients)
				model->reset_backward_per_datapoint();
				batch_loss += model->loss->calculate(Y_train[idx], model->layers[sz(model->layers)-1]->dense->a);
				model->backward();
			}
			
			// Update weights once per batch
			model->gradient_descent();
			
			// Track total loss
			train_total_loss += batch_loss;
			
			// Print progress
			cout << fixed << setprecision(6);
			ll width = to_string(sz(X_train) / model->batch_size + 1).length();
			cout << "\rBatch: " << setw(width) << (batch_start / model->batch_size + 1)
				 << "/" << (sz(X_train) / model->batch_size + 1)
				 << " - batch_loss: " << (batch_loss / actual_batch_size);
			fflush(stdout);
		}
		
		// Evaluate on training set for accuracy
		vector<ll> train_Y_true, train_Y_pred;
		for (ll i = 0; i < sz(X_train); i++) {
			model->reset_forward();
			model->forward(X_train[i]);

			train_Y_true.push_back(argmax(Y_train[i]));
			train_Y_pred.push_back(argmax(model->layers[sz(model->layers)-1]->dense->a));
		}
		db train_accuracy = accuracy(train_Y_true, train_Y_pred);
		
		// Evaluate on validation set after each epoch
		db valid_total_loss = 0;
		vector<ll> valid_Y_true, valid_Y_pred;
		
		for (ll j = 0; j < sz(X_test); j++) {
			model->reset_forward();
			model->forward(X_test[j]);
			
			valid_total_loss += model->loss->calculate(Y_test[j], model->layers[sz(model->layers)-1]->dense->a);
			
			valid_Y_true.push_back(argmax(Y_test[j]));
			valid_Y_pred.push_back(argmax(model->layers[sz(model->layers)-1]->dense->a));
		}
		
		db train_loss = train_total_loss / sz(X_train);
		db valid_loss = valid_total_loss / sz(X_test);
		db valid_accuracy = accuracy(valid_Y_true, valid_Y_pred);
		
		// Save the model if validation accuracy improves
		if (valid_accuracy > best) {
			best = valid_accuracy;
			model->save_model_to_file("best.nn");
		}
		model->save_model_to_file("last.nn");
		
		// Print epoch results
		cout << "\nEpoch " << epoch+1 << " completed"
			 << " - loss: " << train_loss
			 << " - acc: " << train_accuracy
			 << " - val_loss: " << valid_loss
			 << " - val_acc: " << valid_accuracy
			 << " - best: " << best << endl;
	}

	system("pause");
}