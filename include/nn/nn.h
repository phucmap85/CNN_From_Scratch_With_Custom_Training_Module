#ifndef NN_H
#define NN_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <math.h>
#include "../lodepng/lodepng.h"

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

db r2(const db size);

struct Tensor {
	ll filter, height, width, depth; // Assuming RGB Tensors
	db ****arr;

	Tensor(const ll _filter, const ll _height, const ll _width, const ll _depth) : 
		filter(_filter), height(_height), width(_width), depth(_depth) {
		arr = new db***[filter + 5];

		for(ll f=0;f<filter;f++) {
			arr[f] = new db**[depth + 5];
			for(ll c=0;c<depth;c++) {
				arr[f][c] = new db*[height + 5];
				for(ll h=0;h<height;h++) arr[f][c][h] = new db[width + 5];
			}
		}

		for(ll f=0;f<filter;f++) {
			for(ll c=0;c<depth;c++) {
				for(ll h=0;h<height;h++) {
					for(ll w=0;w<width;w++) arr[f][c][h][w] = r2(filter * depth * height * width); // Initialize randomly
				}
			}
		}
	}

	Tensor(const std::str filename) {
		unsigned _width, _height;
		std::vector<unsigned char> pixels;

		unsigned error = lodepng::decode(pixels, _width, _height, filename.c_str());
		if(error) {
			std::cout<<"decoder error "<<error<<": "<<lodepng_error_text(error)<<std::endl;
			exit(1);
		}

		filter = 1; width = _width; height = _height; depth = 3;

		arr = new db***[filter + 5];

		for(ll f=0;f<filter;f++) {
			arr[f] = new db**[depth + 5];
			for(ll c=0;c<depth;c++) {
				arr[f][c] = new db*[height + 5];
				for(ll h=0;h<height;h++) arr[f][c][h] = new db[width + 5];
			}
		}

		for(ll f=0;f<filter;f++) {
			for(ll h=0;h<height;h++) {
				for(ll w=0;w<width;w++) {
					for(ll c=0;c<depth;c++) { // RGB channels
						arr[f][c][h][w] = pixels[(h * height + w) * 4 + c] / 255.0;
					}
				}
			}
		}
	}

	Tensor(const db value) {
		filter = 1; height = 1; width = 1; depth = 1;
		arr = new db***[filter + 5];
		arr[0] = new db**[depth + 5];
		arr[0][0] = new db*[height + 5];
		arr[0][0][0] = new db[width + 5];
		arr[0][0][0][0] = value;
	}

	void fill(const db value) {
		for(ll f=0;f<filter;f++) {
			for(ll c=0;c<depth;c++) {
				for(ll h=0;h<height;h++) {
					for(ll w=0;w<width;w++) arr[f][c][h][w] = value;
				}
			}
		}
	}

	db get_max() {
		db max_val = -1e18;
		for(ll f=0;f<filter;f++) {
			for(ll c=0;c<depth;c++) {
				for(ll h=0;h<height;h++) {
					for(ll w=0;w<width;w++) max_val = std::max(max_val, arr[f][c][h][w]);
				}
			}
		}
		return max_val;
	}

	db get_min() {
		db min_val = 1e18;
		for(ll f=0;f<filter;f++) {
			for(ll c=0;c<depth;c++) {
				for(ll h=0;h<height;h++) {
					for(ll w=0;w<width;w++) min_val = std::min(min_val, arr[f][c][h][w]);
				}
			}
		}
		return min_val;
	}

	void print() {
		std::cout << "Tensor(" << filter << ", " << depth << ", " << height << ", " << width << ")\n" << std::fixed << std::setprecision(6);
		
		// For large tensors, show partial content
		if (filter * depth * height * width > 1000) {
			std::cout << "array([";
			// Number of items to show at beginning and end
			const int show_items = 3;
			
			for (ll f = 0; f < filter; f++) {
				if (f > 0) std::cout << ",\n       ";
				std::cout << "[";
				for (ll c = 0; c < depth; c++) {
					if (c > 0) std::cout << ",\n        ";
					std::cout << "[";
					for (ll h = 0; h < height; h++) {
						if (h > 0) std::cout << ",\n         ";
						std::cout << "[";
						
						// Show first few values
						for (ll w = 0; w < std::min(show_items, width); w++) {
							std::cout << std::setw(9) << arr[f][c][h][w];
							if (w < std::min(show_items, width) - 1) std::cout << ", ";
						}
						
						// Add ellipsis if needed
						if (width > 2 * show_items) {
							std::cout << ", ..., ";
							
							// Show last few values
							for (ll w = std::max(show_items, width - show_items); w < width; w++) {
								std::cout << std::setw(9) << arr[f][c][h][w];
								if (w < width - 1) std::cout << ", ";
							}
						}
						std::cout << "]";
					}
					std::cout << "]";
				}
				std::cout << "]";
			}
			std::cout << "])" << std::endl;
			return;
		}
		
		std::cout << "array([";
		for (ll f = 0; f < filter; f++) {
			if (f > 0) std::cout << ",\n       ";
			std::cout << "[";
			for (ll c = 0; c < depth; c++) {
				if (c > 0) std::cout << ",\n        ";
				std::cout << "[";
				for (ll h = 0; h < height; h++) {
					if (h > 0) std::cout << ",\n         ";
					std::cout << "[";
					for (ll w = 0; w < width; w++)
						std::cout << std::setw(9) << arr[f][c][h][w] << (w < width - 1 ? ", " : "");
					std::cout << "]";
				}
				std::cout << "]";
			}
			std::cout << "]";
		}
		std::cout << "])" << std::endl;
	}

	void export_to_file(const std::str filename, const ll f = 0) {
		if(f < 0 || f >= filter) {
			std::cerr<<"Invalid filter index: "<<f<<". Valid range is [0, "<<filter-1<<"]."<<std::endl;
			return;
		}

		std::vector<unsigned char> pixels;
		pixels.resize(width * height * 4);

		// Find min and max values in the tensor for normalization
		db min_val = 1e18, max_val = -1e18;
		
		for(ll h=0; h<height; h++) {
			for(ll w=0; w<width; w++) {
				for(ll c=0; c<depth; c++) {
					min_val = std::min(min_val, arr[f][c][h][w]);
					max_val = std::max(max_val, arr[f][c][h][w]);
				}
			}
		}
		
		// Avoid division by zero if all values are the same
		db range = max_val - min_val;
		if(range == 0) range = 1;

		for(ll h=0;h<height;h++) {
			for(ll w=0;w<width;w++) {
				for(ll c=0;c<depth;c++) { // RGB channels
					// Normalize to [0,1] based on min/max values
					db normalized = (arr[f][c][h][w] - min_val) / range;
					pixels[(h * width + w) * 4 + c] = (unsigned char) (normalized * 255.0);
				}
				pixels[(h * width + w) * 4 + 3] = 255; // Alpha channel
			}
		}

		unsigned error = lodepng::encode(filename, pixels, width, height);
		
		if(error) {
			std::cout<<"encoder error "<<error<<": "<<lodepng_error_text(error)<<std::endl;
			exit(1);
		}
	}

	~Tensor() {
		if(arr) {
			for(ll f=0;f<filter;f++) {
				for(ll c=0;c<depth;c++) {
					for(ll h=0;h<height;h++) delete[] arr[f][c][h];
					delete[] arr[f][c];
				}
				delete[] arr[f];
			}
			delete[] arr;
		}
	}
};

struct Input {
	ll id, height, width, depth;
	Tensor *a, *dA;

	Input(ll _id, ll _height, ll _width, ll _depth) : id(_id), height(_height), width(_width), depth(_depth) {
		a = new Tensor(1, height, width, depth);
		dA = new Tensor(1, height, width, depth);
	}

	~Input() {
		if(a) delete a;
		if(dA) delete dA;
	}
};

struct Conv {
	ll id;
	ll filter;
	ll stride_h, stride_w;
	ll pad_h, pad_w;
	ll input_layer;
	std::str activation;

	Tensor *kernel, *b, *y, *a;
	
	// For forward and backward pass
	ll batch_cnt = 0;
	Tensor *dA, *dW, *dB;

	// For optimizer
	ll timestep = 0;
	Tensor *m_b, *v_b, *m_w, *v_w;
	
	Conv(ll _id, ll _filter, ll _kernel_h, ll _kernel_w, ll _stride_h, ll _stride_w, ll _pad_h, ll _pad_w, std::str _activation) : 
		id(_id), filter(_filter),
		stride_h(_stride_h), stride_w(_stride_w),
		pad_h(_pad_h), pad_w(_pad_w),
		activation(_activation) {}

	~Conv() {
		if(kernel) delete kernel;
		if(a) delete a;
		if(y) delete y;
		if(b) delete b;
		if(dA) delete dA;
		if(dW) delete dW;
		if(dB) delete dB;

		if(m_b) delete m_b;
		if(v_b) delete v_b;
		if(m_w) delete m_w;
		if(v_w) delete v_w;
	}
};

struct Pooling {
	ll id;
	std::str type;
	ll pool_h, pool_w;
	ll stride_h, stride_w;
	ll pad_h, pad_w;

	Tensor *a, *dA;
	
	Pooling(ll _id, ll _pool_h, ll _pool_w, ll _stride_h, ll _stride_w, ll _pad_h, ll _pad_w, std::str _type) : 
		id(_id),
		pool_h(_pool_h), pool_w(_pool_w),
		stride_h(_stride_h), stride_w(_stride_w),
		pad_h(_pad_h), pad_w(_pad_w),
		type(_type) {}

	~Pooling() {
		if(a) delete a;
		if(dA) delete dA;
	}
};

struct Flatten {
	ll id;
	ll units;
	Tensor *a, *dA;

	Flatten(ll _id) : id(_id), units(0) {}

	~Flatten() {
		if(a) delete a;
		if(dA) delete dA;
	}
};

struct Dense {
	ll id;
	ll units;
	std::str activation;

	Tensor *a, *y, *w, *b;

	// For forward and backward pass
	ll batch_cnt = 0;
	Tensor *dA, *dW, *dB;

	// For optimizer
	ll timestep = 0;
	Tensor *m_b, *v_b, *m_w, *v_w;
	
	Dense(ll _id, ll _units, std::str _activation) : id(_id), units(_units), activation(_activation) {
		a = new Tensor(1, 1, units, 1);
		y = new Tensor(1, 1, units, 1);
		b = new Tensor(1, 1, units, 1);
		b->fill(0.0); // Initialize bias to zero

		dA = new Tensor(1, 1, units, 1);
		dB = new Tensor(1, 1, units, 1);

		m_b = new Tensor(1, 1, units, 1); // Momentum for bias
		v_b = new Tensor(1, 1, units, 1); // Velocity for bias
		m_b->fill(0.0);
		v_b->fill(0.0);
	}

	~Dense() {
		if(w) delete w;
		if(a) delete a;
		if(y) delete y;
		if(b) delete b;

		if(dA) delete dA;
		if(dW) delete dW;
		if(dB) delete dB;

		if(m_b) delete m_b;
		if(v_b) delete v_b;
		if(m_w) delete m_w;
		if(v_w) delete v_w;
	}
};

struct Layer {
	Input *input;
	Conv *conv;
	Pooling *pooling;
	Flatten *flatten;
	Dense *dense;

	Layer(ll id, ll height, ll width, ll depth) {
		input = new Input(id, height, width, depth);
		conv = NULL;
		pooling = NULL;
		flatten = NULL;
		dense = NULL;
	}
	
	Layer(ll id, ll filter, ll kernel_h, ll kernel_w, ll stride_h, ll stride_w, ll pad_h, ll pad_w, std::str activation) {
		conv = new Conv(id, filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation);
		pooling = NULL;
		flatten = NULL;
		dense = NULL;
		input = NULL;
	}
	
	Layer(ll id, ll pool_h, ll pool_w, ll stride_h, ll stride_w, ll pad_h, ll pad_w, std::str type) {
		pooling = new Pooling(id, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, type);
		conv = NULL;
		flatten = NULL;
		dense = NULL;
		input = NULL;
	}
	
	Layer(ll id) {
		flatten = new Flatten(id);
		conv = NULL;
		pooling = NULL;
		dense = NULL;
		input = NULL;
	}
	
	Layer(ll id, ll units, std::str activation) {
		dense = new Dense(id, units, activation);
		conv = NULL;
		pooling = NULL;
		flatten = NULL;
		input = NULL;
	}

	~Layer() {
		if(input) delete input;
		if(conv) delete conv;
		if(pooling) delete pooling;
		if(flatten) delete flatten;
		if(dense) delete dense;
	}
};

struct Loss {
	std::str name;
	db value = 0;
	Tensor *grad = NULL;

	Loss(const std::str &name) : name(name) {}

	db calculate(const Tensor *y_true, const Tensor *y_pred) {
		if(name == "mse") {
			db sum = 0.0;
			for(ll f=0;f<y_true->filter;f++) {
				for(ll h=0;h<y_true->height;h++) {
					for(ll w=0;w<y_true->width;w++) {
						sum += (y_true->arr[f][0][h][w] - y_pred->arr[f][0][h][w]) * (y_true->arr[f][0][h][w] - y_pred->arr[f][0][h][w]);
					}
				}
			}
			value = sum / y_true->filter;
			
			// Compute gradient for MSE
			if(grad) delete grad;
			grad = new Tensor(y_true->filter, y_true->height, y_true->width, 1);
			for(ll f=0;f<y_true->filter;f++) {
				for(ll h=0;h<y_true->height;h++) {
					for(ll w=0;w<y_true->width;w++) {
						grad->arr[f][0][h][w] = 2 * (y_pred->arr[f][0][h][w] - y_true->arr[f][0][h][w]);
					}
				}
			}
			
			return value;
		} else if(name == "binary_crossentropy") {
			db sum = 0.0;
			for(ll f=0;f<y_true->filter;f++) {
				for(ll h=0;h<y_true->height;h++) {
					for(ll w=0;w<y_true->width;w++) {
						db y_t = y_true->arr[f][0][h][w];
						db y_p = y_pred->arr[f][0][h][w];
						sum += y_t * log(y_p + 1e-15) + (1 - y_t) * log(1 - y_p + 1e-15);
					}
				}
			}
			value = -sum / y_true->filter;
			
			// Compute gradient for binary crossentropy
			if(grad) delete grad;
			grad = new Tensor(y_true->filter, y_true->height, y_true->width, 1);
			for(ll f=0;f<y_true->filter;f++) {
				for(ll h=0;h<y_true->height;h++) {
					for(ll w=0;w<y_true->width;w++) {
						db y_t = y_true->arr[f][0][h][w];
						db y_p = y_pred->arr[f][0][h][w];
						grad->arr[f][0][h][w] = -(y_t / (y_p + 1e-15)) + ((1 - y_t) / (1 - y_p + 1e-15));
					}
				}
			}
			
			return value;
		} else if(name == "categorical_crossentropy") {
			db sum = 0.0;
			for(ll f=0;f<y_true->filter;f++) {
				for(ll h=0;h<y_true->height;h++) {
					for(ll w=0;w<y_true->width;w++) {
						db y_t = y_true->arr[f][0][h][w];
						db y_p = y_pred->arr[f][0][h][w];
						sum += y_t * log(y_p + 1e-15);
					}
				}
			}
			value = -sum / y_true->filter;
			
			// Compute gradient for categorical crossentropy
			if(grad) delete grad;
			grad = new Tensor(y_true->filter, y_true->height, y_true->width, 1);
			for(ll f=0;f<y_true->filter;f++) {
				for(ll h=0;h<y_true->height;h++) {
					for(ll w=0;w<y_true->width;w++) {
						db y_t = y_true->arr[f][0][h][w];
						db y_p = y_pred->arr[f][0][h][w];
						grad->arr[f][0][h][w] = y_p - y_t;
					}
				}
			}

			// grad->print(); // Print gradient for debugging
			
			return value;
		} else {
			std::cerr<<"Unknown loss function: "<<name<<std::endl;
			exit(1);
		}
	}

	~Loss() {
		if(grad) delete grad;
	}
};

struct Optimizer {
	std::str name;
	db lr;

	Optimizer(const std::str &_name, db _lr) : name(_name), lr(_lr) {
		if(name != "adam" && name != "sgd" && name != "rmsprop" && name != "adagrad" && name != "adadelta") {
			std::cerr<<"Error: Unknown optimizer: "<<name<<std::endl;
			exit(1);
		}
	}

	void apply_gradient(const ll timestep, const ll f, const ll d, const ll h, const ll w, 
						Tensor *m, Tensor *v, Tensor *input, Tensor *grad) {
		if(name == "adam") {
			const db beta1 = 0.9;
			const db beta2 = 0.999;
			const db epsilon = 1e-8;

			if(lr <= 0) {
				// Set default learning rate if not provided
				std::cerr<<"Warning: Learning rate not set for Adam, using default value of 0.001."<<std::endl;
				lr = 0.001;
			}

			m->arr[f][d][h][w] = beta1 * m->arr[f][d][h][w] + (1 - beta1) * grad->arr[f][d][h][w];
			v->arr[f][d][h][w] = beta2 * v->arr[f][d][h][w] + (1 - beta2) * pow(grad->arr[f][d][h][w], 2);

			db m_hat = m->arr[f][d][h][w] / (1 - pow(beta1, timestep));
			db v_hat = v->arr[f][d][h][w] / (1 - pow(beta2, timestep));
			
			input->arr[f][d][h][w] -= lr * m_hat / (sqrt(v_hat) + epsilon);
		} else if(name == "sgd") {
			const db momentum = 0.9;

			if(lr <= 0) {
				// Set default learning rate if not provided
				std::cerr<<"Warning: Learning rate not set for SGD, using default value of 0.01."<<std::endl;
				lr = 0.01;
			}
			
			v->arr[f][d][h][w] = momentum * v->arr[f][d][h][w] + (1 - momentum) * grad->arr[f][d][h][w];

			input->arr[f][d][h][w] -= lr * v->arr[f][d][h][w];
		} else if(name == "rmsprop") {
			const db decay_rate = 0.99;
			const db epsilon = 1e-8;

			if(lr <= 0) {
				// Set default learning rate if not provided
				std::cerr<<"Warning: Learning rate not set for RMSProp, using default value of 0.001."<<std::endl;
				lr = 0.001;
			}

			v->arr[f][d][h][w] = decay_rate * v->arr[f][d][h][w] + (1 - decay_rate) * pow(grad->arr[f][d][h][w], 2);

			input->arr[f][d][h][w] -= lr * grad->arr[f][d][h][w] / (sqrt(v->arr[f][d][h][w]) + epsilon);
		} else if(name == "adagrad") {
			const db epsilon = 1e-8;

			if(lr <= 0) {
				// Set default learning rate if not provided
				std::cerr<<"Warning: Learning rate not set for Adagrad, using default value of 0.001."<<std::endl;
				lr = 0.001;
			}

			v->arr[f][d][h][w] += pow(grad->arr[f][d][h][w], 2);
			
			input->arr[f][d][h][w] -= lr * grad->arr[f][d][h][w] / (sqrt(v->arr[f][d][h][w]) + epsilon);
		} else if(name == "adadelta") {
			const db decay_rate = 0.95;
			const db epsilon = 1e-8;

			if(lr <= 0) {
				// Set default learning rate if not provided
				std::cerr<<"Warning: Learning rate not set for Adadelta, using default value of 1.0."<<std::endl;
				lr = 1.0;
			}

			// Assuming m is the accumulated squared gradients (E[g^2])
			// Assuming v is the accumulated updates (E[delta_theta^2])

			m->arr[f][d][h][w] = decay_rate * m->arr[f][d][h][w] + (1 - decay_rate) * pow(grad->arr[f][d][h][w], 2);
			
			db rms_g = sqrt(m->arr[f][d][h][w] + epsilon);
			db rms_delta_theta = sqrt(v->arr[f][d][h][w] + epsilon);
			db delta_theta = - lr * (rms_delta_theta / (rms_g + epsilon)) * grad->arr[f][d][h][w];

			v->arr[f][d][h][w] = decay_rate * v->arr[f][d][h][w] + (1 - decay_rate) * pow(delta_theta, 2);

			input->arr[f][d][h][w] += delta_theta;
		} else {
			std::cerr<<"Error: Unknown optimizer: "<<name<<std::endl;
			exit(1);
		}
	}
};

struct Model {
	ll epochs;
	ll batch_size;
	Optimizer *optimizer;
	Loss *loss;
	std::vector<Layer*> layers;

	Model() {}

	void read_model_config(const std::str filename) {
		std::str s;
		std::ifstream ModelConfiguration(filename.c_str());
		if(!ModelConfiguration.is_open()) {
			std::cout<<"Error opening file "<<filename<<std::endl;
			return;
		}

		// std::cout<<"Model Configuration:"<<std::endl;
		// std::cout<<"======================"<<std::endl;

		while(getline(ModelConfiguration, s)) {
			std::str command = s.substr(0, s.find(" "));

			if(command == "input") {
				ll height, width, depth;
				sscanf(s.c_str(), "%s (%d, %d, %d)", command.c_str(), &height, &width, &depth);

				if(height <= 0 || width <= 0 || depth <= 0) {
					std::cerr<<"Error: Invalid input dimensions."<<std::endl;
					exit(1);
				}

				Layer *input_layer = new Layer(sz(layers), height, width, depth);

				layers.push_back(input_layer);
			}
			else if(command == "conv") {
				ll filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
				char _activation[50], trash[50];

				sscanf(s.c_str(), "%s %d (%d, %d) (%d, %d) (%d, %d) %s", trash, &filter, &kernel_h, &kernel_w, 
					&stride_h, &stride_w, &pad_h, &pad_w, _activation);

				std::str activation = std::str(_activation);

				Layer *conv = new Layer(sz(layers), filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation);

				layers.push_back(conv);
				
				// Initialize the convolution layer's kernel and output dimensions
				ll output_height = 0, output_width = 0, input_d = 1;
				
				ll idx = sz(layers) - 1;
				if(layers[idx - 1]->input) {
					output_height = (layers[idx - 1]->input->height - kernel_h + 2 * pad_h) / stride_h + 1;
					output_width = (layers[idx - 1]->input->width - kernel_w + 2 * pad_w) / stride_w + 1;
					input_d = layers[idx - 1]->input->depth; // Input depth from the input layer
				} 
				else if(layers[idx - 1]->pooling) {
					output_height = (layers[idx - 1]->pooling->a->height - kernel_h + 2 * pad_h) / stride_h + 1;
					output_width = (layers[idx - 1]->pooling->a->width - kernel_w + 2 * pad_w) / stride_w + 1;
					input_d = layers[idx - 1]->pooling->a->depth; // Input depth from the pooling layer
				} 
				else if(layers[idx - 1]->conv) {
					output_height = (layers[idx - 1]->conv->a->height - kernel_h + 2 * pad_h) / stride_h + 1;
					output_width = (layers[idx - 1]->conv->a->width - kernel_w + 2 * pad_w) / stride_w + 1;
					input_d = layers[idx - 1]->conv->a->depth; // Input depth from the convolution layer
				} else {
					std::cerr<<"Error: Previous layer is not a convolution, pooling or input layer."<<std::endl;
					exit(1);
				}

				conv->conv->kernel = new Tensor(filter, kernel_h, kernel_w, input_d);
				conv->conv->dW = new Tensor(filter, kernel_h, kernel_w, input_d);

				conv->conv->b = new Tensor(filter, 1, 1, 1);
				conv->conv->b->fill(0.0); // Initialize biases to zero
				conv->conv->dB = new Tensor(filter, 1, 1, 1);

				conv->conv->y = new Tensor(filter, output_height, output_width, 1);

				conv->conv->a = new Tensor(filter, output_height, output_width, 1);
				conv->conv->dA = new Tensor(filter, output_height, output_width, 1);

				conv->conv->m_b = new Tensor(filter, 1, 1, 1); // Momentum for bias
				conv->conv->v_b = new Tensor(filter, 1, 1, 1); // Velocity for bias
				conv->conv->m_b->fill(0.0);
				conv->conv->v_b->fill(0.0);

				conv->conv->m_w = new Tensor(filter, kernel_h, kernel_w, input_d); // Momentum for weights
				conv->conv->v_w = new Tensor(filter, kernel_h, kernel_w, input_d); // Velocity for weights
				conv->conv->m_w->fill(0.0);
				conv->conv->v_w->fill(0.0);

				// // Print layer details
				// std::cout<<"Convolution Layer:"<<std::endl;
				// std::cout<<"  Id: "<<conv->conv->id<<std::endl;
				// std::cout<<"  Filters: "<<conv->conv->filter<<std::endl;
				// std::cout<<"  Kernel Size: ("<<conv->conv->kernel->height<<", "<<conv->conv->kernel->width<<", "<<conv->conv->kernel->depth<<")"<<std::endl;
				// std::cout<<"  Stride: ("<<conv->conv->stride_h<<", "<<conv->conv->stride_w<<")"<<std::endl;
				// std::cout<<"  Padding: ("<<conv->conv->pad_h<<", "<<conv->conv->pad_w<<")"<<std::endl;
				// std::cout<<"  Activation: "<<conv->conv->activation<<std::endl;
				// std::cout<<"  Input Layer: "<<(input_layer ? "Yes" : "No")<<std::endl;
				// std::cout<<"  Output Shape: ("<<conv->conv->a->filter<<", "<<conv->conv->a->height<<", "<<conv->conv->a->width<<", "<<conv->conv->a->depth<<")"<<std::endl;
			}
			else if(command == "maxpool" || command == "avgpool") {
				ll pool_h, pool_w, stride_h, stride_w, pad_h, pad_w;
				char _type[50];

				sscanf(s.c_str(), "%s (%d, %d) (%d, %d) (%d, %d)", _type, &pool_h, &pool_w, &stride_h, &stride_w, &pad_h, &pad_w);

				std::str type = std::str(_type);

				Layer *pooling = new Layer(sz(layers), pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, type);

				layers.push_back(pooling);

				// Initialize the pooling layer's output dimensions
				ll output_height = 0, output_width = 0;

				ll idx = sz(layers) - 1;
				if(layers[idx - 1]->conv) {
					output_height = (layers[idx - 1]->conv->a->height + 2 * pad_h - pool_h) / stride_h + 1;
					output_width = (layers[idx - 1]->conv->a->width + 2 * pad_w - pool_w) / stride_w + 1;
				} 
				else if(layers[idx - 1]->pooling) {
					output_height = (layers[idx - 1]->pooling->a->height + 2 * pad_h - pool_h) / stride_h + 1;
					output_width = (layers[idx - 1]->pooling->a->width + 2 * pad_w - pool_w) / stride_w + 1;
				} else {
					std::cerr<<"Error: Previous layer is not a convolution or pooling layer."<<std::endl;
					exit(1);
				}

				pooling->pooling->a = new Tensor(layers[idx - 1]->conv->filter, output_height, output_width, 1);
				pooling->pooling->dA = new Tensor(layers[idx - 1]->conv->filter, output_height, output_width, 1);

				// // Print layer details
				// std::cout<<"Pooling Layer:"<<std::endl;
				// std::cout<<"  Id: "<<pooling->pooling->id<<std::endl;
				// std::cout<<"  Pooling Type: "<<command<<std::endl;
				// std::cout<<"  Pooling Size: ("<<pooling->pooling->pool_h<<", "<<pooling->pooling->pool_w<<")"<<std::endl;
				// std::cout<<"  Stride: ("<<pooling->pooling->stride_h<<", "<<pooling->pooling->stride_w<<")"<<std::endl;
				// std::cout<<"  Padding: ("<<pooling->pooling->pad_h<<", "<<pooling->pooling->pad_w<<")"<<std::endl;
				// std::cout<<"  Output Shape: ("<<pooling->pooling->a->filter<<", "<<pooling->pooling->a->height<<", "<<pooling->pooling->a->width<<", "<<pooling->pooling->a->depth<<")"<<std::endl;
			}
			else if(command == "dense") {
				ll units; 
				char _activation[50], trash[50];

				sscanf(s.c_str(), "%s %d %s", trash, &units, _activation);

				std::str activation = std::str(_activation);

				Layer *dense = new Layer(sz(layers), units, activation);

				layers.push_back(dense);

				// Initialize the dense layer's weights and biases
				ll input_size = 0;
				ll idx = sz(layers) - 1;
				Layer *prev = layers[idx - 1];

				if(prev->conv) {
					if(prev->conv->a->filter == 1 && prev->conv->a->depth == 1 && prev->conv->a->height == 1) {
						input_size = prev->conv->a->width;
					} else {
						std::cerr<<"Error: Previous convolution layer output shape does not match dense layer input shape."<<std::endl;
						exit(1);
					}
				}
				else if(prev->pooling) {
					if(prev->pooling->a->filter == 1 && prev->pooling->a->depth == 1 && prev->pooling->a->height == 1) {
						input_size = prev->pooling->a->width;
					} else {
						std::cerr<<"Error: Previous pooling layer output shape does not match dense layer input shape."<<std::endl;
						exit(1);
					}
				}
				else if(prev->input) {
					if(prev->input->a->filter == 1 && prev->input->a->depth == 1 && prev->input->a->height == 1) {
						input_size = prev->input->a->width;
					} else {
						std::cerr<<"Error: Previous input layer output shape does not match dense layer input shape."<<std::endl;
						exit(1);
					}
				}
				else if(prev->flatten) {
					input_size = prev->flatten->units;
				}
				else if(prev->dense) {
					input_size = prev->dense->units;
				}
				else {
					std::cerr<<"Error: Previous layer is not a convolution, pooling, flatten, dense or input layer."<<std::endl;
					exit(1);
				}

				dense->dense->w = new Tensor(units, 1, input_size, 1);
				dense->dense->dW = new Tensor(units, 1, input_size, 1);

				dense->dense->m_w = new Tensor(units, 1, input_size, 1); // Momentum for weights
				dense->dense->v_w = new Tensor(units, 1, input_size, 1); // Velocity for weights
				dense->dense->m_w->fill(0.0);
				dense->dense->v_w->fill(0.0);

				// // Print layer details
				// std::cout<<"Dense Layer:"<<std::endl;
				// std::cout<<"  Id: "<<dense->dense->id<<std::endl;
				// std::cout<<"  Units: "<<units<<std::endl;
				// std::cout<<"  Activation: "<<activation<<std::endl;
			}
			else if(command == "flatten") {
				Layer *flatten = new Layer(sz(layers));

				layers.push_back(flatten);

				// Initialize the flatten layer's output dimensions
				ll input_size = 0;
				ll idx = sz(layers) - 1;
				if(layers[idx - 1]->conv) {
					input_size = layers[idx - 1]->conv->a->filter * layers[idx - 1]->conv->a->depth * layers[idx - 1]->conv->a->height * layers[idx - 1]->conv->a->width;
				} 
				else if(layers[idx - 1]->pooling) {
					input_size = layers[idx - 1]->pooling->a->filter * layers[idx - 1]->pooling->a->depth * layers[idx - 1]->pooling->a->height * layers[idx - 1]->pooling->a->width;
				}
				else if(layers[idx - 1]->flatten) {
					input_size = layers[idx - 1]->flatten->units;
				} 
				else if(layers[idx - 1]->dense) {
					input_size = layers[idx - 1]->dense->units;
				}
				else if(layers[idx - 1]->input) {
					input_size = layers[idx - 1]->input->height * layers[idx - 1]->input->width * layers[idx - 1]->input->depth;
				} 
				else {
					std::cerr<<"Error: Previous layer is not a convolution, pooling, dense, flatten or input layer."<<std::endl;
					exit(1);
				}

				flatten->flatten->units = input_size;
				
				flatten->flatten->a = new Tensor(1, 1, input_size, 1);
				flatten->flatten->dA = new Tensor(1, 1, input_size, 1);

				// // Print layer details
				// std::cout<<"Flatten Layer:"<<std::endl;
				// std::cout<<"  Id: "<<flatten->flatten->id<<std::endl;
				// std::cout<<"  Output Shape: ("<<flatten->flatten->a->filter<<", "<<flatten->flatten->a->height<<", "<<flatten->flatten->a->width<<", "<<flatten->flatten->a->depth<<")"<<std::endl;
			}
			else if(command == "compile") {
				char _loss_function[50], _optimizer[50], trash[50];
				db _lr = 0;

				sscanf(s.c_str(), "%s %s %s %d %d %lf", trash, _optimizer, _loss_function, &epochs, &batch_size, &_lr);

				loss = new Loss(std::str(_loss_function));
				optimizer = new Optimizer(std::str(_optimizer), _lr);
				
				// // Print compile details
				// std::cout<<"Compile:"<<std::endl;
				// std::cout<<"  Optimizer: "<<_optimizer<<std::endl;
				// std::cout<<"  Loss Function: "<<_loss_function<<std::endl;
				// std::cout<<"  Epochs: "<<epochs<<std::endl;
				// std::cout<<"  Batch Size: "<<batch_size<<std::endl;
				// std::cout<<"  Learning Rate: "<<_lr<<std::endl;
			}
		}

		// std::cout<<"======================"<<std::endl<<std::endl;
		ModelConfiguration.close();
	}

	void summary() {
		// Print model summary like Keras, including layer types, output shapes, and number of parameters
		std::cout << "Model Summary:" << std::endl;
		std::cout << "==============================================================" << std::endl;
		std::cout << std::left << std::setw(25) << "Layer (type)" << std::setw(25) << "Output Shape" << std::setw(15) << "Param #" << std::endl;
		std::cout << "==============================================================" << std::endl;

		ll total_params = 0;

		for (ll i = 0; i < sz(layers); i++) {
			std::str layer_name = "layer_" + std::to_string(i+1);
			std::str layer_type;
			std::str output_shape;
			ll params = 0;

			if (layers[i]->input) {
				layer_type = "Input";
				output_shape = "(None, " +
							  std::to_string(layers[i]->input->height) + ", " +
							  std::to_string(layers[i]->input->width) + ", " +
							  std::to_string(layers[i]->input->depth) + ")";
				
				params = 0; // Input layers have no trainable parameters
			}
			else if (layers[i]->conv) {
				layer_type = "Conv2D";
				output_shape = "(None, " +
							  std::to_string(layers[i]->conv->a->height) + ", " +
							  std::to_string(layers[i]->conv->a->width) + ", " +
							  std::to_string(layers[i]->conv->filter) + ")";
				
				ll prev_filter = (i > 0 && layers[i-1]->conv) ? layers[i-1]->conv->filter : 
								(layers[i-1]->pooling ? layers[i-1]->pooling->a->filter : 
								(layers[i-1]->input ? layers[i-1]->input->a->filter : 1));

				params = layers[i]->conv->filter * (prev_filter * layers[i]->conv->kernel->height * 
							layers[i]->conv->kernel->width * layers[i]->conv->kernel->depth + 1);
			}
			else if (layers[i]->pooling) {
				layer_type = layers[i]->pooling->type;
				output_shape = "(None, " +
							  std::to_string(layers[i]->pooling->a->height) + ", " +
							  std::to_string(layers[i]->pooling->a->width) + ", " +
							  std::to_string(layers[i]->pooling->a->filter) + ")";
				
				params = 0;
			}
			else if (layers[i]->flatten) {
				layer_type = "Flatten";
				output_shape = "(None, " + std::to_string(layers[i]->flatten->units) + ")";
				
				params = 0;
			}
			else if (layers[i]->dense) {
				layer_type = "Dense";
				output_shape = "(None, " + std::to_string(layers[i]->dense->units) + ")";
				
				params = layers[i]->dense->w->width * layers[i]->dense->units + layers[i]->dense->units;
			}

			std::cout << std::left << std::setw(25) << layer_name + " (" + layer_type + ")" 
					  << std::setw(25) << output_shape << std::setw(15) << params << std::endl;
			
			total_params += params;
		}

		std::cout << "==============================================================" << std::endl;
		std::cout << "Total params: " << total_params << std::endl;
	}

	// Model operations
	void forward(Tensor *input);
	void backward();
	void gradient_descent();

	// Reset states for forward and backward passes
	void reset_forward();
	void reset_backward_per_batch();
	void reset_backward_per_datapoint();

	// Other operations
	void save_model_to_file(const std::str filename);
	bool load_model_from_file(const std::str filename);

	~Model() {
		for(ll i=0;i<sz(layers);i++) delete layers[i];
		layers.clear();

		if(loss) delete loss;
	}
};

ll argmax(Tensor *x);
ll argmin(Tensor *x);

#endif // NN_H