#ifndef NN_H
#define NN_H

#include <bits/stdc++.h>
#include "../lodepng/lodepng.h"

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second
#define r2 (((db) rand() / (db) RAND_MAX) * 2 - 1);

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
					for(ll w=0;w<width;w++) arr[f][c][h][w] = r2; // Initialize randomly
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
		db min_val = std::numeric_limits<db>::max();
		db max_val = std::numeric_limits<db>::lowest();
		
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
					pixels[(h * width + w) * 4 + c] = (unsigned char)(normalized * 255.0);
				}
				pixels[(h * width + w) * 4 + 3] = 255; // Alpha channel
			}
		}

		unsigned error = lodepng::encode(filename, pixels, width, height, 
			depth < 3 ? LCT_GREY : LCT_RGBA);
		
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

struct Conv {
	ll filter;
	ll stride_h, stride_w;
	ll pad_h, pad_w;
	ll input_layer;
	std::str activation;

	Tensor *kernel, *b, *a;
	
	Conv(ll _filter, ll _kernel_h, ll _kernel_w, ll _stride_h, ll _stride_w, ll _pad_h, ll _pad_w, std::str _activation) : 
		filter(_filter),
		stride_h(_stride_h), stride_w(_stride_w),
		pad_h(_pad_h), pad_w(_pad_w),
		activation(_activation) {}

	~Conv() {
		if(kernel) delete kernel;
		if(a) delete a;
		if(b) delete b;
	}
};

struct Pooling {
	std::str type;
	ll pool_h, pool_w;
	ll stride_h, stride_w;
	ll pad_h, pad_w;

	Tensor *a;
	
	Pooling(ll _pool_h, ll _pool_w, ll _stride_h, ll _stride_w, ll _pad_h, ll _pad_w, std::str _type) : 
		pool_h(_pool_h), pool_w(_pool_w),
		stride_h(_stride_h), stride_w(_stride_w),
		pad_h(_pad_h), pad_w(_pad_w),
		type(_type) {}

	~Pooling() {
		if(a) delete a;
	}
};

struct Flatten {
	ll units;
	Tensor *a;

	Flatten() : units(0) {}

	~Flatten() {
		if(a) delete a;
	}
};

struct Dense {
	ll units;
	std::str activation;

	Tensor *a, *y, *w, *b;
	
	Dense(ll _units, std::str _activation) : units(_units), activation(_activation) {
		a = new Tensor(1, 1, units, 1);
		y = new Tensor(1, 1, units, 1);
		b = new Tensor(1, 1, units, 1);
	}

	~Dense() {
		if(w) delete w;
		if(a) delete a;
		if(y) delete y;
		if(b) delete b;
	}
};

struct Layer {
	Conv *conv;
	Pooling *pooling;
	Flatten *flatten;
	Dense *dense;
	
	Layer(ll filter, ll kernel_h, ll kernel_w, ll stride_h, ll stride_w, ll pad_h, ll pad_w, std::str activation) {
		conv = new Conv(filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation);
		pooling = NULL;
		flatten = NULL;
		dense = NULL;
	}
	
	Layer(ll pool_h, ll pool_w, ll stride_h, ll stride_w, ll pad_h, ll pad_w, std::str type) {
		pooling = new Pooling(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, type);
		conv = NULL;
		flatten = NULL;
		dense = NULL;
	}
	
	Layer() {
		flatten = new Flatten();
		conv = NULL;
		pooling = NULL;
		dense = NULL;
	}
	
	Layer(ll units, std::str activation) {
		dense = new Dense(units, activation);
		conv = NULL;
		pooling = NULL;
		flatten = NULL;
	}

	~Layer() {
		if(conv) delete conv;
		if(pooling) delete pooling;
		if(flatten) delete flatten;
		if(dense) delete dense;
	}
};

struct Model {
	ll epochs;
	ll batch_size;
	db lr;
	db best;
	std::str loss_function;
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

			if(command == "conv") {
				ll filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, input_layer = 0, input_h, input_w, input_d;
				char _activation[50], trash[50];

				sscanf(s.c_str(), "%s %d (%d, %d) (%d, %d) (%d, %d) %s %d (%d, %d, %d)", trash, &filter, &kernel_h, &kernel_w, 
					&stride_h, &stride_w, &pad_h, &pad_w, _activation, &input_layer, &input_h, &input_w, &input_d);

				std::str activation = std::str(_activation);

				Layer *conv = new Layer(filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation);

				layers.push_back(conv);
				
				// Initialize the convolution layer's kernel and output dimensions
				ll output_height = 0, output_width = 0;
				
				if(input_layer) {
					output_height = (input_h - kernel_h + 2 * pad_h) / stride_h + 1;
					output_width = (input_w - kernel_w + 2 * pad_w) / stride_w + 1;
				} else {
					ll idx = sz(layers) - 1;
					if(layers[idx - 1]->pooling) {
						output_height = (layers[idx - 1]->pooling->a->height - kernel_h + 2 * pad_h) / stride_h + 1;
						output_width = (layers[idx - 1]->pooling->a->width - kernel_w + 2 * pad_w) / stride_w + 1;
					} 
					else if(layers[idx - 1]->conv) {
						output_height = (layers[idx - 1]->conv->a->height - kernel_h + 2 * pad_h) / stride_h + 1;
						output_width = (layers[idx - 1]->conv->a->width - kernel_w + 2 * pad_w) / stride_w + 1;
					} else {
						std::cerr<<"Error: Previous layer is not a convolution or pooling layer."<<std::endl;
						exit(1);
					}
				}

				conv->conv->kernel = new Tensor(filter, kernel_h, kernel_w, input_layer ? input_d : 1);
				conv->conv->b = new Tensor(filter, 1, 1, 1);
				conv->conv->a = new Tensor(filter, output_height, output_width, 1);

				// // Print layer details
				// std::cout<<"Convolution Layer:"<<std::endl;
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

				Layer *pooling = new Layer(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, type);

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

				// // Print layer details
				// std::cout<<"Pooling Layer:"<<std::endl;
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

				Layer *dense = new Layer(units, activation);

				layers.push_back(dense);

				// Initialize the dense layer's weights and biases
				ll input_size = 0;
				ll idx = sz(layers) - 1;

				if(layers[idx - 1]->dense) {
					input_size = layers[idx - 1]->dense->units;
				} 
				else if(layers[idx - 1]->flatten) {
					input_size = layers[idx - 1]->flatten->units;
				} else {
					std::cerr<<"Error: Previous layer is not a dense or flatten layer."<<std::endl;
					exit(1);
				}

				dense->dense->w = new Tensor(1, 1, input_size * units, 1);

				// // Print layer details
				// std::cout<<"Dense Layer:"<<std::endl;
				// std::cout<<"  Units: "<<units<<std::endl;
				// std::cout<<"  Activation: "<<activation<<std::endl;
			}
			else if(command == "flatten") {
				Layer *flatten = new Layer();

				layers.push_back(flatten);

				// Initialize the flatten layer's output dimensions
				ll input_size = 0;
				ll idx = sz(layers) - 1;
				if(layers[idx - 1]->conv) {
					input_size = layers[idx - 1]->conv->a->filter * layers[idx - 1]->conv->a->depth * layers[idx - 1]->conv->a->height * layers[idx - 1]->conv->a->width;
				} 
				else if(layers[idx - 1]->pooling) {
					input_size = layers[idx - 1]->pooling->a->filter * layers[idx - 1]->pooling->a->depth * layers[idx - 1]->pooling->a->height * layers[idx - 1]->pooling->a->width;
				} else {
					std::cerr<<"Error: Previous layer is not a convolution or pooling layer."<<std::endl;
					exit(1);
				}

				flatten->flatten->units = input_size;
				flatten->flatten->a = new Tensor(1, 1, input_size, 1);

				// // Print layer details
				// std::cout<<"Flatten Layer:"<<std::endl;
				// std::cout<<"  Output Shape: ("<<flatten->flatten->a->filter<<", "<<flatten->flatten->a->height<<", "<<flatten->flatten->a->width<<", "<<flatten->flatten->a->depth<<")"<<std::endl;
			}
			else if(command == "compile") {
				char _loss_function[50], trash[50];

				sscanf(s.c_str(), "%s %s %d %d %lf", trash, _loss_function, &epochs, &batch_size, &lr);

				loss_function = std::str(_loss_function);
				
				// // Print compile details
				// std::cout<<"Compile:"<<std::endl;
				// std::cout<<"  Loss Function: "<<loss_function<<std::endl;
				// std::cout<<"  Epochs: "<<epochs<<std::endl;
				// std::cout<<"  Batch Size: "<<batch_size<<std::endl;
				// std::cout<<"  Learning Rate: "<<lr<<std::endl;
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

			if (layers[i]->conv) {
				layer_type = "Conv2D";
				output_shape = "(None, " +
							  std::to_string(layers[i]->conv->a->height) + ", " +
							  std::to_string(layers[i]->conv->a->width) + ", " +
							  std::to_string(layers[i]->conv->filter) + ")";
				
				ll prev_filter = 1;
				
				if(i > 0) prev_filter = layers[i-1]->conv ? layers[i-1]->conv->filter : 
								(layers[i-1]->pooling ? layers[i-1]->pooling->a->filter : 1);
				
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
				
				params = layers[i]->dense->w->width + layers[i]->dense->units;
			}

			std::cout << std::left << std::setw(25) << layer_name + " (" + layer_type + ")" 
					  << std::setw(25) << output_shape << std::setw(15) << params << std::endl;
			
			total_params += params;
		}

		std::cout << "==============================================================" << std::endl;
		std::cout << "Total params: " << total_params << std::endl;
	}

	void forward(Tensor *input);
	void backward();

	void fit(std::vector<Tensor*> &X, std::vector<ll> &Y, const std::vector<std::str> &classes);

	~Model() {
		for(ll i=0;i<sz(layers);i++) delete layers[i];
		layers.clear();
	}
};

void convolution(Tensor *input, Conv *conv);
void pooling(Tensor *input, Pooling *pool);
void flatten(Tensor *input, Flatten *flatten);
void dense(Tensor *input, Dense *dense);

#endif // NN_H