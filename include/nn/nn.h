#ifndef NN_H
#define NN_H

#include <bits/stdc++.h>
#include "../lodepng/lodepng.h"
#include "../utils/utils.h"

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

struct Tensor {
	ll width, height, depth; // Assuming RGB Tensors
	db ***arr;

	Tensor(const ll _height, const ll _width, const ll _depth) : height(_height), width(_width), depth(_depth) {
		arr = new db**[depth + 5];

		for(ll c=0;c<depth;c++) {
			arr[c] = new db*[height + 5];
			for(ll h=0;h<height;h++) arr[c][h] = new db[width + 5];
		}

		for(ll c=0;c<depth;c++) {
			for(ll h=0;h<height;h++) {
				for(ll w=0;w<width;w++) arr[c][h][w] = r2();
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

		width = _width; height = _height; depth = 3;

		arr = new db**[depth + 5];

		for(ll c=0;c<depth;c++) {
			arr[c] = new db*[height + 5];
			for(ll h=0;h<height;h++) arr[c][h] = new db[width + 5];
		}

		for(ll h=0;h<height;h++) {
			for(ll w=0;w<width;w++) {
				for(ll c=0;c<depth;c++) { // RGB channels
					arr[c][h][w] = pixels[(h * height + w) * 4 + c] / 255.0;
				}
			}
		}
	}

	void export_to_file(const std::str filename) {
		std::vector<unsigned char> pixels;
		pixels.resize(width * height * 4);

		for(ll h=0;h<height;h++) {
			for(ll w=0;w<width;w++) {
				for(ll c=0;c<depth;c++) { // RGB channels
					pixels[(h * width + w) * 4 + c] = (unsigned char) (arr[c][h][w] * 255.0);
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
			for(ll c=0;c<depth;c++) {
				for(ll h=0;h<height;h++) delete[] arr[c][h];
				delete[] arr[c];
			}
			delete[] arr;
		}
	}
};

struct Conv {
	ll filter;
	ll stride_h, stride_w;
	ll pad_h, pad_w;
	std::str activation;

	Tensor **kernel, **a;
	db b;
	
	Conv(ll _filter, ll _kernel_h, ll _kernel_w, ll _stride_h, ll _stride_w, ll _pad_h, ll _pad_w, std::str _activation, bool _input_layer) : 
		filter(_filter),
		stride_h(_stride_h), stride_w(_stride_w),
		pad_h(_pad_h), pad_w(_pad_w),
		activation(_activation) {
			a = new Tensor*[_filter + 5];
			
			kernel = new Tensor*[_filter + 5];
			for(ll i=0;i<_filter;i++) {
				// Assuming 3 channels (RGB) for input layer, 1 for others
				kernel[i] = new Tensor(_kernel_h, _kernel_w, _input_layer ? 3 : 1);
			}

			b = r2();
		}

	~Conv() {
		if(kernel) {
			for(ll i=0;i<filter;i++) {
				if(kernel[i]) delete kernel[i];
			}
			delete[] kernel;
		}
		
		if(a) {
			for(ll i=0;i<filter;i++) {
				if(a[i]) delete a[i];
			}
			delete[] a;
		}
	}
};

struct Pooling {
	std::str type;
	ll pool_h, pool_w;

	Tensor *kernel, **a;
	
	Pooling(ll _pool_h, ll _pool_w, std::str _type) : pool_h(_pool_h), pool_w(_pool_w), type(_type) {
		a = NULL;
		kernel = new Tensor(pool_h, pool_w, 1);
	}
};

struct Flatten {
	ll units;
	db *a;

	Flatten() : units(0) {
		a = NULL;
	}
};

struct Dense {
	ll units;
	std::str activation;

	db *a, *y, *w, *b;
	
	Dense(ll _units, std::str _activation) : units(_units), activation(_activation) {
		a = new db[units + 5];
		y = new db[units + 5];
		b = new db[units + 5];
	}

	~Dense() {
		if(a) delete[] a;
		if(y) delete[] y;
		if(w) delete[] w;
		if(b) delete[] b;
	}
};

struct Layer {
	Conv *conv;
	Pooling *pooling;
	Flatten *flatten;
	Dense *dense;
	
	Layer(ll filter, ll kernel_h, ll kernel_w, ll stride_h, ll stride_w, ll pad_h, ll pad_w, std::str activation, bool input_layer) {
		conv = new Conv(filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation, input_layer);
		pooling = NULL;
		flatten = NULL;
		dense = NULL;
	}
	
	Layer(ll pool_h, ll pool_w, std::str type) {
		pooling = new Pooling(pool_h, pool_w, type);
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

void initialize_weights(std::vector<Layer*> &model);
void convolution(Tensor *input, Conv *conv);

#endif // NN_H
