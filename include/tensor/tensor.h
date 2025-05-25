#ifndef TENSOR_H
#define TENSOR_H

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

#endif // TENSOR_H