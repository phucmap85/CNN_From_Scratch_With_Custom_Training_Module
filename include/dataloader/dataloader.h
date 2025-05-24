#ifndef DATALOADER_H
#define DATALOADER_H

#include <bits/stdc++.h>
#include "../lodepng/lodepng.h"

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

struct Image {
	ll width, height;
	db **R, **G, **B;

	Image(const std::str filename) {
		unsigned _width, _height;
		std::vector<unsigned char> pixels;

		unsigned error = lodepng::decode(pixels, _width, _height, filename.c_str());
		if(error) {
			std::cout<<"decoder error "<<error<<": "<<lodepng_error_text(error)<<std::endl;
			exit(1);
		}

		width = _width; height = _height;

		R = new db*[height + 5];
		G = new db*[height + 5];
		B = new db*[height + 5];
		
		ll temp = 0;
		for(ll h=0;h<height;h++) {
			R[h] = new db[width + 5];
			G[h] = new db[width + 5];
			B[h] = new db[width + 5];
		}

		for(ll h=0;h<height;h++) {
			for(ll w=0;w<width;w++) {
				R[h][w] = (db) (ll(pixels[temp + 0]) / 255.0);
				G[h][w] = (db) (ll(pixels[temp + 1]) / 255.0);
				B[h][w] = (db) (ll(pixels[temp + 2]) / 255.0);
				temp += 4;
			}
		}
	}

	~Image() {
		for(ll h=0;h<height;h++) {
			delete[] R[h];
			delete[] G[h];
			delete[] B[h];
		}

		delete[] R;
		delete[] G;
		delete[] B;
	}
};

void load_dataset(const ll maxLength, const std::str path, std::vector<Image*> &X, std::vector<ll> &Y, std::vector<std::str> &classes);

#endif // DATALOADER_H
