#ifndef NN_H
#define NN_H

#include <bits/stdc++.h>

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

struct Conv {
	ll filter;
	ll kernel_w, kernel_h;
	ll stride_w, stride_h;
	ll pad_w, pad_h;
	std::str activation;

	db **R, **G, **B, b, **a;
	
	Conv(ll _filter, ll _kernel_h, ll _kernel_w, ll _stride_h, ll _stride_w, ll _pad_h, ll _pad_w, std::str _activation) : 
		filter(_filter), kernel_h(_kernel_h), kernel_w(_kernel_w),
		stride_h(_stride_h), stride_w(_stride_w),
		pad_h(_pad_h), pad_w(_pad_w),
		activation(_activation) {}

	~Conv() {
		for(ll h=0;h<kernel_h;h++) {
			delete[] R[h];
			delete[] G[h];
			delete[] B[h];
		}

		delete[] R;
		delete[] G;
		delete[] B;
		if(a) delete[] a;
	}
};

struct Pooling {
	std::str type;
	ll pool_w, pool_h;
	
	Pooling(ll _pool_h, ll _pool_w, std::str _type) : pool_h(_pool_h), pool_w(_pool_w), type(_type) {}
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
	std::str type;
	Conv *conv;
	Pooling *pooling;
	Flatten *flatten;
	Dense *dense;
	
	Layer(std::str _type, ll filter, ll kernel_h, ll kernel_w, ll stride_h, ll stride_w, ll pad_h, ll pad_w, std::str activation) : type(_type) {
		conv = new Conv(filter, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation);
		pooling = NULL;
		flatten = NULL;
		dense = NULL;
	}
	
	Layer(std::str _type, ll pool_h, ll pool_w, std::str type) : type(_type) {
		pooling = new Pooling(pool_h, pool_w, type);
		conv = NULL;
		flatten = NULL;
		dense = NULL;
	}
	
	Layer( std::str _type) : type(_type) {
		flatten = new Flatten();
		conv = NULL;
		pooling = NULL;
		dense = NULL;
	}
	
	Layer(std::str _type, ll units, std::str activation) : type(_type) {
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

db r2();
void initialize_weights(std::vector<Layer*> &model);

#endif