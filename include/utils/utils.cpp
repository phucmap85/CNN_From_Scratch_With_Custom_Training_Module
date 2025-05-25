#include "utils.h"

db r2() {return ((db) rand() / (db) RAND_MAX) * 2 - 1;}

std::str to_str(ll k) {
	std::str res = "";
	while(k > 0) {
		res = char('0' + k % 10) + res;
		k /= 10;
	}
	return res.empty() ? "0" : res;
}