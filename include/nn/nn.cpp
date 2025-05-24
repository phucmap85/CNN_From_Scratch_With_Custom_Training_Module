#include "nn.h"

// Generate random number between -1 and 1
db r2() {return ((db) rand() / (db) RAND_MAX) * 2 - 1;}

void initialize_weights(std::vector<Layer*> &model) {
    for(ll i=0;i<sz(model);i++) {
        if(model[i]->dense) {
            ll w_size = (model[i-1]->dense ? model[i-1]->dense->units : model[i-1]->flatten->units) * model[i]->dense->units;
            model[i]->dense->w = new db[w_size + 5];
            
            for(ll j=0;j<=w_size;j++) model[i]->dense->w[j] = r2();
            for(ll j=0;j<=model[i]->dense->units;j++) model[i]->dense->b[j] = r2();
        }
        else if(model[i]->conv) {
            model[i]->conv->R = new db*[model[i]->conv->kernel_h + 5];
            model[i]->conv->G = new db*[model[i]->conv->kernel_h + 5];
            model[i]->conv->B = new db*[model[i]->conv->kernel_h + 5];

            for(ll h=0;h<model[i]->conv->kernel_h;h++) {
                model[i]->conv->R[h] = new db[model[i]->conv->kernel_w + 5];
                model[i]->conv->B[h] = new db[model[i]->conv->kernel_w + 5];
                model[i]->conv->G[h] = new db[model[i]->conv->kernel_w + 5];
            }

            for(ll h=0;h<model[i]->conv->kernel_h;h++) {
                for(ll w=0;w<model[i]->conv->kernel_w;w++) {
                    model[i]->conv->R[h][w] = r2();
                    model[i]->conv->G[h][w] = r2();
                    model[i]->conv->B[h][w] = r2();
                }
            }

            model[i]->conv->b = r2();
        }
    }
}