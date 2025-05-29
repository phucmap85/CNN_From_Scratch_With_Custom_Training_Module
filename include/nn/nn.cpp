#include "nn.h"

// Activation function
db sigmoid(db x) {return 1 / (1 + exp(-x));}
db sigmoid_dx(db x) {return sigmoid(x) * (1 - sigmoid(x));}
db relu(db x) {return x > 0 ? x : 0;}
db relu_dx(db x) {return x > 0;}
db tanh(db x) {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));}
db tanh_dx(db x) {return 1 - tanh(x) * tanh(x);}
db linear(db x) {return x;}
db linear_dx(db x) {return 1;}

void initialize_weights(std::vector<Layer*> &model) {
    for(ll i=0;i<sz(model);i++) {
        if(model[i]->dense) {
            ll w_size = (model[i-1]->dense ? model[i-1]->dense->units : model[i-1]->flatten->units) * model[i]->dense->units;
            model[i]->dense->w = new db[w_size + 5];
            
            for(ll j=0;j<=w_size;j++) model[i]->dense->w[j] = r2();
            for(ll j=0;j<=model[i]->dense->units;j++) model[i]->dense->b[j] = r2();
        }
    }
}

void convolution(Tensor *input, Conv *conv) {
    ll filter = conv->filter;
    ll kernel_h = conv->kernel[0]->height;
    ll kernel_w = conv->kernel[0]->width;
    ll stride_h = conv->stride_h;
    ll stride_w = conv->stride_w;
    ll pad_h = conv->pad_h;
    ll pad_w = conv->pad_w;

    // Calculate output dimensions
    ll input_height = input->height;
    ll input_width = input->width;
    ll output_height = (input_height + 2 * pad_h - kernel_h) / stride_h + 1;
    ll output_width = (input_width + 2 * pad_w - kernel_w) / stride_w + 1;

    for(ll kernel_index=0;kernel_index<filter;kernel_index++) {
        Tensor *kernel = conv->kernel[kernel_index];
        Tensor *output = new Tensor(output_height, output_width, 1);

        for(ll h=0;h<output_height;h++) {
            for(ll w=0;w<output_width;w++) {
                db sum = 0.0;
                for(ll kh=0; kh<kernel_h;kh++) {
                    for(ll kw=0;kw<kernel_w;kw++) {
                        ll input_h = h * stride_h - pad_h + kh;
                        ll input_w = w * stride_w - pad_w + kw;
                        if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                            for(ll c=0;c<input->depth;c++) {
                                sum += input->arr[c][input_h][input_w] * kernel->arr[c][kh][kw];
                            }
                        }
                    }
                }
                output->arr[0][h][w] = sum + conv->b;
            }
        }

        // Apply activation function
        if(conv->activation == "sigmoid") {
            for(ll h = 0; h < output_height; h++) {
                for(ll w = 0; w < output_width; w++) {
                    output->arr[0][h][w] = sigmoid(output->arr[0][h][w]);
                }
            }
        } else if(conv->activation == "relu") {
            for(ll h = 0; h < output_height; h++) {
                for(ll w = 0; w < output_width; w++) {
                    output->arr[0][h][w] = relu(output->arr[0][h][w]);
                }
            }
        } else if(conv->activation == "tanh") {
            for(ll h = 0; h < output_height; h++) {
                for(ll w = 0; w < output_width; w++) {
                    output->arr[0][h][w] = tanh(output->arr[0][h][w]);
                }
            }
        }

        conv->a[kernel_index] = new Tensor(output_height, output_width, 1);
        for(ll h = 0; h < output_height; h++) {
            for(ll w = 0; w < output_width; w++) {
                conv->a[kernel_index]->arr[0][h][w] = output->arr[0][h][w];
            }
        }

        delete output;

        conv->a[kernel_index]->export_to_file("test_image/output_" + to_str(kernel_index) + ".png");
    }
}

void pooling(Conv *input, Pooling *pool) {
    ll pool_h = pool->pool_h;
    ll pool_w = pool->pool_w;
    std::str type = pool->type;

    ll input_height = input->a[0]->height;
    ll input_width = input->a[0]->width;
    ll output_height = (input_height - pool_h) + 1;
    ll output_width = (input_width - pool_w) + 1;

    for(ll kernel_index = 0; kernel_index < input->filter; kernel_index++) {
        Tensor *output = new Tensor(output_height, output_width, 1);

        for(ll h=0;h<output_height;h++) {
            for(ll w=0;w<output_width;w++) {
                db value = (type == "max") ? -1e9 : 0.0;
                for(ll ph=0; ph<pool_h;ph++) {
                    for(ll pw=0;pw<pool_w;pw++) {
                        ll input_h = h * 2 + ph;
                        ll input_w = w * 2 + pw;
                        if(input_h < input_height && input_w < input_width) {
                            db current_value = input->a[kernel_index]->arr[0][input_h][input_w];
                            if(type == "max") {
                                value = std::max(value, current_value);
                            } else if(type == "avg") {
                                value += current_value / (pool_h * pool_w);
                            }
                        }
                    }
                }
                output->arr[0][h][w] = value;
            }
        }

        pool->a[kernel_index] = new Tensor(output_height, output_width, 1);
        for(ll h = 0; h < output_height; h++) {
            for(ll w = 0; w < output_width; w++) {
                pool->a[kernel_index]->arr[0][h][w] = output->arr[0][h][w];
            }
        }

        delete output;

        pool->a[kernel_index]->export_to_file("test_image/pool_output_" + to_str(kernel_index) + ".png");
    }
}
