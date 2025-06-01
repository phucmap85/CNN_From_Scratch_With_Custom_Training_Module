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

db activation_fn(db x, std::str activation) {
    if(activation == "relu") return relu(x);
    else if(activation == "sigmoid") return sigmoid(x);
    else if(activation == "tanh") return tanh(x);
    else if(activation == "softmax") return x; // Softmax is handled separately
    else return linear(x);
}

db activation_fn_dx(db x, std::str activation) {
    if(activation == "relu") return relu_dx(x);
    else if(activation == "sigmoid") return sigmoid_dx(x);
    else if(activation == "tanh") return tanh_dx(x);
    else if(activation == "softmax") return 1; // Softmax is handled separately
    else return linear_dx(x);
}


void convolution(Tensor *input, Conv *conv) {
    ll stride_h = conv->stride_h;
    ll stride_w = conv->stride_w;
    ll pad_h = conv->pad_h;
    ll pad_w = conv->pad_w;

    ll kernel_height = conv->kernel->height;
    ll kernel_width = conv->kernel->width;
    ll kernel_depth = conv->kernel->depth;

    ll input_height = input->height;
    ll input_width = input->width;
    ll input_depth = input->depth;
    ll output_height = conv->a->height;
    ll output_width = conv->a->width;

    if(kernel_depth != input_depth) {
        std::cerr<<"Error: Kernel depth does not match input depth."<<std::endl;
        return;
    }
    
    for(ll kernel_index=0;kernel_index<conv->filter;kernel_index++) {
        for(ll h=0;h<output_height;h++) {
            for(ll w=0;w<output_width;w++) {
                db value = 0.0;
                for(ll kh=0;kh<kernel_height;kh++) {
                    for(ll kw=0;kw<kernel_width;kw++) {
                        ll input_h = h * stride_h - pad_h + kh;
                        ll input_w = w * stride_w - pad_w + kw;
                        if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                            db value = 0.0;
                            for(ll input_index=0;input_index<input->filter;input_index++) {
                                for(ll kd=0;kd<kernel_depth;kd++) {
                                    value += conv->kernel->arr[kernel_index][kd][kh][kw] * 
                                            input->arr[input_index][kd][input_h][input_w];
                                }
                            }
                            conv->a->arr[kernel_index][0][h][w] = 
                                activation_fn(value + conv->b->arr[kernel_index][0][0][0], conv->activation);
                        }
                    }
                }
            }
        }
    }

    for(ll f=0;f<conv->filter;f++) {
        conv->a->export_to_file("test_image/conv_output_" + to_str(f) + ".png", f);
    }
}

void pooling(Tensor *input, Pooling *pool) {
    ll pool_height = pool->pool_h;
    ll pool_width = pool->pool_w;
    ll pool_depth = 1;
    ll pool_stride_h = pool->stride_h;
    ll pool_stride_w = pool->stride_w;
    ll pool_pad_h = pool->pad_h;
    ll pool_pad_w = pool->pad_w;
    std::str type = pool->type;

    ll input_height = input->height;
    ll input_width = input->width;
    ll input_depth = input->depth;
    ll output_height = pool->a->height;
    ll output_width = pool->a->width;

    if(pool_depth != input_depth) {
        std::cerr<<"Error: Pooling kernel depth does not match input depth."<<std::endl;
        return;
    }
    
    for(ll input_index=0;input_index<input->filter;input_index++) {
        for(ll h=0;h<output_height;h++) {
            for(ll w=0;w<output_width;w++) {
                db value = 0.0;
                if(type == "maxpool") {
                    value = -1e9;
                    for(ll ph=0;ph<pool_height;ph++) {
                        for(ll pw=0;pw<pool_width;pw++) {
                            ll input_h = h * pool_stride_h - pool_pad_h + ph;
                            ll input_w = w * pool_stride_w - pool_pad_w + pw;

                            // Ensure input_h and input_w are within bounds
                            if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                                value = std::max(value, input->arr[input_index][0][input_h][input_w]);
                            }
                        }
                    }
                } else if(type == "avgpool") {
                    db count = 0;
                    for(ll ph=0;ph<pool_height;ph++) {
                        for(ll pw=0;pw<pool_width;pw++) {
                            ll input_h = h * pool_stride_h - pool_pad_h + ph;
                            ll input_w = w * pool_stride_w - pool_pad_w + pw;

                            // Ensure input_h and input_w are within bounds
                            if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                                value += input->arr[input_index][0][input_h][input_w];
                                count++;
                            }
                        }
                    }
                    value /= count;
                }
                pool->a->arr[input_index][0][h][w] = value;
            }
        }
    }

    for(ll f=0;f<input->filter;f++) {
        pool->a->export_to_file("test_image/pool_output_" + to_str(f) + ".png", f);
    }
}