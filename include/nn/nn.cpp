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

Tensor* softmax(Tensor *x) {
    db max_val = -1e9;

    for(ll i=0;i<x->filter;i++) {
        for(ll j=0;j<x->depth;j++) {
            for(ll h=0;h<x->height;h++) {
                for(ll w=0;w<x->width;w++) max_val = std::max(max_val, x->arr[i][j][h][w]);
            }
        }
    }

    db sum = 0.0;
    for(ll i=0;i<x->filter;i++) {
        for(ll j=0;j<x->depth;j++) {
            for(ll h=0;h<x->height;h++) {
                for(ll w=0;w<x->width;w++) sum += exp(x->arr[i][j][h][w] - max_val);
            }
        }
    }

    Tensor *output = new Tensor(x->filter, x->height, x->width, x->depth);
    for(ll i=0;i<x->filter;i++) {
        for(ll j=0;j<x->depth;j++) {
            for(ll h=0;h<x->height;h++) {
                for(ll w=0;w<x->width;w++) output->arr[i][j][h][w] = exp(x->arr[i][j][h][w] - max_val) / sum;
            }
        }
    }

    return output;
}

db activation_fn(Tensor *x, std::str activation) {
    if(activation == "relu") return relu(x->arr[0][0][0][0]);
    else if(activation == "sigmoid") return sigmoid(x->arr[0][0][0][0]);
    else if(activation == "tanh") return tanh(x->arr[0][0][0][0]);
    else if(activation == "softmax") return 0; // Softmax is handled separately
    else return linear(x->arr[0][0][0][0]);
}

db activation_fn_dx(Tensor *x, std::str activation) {
    if(activation == "relu") return relu_dx(x->arr[0][0][0][0]);
    else if(activation == "sigmoid") return sigmoid_dx(x->arr[0][0][0][0]);
    else if(activation == "tanh") return tanh_dx(x->arr[0][0][0][0]);
    else if(activation == "softmax") return 0; // Softmax is handled separately
    else return linear_dx(x->arr[0][0][0][0]);
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
                                activation_fn(new Tensor(value + conv->b->arr[kernel_index][0][0][0]), conv->activation);
                        }
                    }
                }
            }
        }
    }

    // for(ll f=0;f<conv->filter;f++) {
    //     conv->a->export_to_file("test_image/conv_output_" + to_str(f) + ".png", f);
    // }
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

    // for(ll f=0;f<input->filter;f++) {
    //     pool->a->export_to_file("test_image/pool_output_" + to_str(f) + ".png", f);
    // }
}

void flatten(Tensor *input, Flatten *flatten) {
    ll idx = 0;

    for(ll f=0;f<input->filter;f++) {
        for(ll c=0;c<input->depth;c++) {
            for(ll h=0;h<input->height;h++) {
                for(ll w=0;w<input->width;w++) flatten->a->arr[0][0][0][idx++] = input->arr[f][c][h][w];
            }
        }
    }
}

void dense(Tensor *input, Dense *dense) {
    // Assuming input is a flattened tensor
    ll input_size = input->filter * input->height * input->width * input->depth;

    ll w_idx = 0;
    for(ll i=0;i<dense->units;i++) {
        for(ll j=0;j<input_size;j++) {
            dense->y->arr[0][0][0][i] += dense->w->arr[0][0][0][w_idx++] * input->arr[0][0][0][j];
        }
        dense->a->arr[0][0][0][i] = activation_fn(new Tensor(dense->y->arr[0][0][0][i] + dense->b->arr[0][0][0][i]), dense->activation);
    }

    if(dense->activation == "softmax") dense->a = softmax(dense->y);
}

//////////////////////// Model class implementation ////////////////////////

void Model::forward(Tensor *input) {
    for(ll i=0;i<sz(layers);i++) {
        if(layers[i]->conv) {
            if(!i) convolution(input, layers[i]->conv);
            else {
                if(layers[i-1]->pooling) convolution(layers[i-1]->pooling->a, layers[i]->conv);
                else if(layers[i-1]->conv) convolution(layers[i-1]->conv->a, layers[i]->conv);
                else {
                    std::cerr<<"Error: Previous layer is not a convolution or pooling layer."<<std::endl;
                    exit(1);
                }
            }
        } else if(layers[i]->pooling) {
            if(!i) {
                std::cerr<<"Error: Pooling layer cannot be the first layer."<<std::endl;
                exit(1);
            } else {
                if(layers[i-1]->conv) pooling(layers[i-1]->conv->a, layers[i]->pooling);
                else if(layers[i-1]->pooling) pooling(layers[i-1]->pooling->a, layers[i]->pooling);
                else {
                    std::cerr<<"Error: Previous layer is not a convolution or pooling layer."<<std::endl;
                    exit(1);
                }
            }
        } else if(layers[i]->flatten) {
            if(!i) {
                std::cerr<<"Error: Flatten layer cannot be the first layer."<<std::endl;
                exit(1);
            } else {
                if(layers[i-1]->conv) flatten(layers[i-1]->conv->a, layers[i]->flatten);
                else if(layers[i-1]->pooling) flatten(layers[i-1]->pooling->a, layers[i]->flatten);
                else if(layers[i-1]->flatten) flatten(layers[i-1]->flatten->a, layers[i]->flatten);
                else {
                    std::cerr<<"Error: Previous layer is not a convolution, pooling, or flatten layer."<<std::endl;
                    exit(1);
                }
            }
        } else if(layers[i]->dense) {
            if(!i) {
                if(input->filter == 1 && input->depth == 1 && input->height == 1) {
                    dense(input, layers[i]->dense);
                } else {
                    std::cerr<<"Error: Dense layer cannot be the first layer with non-matching input shape."<<std::endl;
                    exit(1);
                }
            } else {
                if(layers[i-1]->flatten) dense(layers[i-1]->flatten->a, layers[i]->dense);
                else if(layers[i-1]->conv) {
                    if(layers[i-1]->conv->a->filter == 1 && layers[i-1]->conv->a->depth == 1 && layers[i-1]->conv->a->height == 1) {
                        dense(layers[i-1]->conv->a, layers[i]->dense);
                    } else {
                        std::cerr<<"Error: Previous convolution layer output shape does not match dense layer input shape."<<std::endl;
                        exit(1);
                    }
                } else if(layers[i-1]->pooling) {
                    if(layers[i-1]->pooling->a->filter == 1 && layers[i-1]->pooling->a->depth == 1 && layers[i-1]->pooling->a->height == 1) {
                        dense(layers[i-1]->pooling->a, layers[i]->dense);
                    } else {
                        std::cerr<<"Error: Previous pooling layer output shape does not match dense layer input shape."<<std::endl;
                        exit(1);
                    }
                } 
                else if(layers[i-1]->dense) dense(layers[i-1]->dense->a, layers[i]->dense);
                else {
                    std::cerr<<"Error: Previous layer is not a flatten, convolution, or pooling layer."<<std::endl;
                    exit(1);
                }
            }
        } else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }
}

void Model::backward() {

}

void Model::fit(std::vector<Tensor*> &X, std::vector<ll> &Y, const std::vector<std::str> &classes) {
    // Implement the training logic here
    // This will include forward pass, loss calculation, backward pass, and weight updates
}