#include "nn.h"

// Kaiming initialization
db r2(const db size) {
	db scale = sqrt(2.0f / size);
	return ((db) rand() / (db) RAND_MAX) * scale - (scale / 2.0f);
}

ll argmax(Tensor *x) {
    // Asmuming x is a 1D tensor
    ll max_index = 0;
    db max_value = x->arr[0][0][0][0];
    for(ll w=0;w<x->width;w++) {
        if(x->arr[0][0][0][w] > max_value) {
            max_value = x->arr[0][0][0][w];
            max_index = w;
        }
    }
    return max_index;
}

ll argmin(Tensor *x) {
    // Asmuming x is a 1D tensor
    ll min_index = 0;
    db min_value = x->arr[0][0][0][0];
    for(ll w=0;w<x->width;w++) {
        if(x->arr[0][0][0][w] < min_value) {
            min_value = x->arr[0][0][0][w];
            min_index = w;
        }
    }
    return min_index;
}

// Activation function
db sigmoid(const db x) {return 1 / (1 + exp(-x));}
db sigmoid_dx(const db x) {return sigmoid(x) * (1 - sigmoid(x));}
db relu(const db x) {return x > 0 ? x : 0;}
db relu_dx(const db x) {return x > 0;}
db tanh(const db x) {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));}
db tanh_dx(const db x) {return 1 - tanh(x) * tanh(x);}
db linear(const db x) {return x;}
db linear_dx(const db x) {return 1;}

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
    db temp = x->arr[0][0][0][0];
    delete x;

    if(activation == "relu") return relu(temp);
    else if(activation == "sigmoid") return sigmoid(temp);
    else if(activation == "tanh") return tanh(temp);
    else if(activation == "softmax") return 1; // Softmax is handled separately
    else return linear(temp);
}

db activation_fn_dx(Tensor *x, std::str activation) {
    db temp = x->arr[0][0][0][0];
    delete x;

    if(activation == "relu") return relu_dx(temp);
    else if(activation == "sigmoid") return sigmoid_dx(temp);
    else if(activation == "tanh") return tanh_dx(temp);
    else if(activation == "softmax") return 1; // Softmax is handled separately
    else return linear_dx(temp);
}

//////////////////////// Forward Operation ////////////////////////

void convolution_fw(Layer *prev, Conv *conv) {
    Tensor *input;
    if(prev->conv) input = prev->conv->a;
    else if(prev->pooling) input = prev->pooling->a;
    else if(prev->input) input = prev->input->a;
    else {
        std::cerr<<"Error in convolution_fx"<<std::endl;
        return;
    }

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
                for(ll kh=0;kh<kernel_height;kh++) {
                    for(ll kw=0;kw<kernel_width;kw++) {
                        ll input_h = h * stride_h - pad_h + kh;
                        ll input_w = w * stride_w - pad_w + kw;
                        if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                            for(ll input_index=0;input_index<input->filter;input_index++) {
                                for(ll kd=0;kd<kernel_depth;kd++) {
                                    conv->y->arr[kernel_index][0][h][w] += conv->kernel->arr[kernel_index][kd][kh][kw] * 
                                            input->arr[input_index][kd][input_h][input_w];
                                }
                            }
                        }
                    }
                }
                conv->y->arr[kernel_index][0][h][w] += conv->b->arr[kernel_index][0][0][0];

                conv->a->arr[kernel_index][0][h][w] = activation_fn(new Tensor(conv->y->arr[kernel_index][0][h][w]), conv->activation);
            }
        }
    }

    // for(ll f=0;f<conv->filter;f++) {
    //     conv->a->export_to_file("test_image/conv_layer_" + std::to_string(conv->id) + "_output_" + std::to_string(f) + ".png", f);
    // }
}

void pooling_fw(Layer *prev, Pooling *pool) {
    Tensor *input;
    if(prev->conv) input = prev->conv->a;
    else if(prev->pooling) input = prev->pooling->a;
    else {
        std::cerr<<"Error in pooling_fw"<<std::endl;
        return;
    }

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
    //     pool->a->export_to_file("test_image/pool_layer_" + std::to_string(pool->id) + "_output_" + std::to_string(f) + ".png", f);
    // }
}

void flatten_fw(Layer *prev, Flatten *flatten) {
    Tensor *input;
    if(prev->conv) input = prev->conv->a;
    else if(prev->pooling) input = prev->pooling->a;
    else if(prev->flatten) input = prev->flatten->a;
    else if(prev->dense) input = prev->dense->a;
    else if(prev->input) input = prev->input->a;
    else {
        std::cerr<<"Error in flatten_fw"<<std::endl;
        return;
    }

    ll idx = 0;

    for(ll f=0;f<input->filter;f++) {
        for(ll c=0;c<input->depth;c++) {
            for(ll h=0;h<input->height;h++) {
                for(ll w=0;w<input->width;w++) flatten->a->arr[0][0][0][idx++] = input->arr[f][c][h][w];
            }
        }
    }
}

void dense_fw(Layer *prev, Dense *dense) {
    Tensor *input;
    if(prev->conv) input = prev->conv->a;
    else if(prev->pooling) input = prev->pooling->a;
    else if(prev->input) input = prev->input->a;
    else if(prev->flatten) input = prev->flatten->a;
    else if(prev->dense) input = prev->dense->a;
    else {
        std::cerr<<"Error in dense_fw"<<std::endl;
        return;
    }

    // Assuming input is a flattened tensor
    ll input_size = input->filter * input->height * input->width * input->depth;

    for(ll i=0;i<dense->units;i++) {
        for(ll j=0;j<input_size;j++) {
            dense->y->arr[0][0][0][i] += dense->w->arr[i][0][0][j] * input->arr[0][0][0][j];
        }
        dense->y->arr[0][0][0][i] += dense->b->arr[0][0][0][i];

        dense->a->arr[0][0][0][i] = activation_fn(new Tensor(dense->y->arr[0][0][0][i]), dense->activation);
    }

    if(dense->activation == "softmax") dense->a = softmax(dense->y);
}

//////////////////////// Backward Operation ////////////////////////

void convolution_bw(Layer *prev, Conv *conv) {
    Tensor *input, *prev_dA;

    if(prev->conv) {
        input = prev->conv->a;
        prev_dA = prev->conv->dA;
    }
    else if(prev->pooling) {
        input = prev->pooling->a;
        prev_dA = prev->pooling->dA;
    }
    else if(prev->input) {
        input = prev->input->a;
        prev_dA = prev->input->dA;
    }
    else {
        std::cerr<<"Error in convolution_bw"<<std::endl;
        return;
    }

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
                for(ll kh=0;kh<kernel_height;kh++) {
                    for(ll kw=0;kw<kernel_width;kw++) {
                        ll input_h = h * stride_h - pad_h + kh;
                        ll input_w = w * stride_w - pad_w + kw;
                        if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                            for(ll input_index=0;input_index<input->filter;input_index++) {
                                for(ll kd=0;kd<kernel_depth;kd++) {
                                    // Calculate the gradient for the kernel
                                    conv->dW->arr[kernel_index][kd][kh][kw] += conv->dA->arr[kernel_index][0][h][w] * 
                                        activation_fn_dx(new Tensor(conv->y->arr[kernel_index][0][h][w]), conv->activation) * 
                                        input->arr[input_index][kd][input_h][input_w];

                                    // Calculate the gradient for the input
                                    prev_dA->arr[input_index][kd][input_h][input_w] += conv->dA->arr[kernel_index][0][h][w] *
                                        activation_fn_dx(new Tensor(conv->y->arr[kernel_index][0][h][w]), conv->activation) *
                                        conv->kernel->arr[kernel_index][kd][kh][kw];
                                }
                            }
                        }
                    }
                }
                // Calculate the gradient for the bias
                conv->dB->arr[kernel_index][0][0][0] += conv->dA->arr[kernel_index][0][h][w] * 
                    activation_fn_dx(new Tensor(conv->y->arr[kernel_index][0][h][w]), conv->activation);
            }
        }
    }
}

void pooling_bw(Layer *prev, Pooling *pool) {
    Tensor *input, *prev_dA;

    if(prev->conv) {
        input = prev->conv->a;
        prev_dA = prev->conv->dA;
    }
    else if(prev->pooling) {
        input = prev->pooling->a;
        prev_dA = prev->pooling->dA;
    }
    else {
        std::cerr<<"Error in pooling_bw"<<std::endl;
        return;
    }

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
                if(type == "maxpool") {
                    db value = -1e9;
                    ll max_h = -1, max_w = -1; // To track the position of the max value
                    for(ll ph=0;ph<pool_height;ph++) {
                        for(ll pw=0;pw<pool_width;pw++) {
                            ll input_h = h * pool_stride_h - pool_pad_h + ph;
                            ll input_w = w * pool_stride_w - pool_pad_w + pw;

                            // Ensure input_h and input_w are within bounds
                            if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                                if(input->arr[input_index][0][input_h][input_w] > value) {
                                    value = input->arr[input_index][0][input_h][input_w];
                                    max_h = input_h;
                                    max_w = input_w;
                                }
                            }
                        }
                    }

                    // Backpropagate the gradient only to the max position
                    if(max_h != -1 && max_w != -1) {
                        prev_dA->arr[input_index][0][max_h][max_w] += pool->dA->arr[input_index][0][h][w];
                    }
                } else if(type == "avgpool") {
                    db count = 0;
                    for(ll ph=0;ph<pool_height;ph++) {
                        for(ll pw=0;pw<pool_width;pw++) {
                            ll input_h = h * pool_stride_h - pool_pad_h + ph;
                            ll input_w = w * pool_stride_w - pool_pad_w + pw;

                            // Ensure input_h and input_w are within bounds
                            count += (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width);
                        }
                    }
                    
                    for(ll ph=0;ph<pool_height;ph++) {
                        for(ll pw=0;pw<pool_width;pw++) {
                            ll input_h = h * pool_stride_h - pool_pad_h + ph;
                            ll input_w = w * pool_stride_w - pool_pad_w + pw;

                            // Ensure input_h and input_w are within bounds
                            if(input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                                // Backpropagate the gradient evenly across all positions in the pooling window
                                prev_dA->arr[input_index][0][input_h][input_w] += pool->dA->arr[input_index][0][h][w] / count;
                            }
                        }
                    }
                }
            }
        }
    }
}

void flatten_bw(Layer *prev, Flatten *flatten) {
    Tensor *input, *prev_dA;

    if(prev->conv) {
        input = prev->conv->a;
        prev_dA = prev->conv->dA;
    }
    else if(prev->pooling) {
        input = prev->pooling->a;
        prev_dA = prev->pooling->dA;
    }
    else if(prev->flatten) {
        input = prev->flatten->a;
        prev_dA = prev->flatten->dA;
    }
    else if(prev->dense) {
        input = prev->dense->a;
        prev_dA = prev->dense->dA;
    }
    else if(prev->input) {
        input = prev->input->a;
        prev_dA = prev->input->dA;
    }
    else {
        std::cerr<<"Error in flatten_bw"<<std::endl;
        return;
    }

    ll idx = 0;

    for(ll f=0;f<input->filter;f++) {
        for(ll c=0;c<input->depth;c++) {
            for(ll h=0;h<input->height;h++) {
                for(ll w=0;w<input->width;w++) {
                    prev_dA->arr[f][c][h][w] += flatten->dA->arr[0][0][0][idx++];
                }
            }
        }
    }
}

void dense_bw(Layer *prev, Dense *dense) {
    Tensor *input, *prev_dA;

    if(prev->flatten) {
        input = prev->flatten->a;
        prev_dA = prev->flatten->dA;
    }
    else if(prev->conv) {
        input = prev->conv->a;
        prev_dA = prev->conv->dA;
    }
    else if(prev->pooling) {
        input = prev->pooling->a;
        prev_dA = prev->pooling->dA;
    }
    else if(prev->dense) {
        input = prev->dense->a;
        prev_dA = prev->dense->dA;
    }
    else if(prev->input) {
        input = prev->input->a;
        prev_dA = prev->input->dA;
    }
    else {
        std::cerr<<"Error in dense_bw"<<std::endl;
        return;
    }

    ll input_size = input->filter * input->height * input->width * input->depth;

    for(ll i=0;i<dense->units;i++) {
        for(ll j=0;j<input_size;j++) {
            // Calculate the gradient for the weights
            dense->dW->arr[i][0][0][j] += dense->dA->arr[0][0][0][i] * 
                activation_fn_dx(new Tensor(dense->y->arr[0][0][0][i]), dense->activation) * 
                input->arr[0][0][0][j];

            // Calculate the gradient for the previous layer
            prev_dA->arr[0][0][0][j] += dense->dA->arr[0][0][0][i] * 
                activation_fn_dx(new Tensor(dense->y->arr[0][0][0][i]), dense->activation) *
                dense->w->arr[i][0][0][j];
        }
        // Calculate the gradient for the bias
        dense->dB->arr[0][0][0][i] += dense->dA->arr[0][0][0][i] * 
            activation_fn_dx(new Tensor(dense->y->arr[0][0][0][i]), dense->activation);
    }
}

//////////////////////// Model Implementation ////////////////////////

void Model::forward(Tensor *input) {
    // Set the input layer
    if(layers[0]->input) {
        for(ll i=0;i<layers[0]->input->a->filter;i++) {
            for(ll j=0;j<layers[0]->input->a->depth;j++) {
                for(ll h=0;h<layers[0]->input->a->height;h++) {
                    for(ll w=0;w<layers[0]->input->a->width;w++) {
                        layers[0]->input->a->arr[i][j][h][w] = input->arr[i][j][h][w];
                    }
                }
            }
        }
    } else {
        std::cerr<<"Error: First layer must be an input layer."<<std::endl;
        exit(1);
    }

    // Forward pass through the layers
    for(ll i=1;i<sz(layers);i++) {
        if(layers[i]->conv) convolution_fw(layers[i-1], layers[i]->conv);
        else if(layers[i]->pooling) pooling_fw(layers[i-1], layers[i]->pooling);
        else if(layers[i]->flatten) flatten_fw(layers[i-1], layers[i]->flatten);
        else if(layers[i]->dense) dense_fw(layers[i-1], layers[i]->dense);
        else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }

    // // Print all parameters for debugging
    // for(ll i=0;i<sz(layers);i++) {
    //     if(layers[i]->conv) {
    //         Conv *conv = layers[i]->conv;
    //         std::cout<<"Conv Layer "<<i<<": Kernel: ";
    //         conv->kernel->print();
    //         std::cout<<"Conv Layer "<<i<<": Bias: ";
    //         conv->b->print();
    //         std::cout<<"Conv Layer "<<i<<": Output: ";
    //         conv->a->print();
    //     } else if(layers[i]->pooling) {
    //         Pooling *pool = layers[i]->pooling;
    //         std::cout<<"Pooling Layer "<<i<<": Output: ";
    //         pool->a->print();
    //     } else if(layers[i]->flatten) {
    //         Flatten *flatten = layers[i]->flatten;
    //         std::cout<<"Flatten Layer "<<i<<": Output: ";
    //         flatten->a->print();
    //     } else if(layers[i]->dense) {
    //         Dense *dense = layers[i]->dense;
    //         std::cout<<"Dense Layer "<<i<<": Weights: ";
    //         dense->w->print();
    //         std::cout<<"Dense Layer "<<i<<": Bias: ";
    //         dense->b->print();
    //         std::cout<<"Dense Layer "<<i<<": Output: ";
    //         dense->a->print();
    //     } else if(layers[i]->input) {
    //         Input *input_layer = layers[i]->input;
    //         std::cout<<"Input Layer "<<i<<": Output: ";
    //         input_layer->a->print();
    //     } else {
    //         std::cerr<<"Error: Unknown layer type."<<std::endl;
    //         exit(1);
    //     }
    // }
}

void Model::backward() {
    // Copy the loss gradient to the last layer's dA
    if(layers[sz(layers)-1]->dense) {
        Dense *last_dense = layers[sz(layers)-1]->dense;
        for(ll i=0;i<last_dense->units;i++) {
            last_dense->dA->arr[0][0][0][i] = loss->grad->arr[0][0][0][i];
        }
    } else {
        std::cerr<<"Error: Last layer must be a dense layer for backward propagation."<<std::endl;
        exit(1);
    }
    
    // Backward pass through the layers
    for(ll i=sz(layers)-1;i>0;i--) {
        if(layers[i]->dense) {
            layers[i]->dense->batch_cnt++; // Increment batch count for averaging gradients
            dense_bw(layers[i-1], layers[i]->dense);
        }
        else if(layers[i]->flatten) flatten_bw(layers[i-1], layers[i]->flatten);
        else if(layers[i]->pooling) pooling_bw(layers[i-1], layers[i]->pooling);
        else if(layers[i]->conv) {
            layers[i]->conv->batch_cnt++; // Increment batch count for averaging gradients
            convolution_bw(layers[i-1], layers[i]->conv);
        }
        else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }

    // // Print gradients for debugging
    // for(ll i=0;i<sz(layers);i++) {
    //     if(layers[i]->conv) {
    //         Conv *conv = layers[i]->conv;
    //         std::cout<<"Conv Layer "<<i<<": dW: ";
    //         conv->dW->print();
    //         std::cout<<"dB: ";
    //         conv->dB->print();
    //         std::cout<<"dA: ";
    //         conv->dA->print();
    //     } else if(layers[i]->pooling) {
    //         Pooling *pool = layers[i]->pooling;
    //         std::cout<<"Pooling Layer "<<i<<": dA: ";
    //         pool->dA->print();
    //     } else if(layers[i]->flatten) {
    //         Flatten *flatten = layers[i]->flatten;
    //         std::cout<<"Flatten Layer "<<i<<": dA: ";
    //         flatten->dA->print();
    //     } else if(layers[i]->dense) {
    //         Dense *dense = layers[i]->dense;
    //         std::cout<<"Dense Layer "<<i<<": dW: ";
    //         dense->dW->print();
    //         std::cout<<"dB: ";
    //         dense->dB->print();
    //         std::cout<<"dA: ";
    //         dense->dA->print();
    //     } else if(layers[i]->input) {
    //         Input *input_layer = layers[i]->input;
    //         std::cout<<"Input Layer "<<i<<": dA: ";
    //         input_layer->dA->print();
    //     } else {
    //         std::cerr<<"Error: Unknown layer type."<<std::endl;
    //         exit(1);
    //     }
    // }
}

void Model::gradient_descent() {
    // Normalize the gradients by the batch size
    for(ll i=1;i<sz(layers);i++) {
        if(layers[i]->conv) {
            Conv *conv = layers[i]->conv;

            for(ll f=0;f<conv->filter;f++) {
                for(ll d=0;d<conv->kernel->depth;d++) {
                    for(ll h=0;h<conv->kernel->height;h++) {
                        for(ll w=0;w<conv->kernel->width;w++) {
                            conv->dW->arr[f][d][h][w] /= (db) conv->batch_cnt;
                        }
                    }
                }

                conv->dB->arr[f][0][0][0] /= (db) conv->batch_cnt;
            }

            conv->batch_cnt = 0; // Reset batch count after normalization
        } else if(layers[i]->pooling) {
            // Pooling layers do not have weights to update
        } else if(layers[i]->flatten) {
            // Flatten layers do not have weights to update
        } else if(layers[i]->dense) {
            Dense *dense = layers[i]->dense;

            for(ll j=0;j<dense->units;j++) {
                for(ll k=0;k<dense->w->width;k++) {
                    dense->dW->arr[j][0][0][k] /= (db) dense->batch_cnt;
                }

                dense->dB->arr[0][0][0][j] /= (db) dense->batch_cnt;
            }

            dense->batch_cnt = 0; // Reset batch count after normalization
        } else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }

    // Apply gradient descent using the optimizer
    for(ll i=1;i<sz(layers);i++) {
        if(layers[i]->conv) {
            Conv *conv = layers[i]->conv;
            conv->timestep++;

            for(ll f=0;f<conv->filter;f++) {
                for(ll d=0;d<conv->kernel->depth;d++) {
                    for(ll h=0;h<conv->kernel->height;h++) {
                        for(ll w=0;w<conv->kernel->width;w++) {
                            // Apply optimizer for kernel weights
                            optimizer->apply_gradient(conv->timestep, f, d, h, w, conv->m_w, conv->v_w, conv->kernel, conv->dW);
                        }
                    }
                }
                // Apply optimizer for bias
                optimizer->apply_gradient(conv->timestep, f, 0, 0, 0, conv->m_b, conv->v_b, conv->b, conv->dB);
            }
        } else if(layers[i]->pooling) {
            // Pooling layers do not have weights to update
        } else if(layers[i]->flatten) {
            // Flatten layers do not have weights to update
        } else if(layers[i]->dense) {
            Dense *dense = layers[i]->dense;
            dense->timestep++;

            for(ll j=0;j<dense->units;j++) {
                for(ll k=0;k<dense->w->width;k++) {
                    // Apply optimizer for weights
                    optimizer->apply_gradient(dense->timestep, j, 0, 0, k, dense->m_w, dense->v_w, dense->w, dense->dW);
                }

                // Apply optimizer for bias
                optimizer->apply_gradient(dense->timestep, 0, 0, 0, j, dense->m_b, dense->v_b, dense->b, dense->dB);
            }
        } else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }
}

void Model::reset_forward() {
    for(ll i=1;i<sz(layers);i++) {
        if(layers[i]->conv) {
            // Reset a and y for convolution layers
            layers[i]->conv->y->fill(0.0);
            layers[i]->conv->a->fill(0.0);
        } else if(layers[i]->pooling) {
            // Reset a for pooling layers
            layers[i]->pooling->a->fill(0.0);
        } else if(layers[i]->flatten) {
            // Reset a for flatten layers
            layers[i]->flatten->a->fill(0.0);
        } else if(layers[i]->dense) {
            // Reset a and y for dense layers
            layers[i]->dense->y->fill(0.0);
            layers[i]->dense->a->fill(0.0);
        } else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }
}

void Model::reset_backward_per_batch() {
    for(ll i=0;i<sz(layers);i++) {
        if(layers[i]->conv) {
            // Reset dA, dW, and dB for convolution layers
            layers[i]->conv->dA->fill(0.0);
            layers[i]->conv->dW->fill(0.0);
            layers[i]->conv->dB->fill(0.0);
        } else if(layers[i]->pooling) {
            // Reset dA for pooling layers
            layers[i]->pooling->dA->fill(0.0);
        } else if(layers[i]->flatten) {
            // Reset dA for flatten layers
            layers[i]->flatten->dA->fill(0.0);
        } else if(layers[i]->dense) {
            // Reset dA, dW, and dB for dense layers
            layers[i]->dense->dA->fill(0.0);
            layers[i]->dense->dW->fill(0.0);
            layers[i]->dense->dB->fill(0.0);
        } else if(layers[i]->input) {
            // Reset dA for input layer
            layers[i]->input->dA->fill(0.0);
        } else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }
}

void Model::reset_backward_per_datapoint() {
    // Reset dA for all layers
    for(ll i=0;i<sz(layers);i++) {
        if(layers[i]->conv) layers[i]->conv->dA->fill(0.0);
        else if(layers[i]->pooling) layers[i]->pooling->dA->fill(0.0);
        else if(layers[i]->flatten) layers[i]->flatten->dA->fill(0.0);
        else if(layers[i]->dense) layers[i]->dense->dA->fill(0.0);
        else if(layers[i]->input) layers[i]->input->dA->fill(0.0);
        else {
            std::cerr<<"Error: Unknown layer type."<<std::endl;
            exit(1);
        }
    }
}

//////////////////////// Other operations ////////////////////////

void Model::save_model_to_file(const std::str filename) {
    std::ofstream file(filename);
    if(!file.is_open()) {
        std::cerr<<"Error: Could not open file "<<filename<<" for writing."<<std::endl;
        return;
    }

    // Write the number of layers
    file<<sz(layers)<<std::endl;

    // For each layer, write the layer type and its parameters
    for(ll i=0;i<sz(layers);i++) {
        if(layers[i]->input) {
            Input *input=layers[i]->input;
            file<<"input"<<std::endl;
            file<<input->a->filter<<" "<<input->a->depth<<" "<<input->a->height<<" "<<input->a->width<<std::endl;
        }
        else if(layers[i]->conv) {
            Conv *conv=layers[i]->conv;
            file<<"conv"<<std::endl;
            file<<conv->filter<<" "<<conv->kernel->depth<<" "<<conv->kernel->height<<" "<<conv->kernel->width<<std::endl;
            file<<conv->stride_h<<" "<<conv->stride_w<<std::endl;
            file<<conv->pad_h<<" "<<conv->pad_w<<std::endl;
            file<<conv->activation<<std::endl;
            
            // Flatten and save kernel weights
            for(ll f=0;f<conv->filter;f++) {
                for(ll d=0;d<conv->kernel->depth;d++) {
                    for(ll h=0;h<conv->kernel->height;h++) {
                        for(ll w=0;w<conv->kernel->width;w++) {
                            file<<conv->kernel->arr[f][d][h][w]<<" ";
                        }
                    }
                }
            }
            file<<std::endl;
            
            // Flatten and save biases
            for(ll f=0;f<conv->filter;f++) {
                file<<conv->b->arr[f][0][0][0]<<" ";
            }
            file<<std::endl;
        }
        else if(layers[i]->pooling) {
            Pooling *pool=layers[i]->pooling;
            file<<"pooling"<<std::endl;
            file<<pool->type<<std::endl;
            file<<pool->pool_h<<" "<<pool->pool_w<<std::endl;
            file<<pool->stride_h<<" "<<pool->stride_w<<std::endl;
            file<<pool->pad_h<<" "<<pool->pad_w<<std::endl;
        }
        else if(layers[i]->flatten) {
            file<<"flatten"<<std::endl;
        }
        else if(layers[i]->dense) {
            Dense *dense=layers[i]->dense;
            file<<"dense"<<std::endl;
            file<<dense->units<<" "<<dense->w->width<<std::endl;
            file<<dense->activation<<std::endl;
            
            // Flatten and save weights
            for(ll j=0;j<dense->units;j++) {
                for(ll k=0;k<dense->w->width;k++) {
                    file<<dense->w->arr[j][0][0][k]<<" ";
                }
            }
            file<<std::endl;
            
            // Flatten and save biases
            for(ll j=0;j<dense->units;j++) file<<dense->b->arr[0][0][0][j]<<" ";
            file<<std::endl;
        }
    }

    file.close();
}

bool Model::load_model_from_file(const std::str filename) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        std::cerr<<"Error: Could not open file "<<filename<<" for reading."<<std::endl;
        return 0;
    }

    ll num_layers;
    file>>num_layers;

    if(num_layers <= 0 || sz(layers) > num_layers) {
        std::cerr<<"Error: Invalid number of layers in the model file."<<std::endl;
        exit(1);
    }

    for(ll i=0;i<num_layers;i++) {
        std::str layer_type;
        file>>layer_type;

        if(layer_type == "input") {
            ll filter, depth, height, width;
            file>>filter>>depth>>height>>width;
            
            if(layers[i]->input) {
                Input *input = layers[i]->input;
                if(filter == input->a->filter && depth == input->a->depth && height == input->a->height && width == input->a->width) {
                    // Okey, just read the input tensor
                } else {
                    std::cerr<<"Error: Input layer mismatch."<<std::endl;
                    exit(1);
                }
            } else {
                std::cerr<<"Error: Input layer mismatch."<<std::endl;
                exit(1);
            }
        }
        else if(layer_type == "conv") {
            if(layers[i]->conv) {
                Conv *conv = layers[i]->conv;
                ll filter, depth, height, width, stride_h, stride_w, pad_h, pad_w;
                std::str activation;
                
                file>>filter>>depth>>height>>width;
                file>>stride_h>>stride_w;
                file>>pad_h>>pad_w;
                file>>activation;

                if(filter == conv->filter && depth == conv->kernel->depth && height == conv->kernel->height && width == conv->kernel->width) {
                    for(ll f=0;f<filter;f++) {
                        for(ll d=0;d<depth;d++) {
                            for(ll h=0;h<height;h++) {
                                for(ll w=0;w<width;w++) file>>conv->kernel->arr[f][d][h][w];
                            }
                        }
                    }

                    for(ll f=0;f<filter;f++) file>>conv->b->arr[f][0][0][0];
                } else {
                    std::cerr<<"Error: Convolution layer mismatch."<<std::endl;
                    exit(1);
                }
            } else {
                std::cerr<<"Error: Convolution layer mismatch."<<std::endl;
                exit(1);
            }
        }
        else if(layer_type == "pooling") {
            if(layers[i]->pooling) {
                Pooling *pool = layers[i]->pooling;
                
                std::str type;
                file>>type;
                
                if(type != pool->type) {
                    std::cerr<<"Error: Pooling layer type mismatch."<<std::endl;
                    exit(1);
                }
                
                ll pool_h, pool_w, stride_h, stride_w, pad_h, pad_w;
                file>>pool_h>>pool_w;
                file>>stride_h>>stride_w;
                file>>pad_h>>pad_w;

                if(pool_h == pool->pool_h && pool_w == pool->pool_w && stride_h == pool->stride_h && stride_w == pool->stride_w && pad_h == pool->pad_h && pad_w == pool->pad_w) {
                    // No parameters to load for pooling layer
                } else {
                    std::cerr<<"Error: Pooling layer parameters mismatch."<<std::endl;
                    exit(1);
                }
            } else {
                std::cerr<<"Error: Pooling layer mismatch."<<std::endl;
                exit(1);
            }
        }
        else if(layer_type == "flatten") {
            if(layers[i]->flatten) {
                Flatten *flatten = layers[i]->flatten;
                // No parameters to load for flatten layer
            } else {
                std::cerr<<"Error: Flatten layer mismatch."<<std::endl;
                exit(1);
            }
        }
        else if(layer_type == "dense") {
            if(layers[i]->dense) {
                Dense *dense = layers[i]->dense;
                ll units, width;
                std::str activation;
                file>>units>>width;
                file>>activation;

                if(units == dense->units && width == dense->w->width) {
                    for(ll j=0;j<units;j++) {
                        for(ll k=0;k<width;k++) file>>dense->w->arr[j][0][0][k];
                    }

                    for(ll j=0;j<units;j++) file>>dense->b->arr[0][0][0][j];
                } else {
                    std::cerr<<"Error: Dense layer mismatch."<<std::endl;
                    exit(1);
                }
            } else {
                std::cerr<<"Error: Dense layer mismatch."<<std::endl;
                exit(1);
            }
        } else {
            std::cerr<<"Error: Unknown layer type "<<layer_type<<std::endl;
            exit(1);
        }
    }

    file.close();

    return 1;
}