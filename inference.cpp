#include "include/nn/nn.h"
#include "include/lodepng/lodepng.h"
using namespace std;

#define ll int
#define str string
#define db double
#define sz(s) ((ll) s.size())
#define pii pair<ll, ll>
#define fi first
#define se second

int main() {
    // Load image
    Tensor *input_image = new Tensor("test.png");

    // Create the model
	Model *model = new Model();
	model->read_model_config("model.conf");
	// model->summary();
	model->load_model_from_file("best.nn");

    model->reset_forward();
    model->forward(input_image);

    cout<<"Output: "<<argmax(model->layers.back()->dense->a)<<endl;
}