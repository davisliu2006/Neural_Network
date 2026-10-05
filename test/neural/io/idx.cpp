#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../../test.hpp"
#include "../../../neural/io/idx.hpp"

using namespace std;
using namespace lin_alg;
using namespace neural;

static void generate_input(const string& images_path, const string& labels_path) {
    // 2 images, 2x2 pixels, labels 3 and 7
    {
        ofstream out(images_path, ios::binary);
        idx::write_u32(out, 2051);
        idx::write_u32(out, 2);
        idx::write_u32(out, 2);
        idx::write_u32(out, 2);
        unsigned char pixels[] = {
            0, 255, 0, 255,
            128, 0, 64, 192
        };
        out.write(reinterpret_cast<char*>(pixels), sizeof(pixels));
    }
    {
        ofstream out(labels_path, ios::binary);
        idx::write_u32(out, 2049);
        idx::write_u32(out, 2);
        unsigned char labels[] = {3, 7};
        out.write(reinterpret_cast<char*>(labels), sizeof(labels));
    }
}

TestBatch tests({
    {"load mnist idx", []() {
        string images_path = "test/bin/test_images.idx";
        string labels_path = "test/bin/test_labels.idx";
        generate_input(images_path, labels_path);

        auto examples = idx::load_mnist<double>(images_path, labels_path);
        assert(examples.size() == 2);
        assert(examples[0].input.size() == 4);
        assert(examples[0].output.size() == 10);
        assert(examples[0].input[0] == 0.0);
        assert(examples[0].input[1] == 1.0);
        assert(examples[0].output[3] == 1.0);
        assert(examples[0].output[0] == 0.0);
        assert(examples[1].output[7] == 1.0);
        assert(examples[1].input[0] == 128.0/255.0);
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(15);

    tests.run();

    return 0;
}
