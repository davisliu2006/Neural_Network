#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../base.hpp"
#include "../trainable_network.hpp"

namespace neural {
    using std::string, std::vector;
    using std::ifstream, std::ofstream, std::ios;

    /*
    We read unsigned char to ensure that we do unsigned extensions.
    */
    using uchar = unsigned char;

    namespace idx {
        inline uint32_t read_u32(ifstream& in) {
            uchar b[4];
            in.read((char*)(b), 4);
            assert(in);
            return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16)
                | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
        }

        inline void write_u32(ofstream& out, uint32_t v) {
            uchar b[4] = {
                uchar((v >> 24) & 0xff),
                uchar((v >> 16) & 0xff),
                uchar((v >> 8) & 0xff),
                uchar(v & 0xff)
            };
            out.write((char*)(b), 4);
        }

        /**
         * Loads the labels file (defines the target outputs).
         */
        inline vector<uint8_t> load_labels(const string& path) {
            ifstream in(path, ios::binary);
            assert(in);
            assert(read_u32(in) == 2049); // assert is label file
            int n = int(read_u32(in));

            vector<uint8_t> labels(n);
            in.read((char*)(labels.data()), n);
            assert(in);
            return labels;
        }

        /**
         * Loads the images file.
         * Multiplies each pixel by `scale`.
         */
        template <class type = double>
        vector<vvect_t<type>> load_images(
            const string& path,
            type scale = type(1)/type(255)
        ) {
            ifstream in(path, ios::binary);
            assert(in);
            assert(read_u32(in) == 2051); // assert is image file
            int n = int(read_u32(in));
            int rows = int(read_u32(in));
            int cols = int(read_u32(in));
            int pixels = rows*cols;

            vector<vvect_t<type>> images;
            images.reserve(n);
            vector<uchar> buf(pixels);
            for (int i = 0; i < n; i++) {
                in.read((char*)(buf.data()), pixels);
                assert(in);
                vvect_t<type> img(pixels);
                for (int j = 0; j < pixels; j++) {
                    img[j] = type(buf[j])*scale;
                }
                images.push_back(img);
            }
            return images;
        }

        /**
         * Loads the MNIST dataset.
         * @param n_classes - the number of classes for the labels
         * @param scale - scale factor for pixels
         */
        template <class type = double>
        vector<TrainingExample<type>> load_mnist(
            const string& images_path,
            const string& labels_path,
            int n_classes = 10,
            type scale = type(1)/type(255)
        ) {
            double t0 = time();
            auto images = load_images<type>(images_path, scale);
            auto labels = load_labels(labels_path);
            assert(images.size() == labels.size());

            vector<TrainingExample<type>> examples;
            examples.reserve(images.size());
            for (int i = 0; i < int(images.size()); i++) {
                assert(labels[i] < n_classes);
                vvect_t<type> y(n_classes, type(0));
                y[labels[i]] = type(1);
                examples.push_back(TrainingExample<type>(images[i], y));
            }
            double t1 = time();
            std::cout << "Loaded " << examples.size() << " examples in " << t1-t0 << "s" << std::endl;
            return examples;
        }
    }
}
