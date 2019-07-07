
#ifndef  nullptr
#define nullptr 0
#endif
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h> 
#else

#include <unistd.h>

#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"


#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image_resize.h"
#include "net.h"

#include <stdint.h>
#include <algorithm>
#include "timing.h"

#include <iostream>
#include <istream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>


#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a): (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) > (b)) ? (b): (a))
#endif

typedef struct {
    std::string class_name;
    int environment_type{};
} AttrItem;

class LabelParser {
public:
    explicit LabelParser(const std::string &label_file_path);

    AttrItem getLabelById(int class_id);

private:
    void parseLabelFile(const std::string &label_file_path);

    std::unordered_map<int, AttrItem> detection_object;
};

struct Item {
    int class_id{};
    AttrItem attrItem;

    friend std::istream &operator>>(std::istream &fin, Item &item) {
        std::string line;

        if (!getline(fin, line)) {
            fin.setstate(std::ios::failbit);
            return fin;
        }
        getline(fin, line);
        getline(fin, line);

        item.class_id = stoi(line.substr(line.find_last_of(' ') + 1));
        getline(fin, line);

        auto start = line.find_first_of('\"');
        auto end = line.find_last_of('\"');

        item.attrItem.class_name = line.substr(start + 1, end - start - 1);

        getline(fin, line);
        item.attrItem.environment_type = stoi(line.substr(line.find_last_of(' ') + 1));
        getline(fin, line);

        return fin;
    }
};


LabelParser::LabelParser(const std::string &label_file_path) {
    parseLabelFile(label_file_path);
}

void LabelParser::parseLabelFile(const std::string &label_file_path) {
    std::ifstream label_file(label_file_path);

    std::transform(std::istream_iterator<Item>(label_file), {},
                   std::inserter(detection_object, detection_object.end()),
                   [](const Item &item) {
                       return std::make_pair(item.class_id, item.attrItem);
                   });
}

AttrItem LabelParser::getLabelById(int class_id) {
    return detection_object.find(class_id)->second;
}


void cropImage(unsigned char *image, int stride, int comp, float x, float y, float newWidth, float newHeight) {

    for (int i = 0; i < newHeight; ++i) {
        unsigned char *scanOut = image + (int) (i * newWidth * comp);
        const unsigned char *scanIn = image + (int) ((i + y) * stride + x * comp);
        memcpy(scanOut, scanIn, (int) (newWidth * comp));
    }
}

void centerCrop(unsigned char *image, int Width, int Height, int comp, int newWidth, int newHeight) {
    float xRatio = Width / (float) newWidth;
    float yRatio = Height / (float) newHeight;
    int stride = Width * comp;
    if (xRatio > yRatio) {
        float dx = (Width / yRatio) - newWidth;
        cropImage(image, stride, comp, -dx / 2.0f, 0.0f, newWidth + dx, newHeight);
    } else if (yRatio > xRatio) {
        float dy = (Height / xRatio) - newHeight;
        cropImage(image, stride, comp, 0.0f, -dy / 2.0f, newWidth, newHeight + dy);
    } else {
        cropImage(image, stride, comp, 0, 0, newWidth, newHeight);
    }
}


int detectPlaces365(unsigned char *rgb, int width, int height, int comp, std::vector<float> &cls_scores) {
    ncnn::Net googleNet;
    googleNet.load_param("../models/places365_fp32.param");
    googleNet.load_model("../models/places365_fp32.bin");
    int target_width = 224;
    int target_height = 224;
    centerCrop(rgb, width, height, comp, target_width, target_height);
    ncnn::Mat in = ncnn::Mat::from_pixels(rgb, ncnn::Mat::PIXEL_RGB, target_width, target_height);

    const float mean_vals[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    in.substract_mean_normalize(mean_vals, nullptr);
    ncnn::Extractor ex = googleNet.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
    out = out.reshape(out.w * out.h * out.c);
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++) {
        cls_scores[j] = out[j];
    }

    return 0;
}

int printTopk(LabelParser &labels, const std::vector<float> &cls_scores, int topk) {
    size_t size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<float, int> >());
    for (int i = 0; i < topk; i++) {
        float score = vec[i].first;
        int index = vec[i].second;
        AttrItem attrItem = labels.getLabelById(index);
        // 1 is indoor, 2 is outdoor
        if (attrItem.environment_type == 2)
            fprintf(stderr, "outdoor [%s] = %f\n", attrItem.class_name.c_str(), score);
        else
            fprintf(stderr, "indoor [%s] = %f\n", attrItem.class_name.c_str(), score);
    }
    return 0;
}


int main(int argc, char **argv) {
    if (argc < 1) {
        printf("Places365-CNNs (GoogleNet) implementation base on NCNN\n");
        printf("blog:http://cpuimage.cnblogs.com/\n");
        printf("usage: %s image_file \n ", argv[0]);
        printf("eg: %s  ../sample.jpg \n ", argv[0]);
        printf("press any key to exit. \n");
        getchar();
        return 0;
    }
    const char *filename = argv[1];
    int Width = 0;
    int Height = 0;
    int Channels = 0;
    const int target_width = 256;
    const int target_height = 256;
    const int comp = 3;
    unsigned char *input_pixels = stbi_load(filename, &Width, &Height, &Channels, comp);
    unsigned char output_pixels[target_width * target_height * comp] = {0};
    if (input_pixels == nullptr || Channels != comp) return -1;
    stbir_resize_uint8(input_pixels, Width, Height, 0,
                       output_pixels, target_width, target_height, 0, comp);
    stbi_image_free(input_pixels);
    std::vector<float> cls_scores;
    LabelParser labels("../models/categories_places365_voc.txt");
    double startTime = now();
    detectPlaces365(output_pixels, target_width, target_height, comp, cls_scores);
    double nDetectTime = calcElapsed(startTime, now());
    int topk = 5;
    printTopk(labels, cls_scores, topk);
    printf("time: %f ms.\n ", (nDetectTime * 1000));
    printf("press any key to exit. \n");
    getchar();
    return 0;
}