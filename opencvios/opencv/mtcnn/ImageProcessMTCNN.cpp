//
//  ImageProcessMTCNN.cpp
//  opencvios
//
//  Created by Eugene Golovanov on 4/30/19.
//  Copyright © 2019 Eugene Golovanov. All rights reserved.
//

#include "ImageProcessMTCNN.hpp"
#include "H5Cpp.h"

ImageProcessMTCNN::ImageProcessMTCNN(std::string path, std::string path1)
{
    this->path = path;
    std::cout<<"printing path: "<<path<<std::endl;
    
    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = path + "det1.caffemodel";
    pConfig.protoText = path + "det1.prototxt";
    pConfig.threshold = 0.6f;
    
    RefineNetwork::Config rConfig;
    rConfig.caffeModel = path + "det2.caffemodel";
    rConfig.protoText = path + "det2.prototxt";
    rConfig.threshold = 0.7f;
    
    OutputNetwork::Config oConfig;
    oConfig.caffeModel = path + "det3.caffemodel";
    oConfig.protoText = path + "det3.prototxt";
    oConfig.threshold = 0.7f;
    
    
    
    
    #define MAX_NAME_LENGTH 32
    const std::string FileName(path1 +"SimpleCompound.h5");
    const std::string DatasetName("PersonalInformation");
    const std::string member_age("Age");
    const std::string member_sex("Sex");
    const std::string member_name("Name");
    const std::string member_height("Height");

    typedef struct {
        int age;
        char sex;
        char name[MAX_NAME_LENGTH];
        float height;
    } PersonalInformation;

    // Data to write
    PersonalInformation person_list[] = {
        { 18, 'M', "Mary",  152.0   },
        { 32, 'F', "Tom",   178.6   },
        { 29, 'M', "Tarou", 166.6   }
    };
    // the length of the data
    int length = sizeof(person_list) / sizeof(PersonalInformation);
    // the array of each length of multidimentional data.
    hsize_t dim[1];
    dim[0] = sizeof(person_list) / sizeof(PersonalInformation);

    // the length of dim
    int rank = sizeof(dim) / sizeof(hsize_t);

    // defining the datatype to pass HDF55
    H5::CompType mtype(sizeof(PersonalInformation));
    mtype.insertMember(member_age, HOFFSET(PersonalInformation, age), H5::PredType::NATIVE_INT);
    mtype.insertMember(member_sex, HOFFSET(PersonalInformation, sex), H5::PredType::C_S1);
    mtype.insertMember(member_name, HOFFSET(PersonalInformation, name), H5::StrType(H5::PredType::C_S1, MAX_NAME_LENGTH));
    mtype.insertMember(member_height, HOFFSET(PersonalInformation, height), H5::PredType::NATIVE_FLOAT);
    
    // preparation of a dataset and a file.
    H5::DataSpace space(rank, dim);
    H5::H5File *file = new H5::H5File(FileName, H5F_ACC_TRUNC);
    H5::DataSet *dataset = new H5::DataSet(file->createDataSet(DatasetName, mtype, space));
    // Write
    dataset->write(person_list, mtype);
    
    delete dataset;
    delete file;

    
    
    
    
    this->detector = MTCNNDetector(pConfig, rConfig, oConfig);
}


using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;
static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data)
{
    cv::Mat outImg;
    img.convertTo(outImg, CV_8UC3);
    
    for (auto &d : data) {
        cv::rectangle(outImg, d.first, cv::Scalar(0, 255, 255), 2);
        auto pts = d.second;
        for (size_t i = 0; i < pts.size(); ++i) {
            cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
        }
    }
    return outImg;
}


cv::Mat ImageProcessMTCNN::filterMTCNN(cv::Mat src)
{
    if (this->path == "") {
        return src;
    }
    
    std::vector<Face> faces;
    
    {
        faces = this->detector.detect(src, 20.f, 0.709f);
    }
    
    std::cout << "Number of faces found in the supplied image - " << faces.size()
    << std::endl;
    
    std::vector<rectPoints> data;
    
    // show the image with faces in it
    for (size_t i = 0; i < faces.size(); ++i) {
        std::vector<cv::Point> pts;
        for (int p = 0; p < NUM_PTS; ++p) {
            pts.push_back(
                          cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
        }
        
        auto rect = faces[i].bbox.getRect();
        auto d = std::make_pair(rect, pts);
        data.push_back(d);
    }
    
    auto resultImg = drawRectsAndPoints(src, data);
    
    return resultImg;
}
