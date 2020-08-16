//
//  ImageProcessMTCNN.cpp
//  opencvios
//
//  Created by Eugene Golovanov on 4/30/19.
//  Copyright © 2019 Eugene Golovanov. All rights reserved.
//

#include "ImageProcessMTCNN.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////  HDF5 START  ////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include "H5Cpp.h"
using namespace H5;
// Operator function
extern "C" herr_t file_info(hid_t loc_id, const char *name, const H5L_info2_t *linfo,
    void *opdata);
std::vector<std::string> groupNames;

using std::cout;
using std::endl;
//////////////////////////////////////////////////////////////////////////////////////  HDF5 END  ////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



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
    
    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////  HDF5 START  ////////////////////////////////////////////////////////////////////////////////////////////////////////

    const H5std_string FILE_NAME( path + "dataset_targarien.h5" );

    // Try block to detect exceptions raised by any of the calls inside it
    try
    {
        /*
         * Turn off the auto-printing when failure occurs so that we can
         * handle the errors appropriately
         */
        Exception::dontPrint();

        /*
         * Create the named file, truncating the existing one if any,
         * using default create and access property lists.
         */
        H5File *file = new H5File( FILE_NAME, H5F_ACC_RDONLY );


        /*
         * Use iterator to see the names of the objects in the file
         * root directory.
         */
        cout << endl << "Iterating over elements in the file" << endl;
        herr_t idx = H5Literate2(file->getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);
        cout << endl;


        cout << "Extracted group Names: \n";
        for (int i=0; i<groupNames.size();i++ )
        {
            cout << "\n\n  extracted name: " << groupNames[i] << endl;

            H5std_string groupName( groupNames[i] );

            Group* group = new Group(file->openGroup(groupName));

            DataSet* dataset;
            try {  // to determine if the dataset exists in the group
                 dataset = new DataSet( group->openDataSet( "embedding" ));
            }
            catch( GroupIException not_found_error ) {
                cout << "\t ERROR: Dataset is not found." << endl;
//                return 0;
            }
            
            // Read Embed
            double  embedding[512]; /* output buffer */
            dataset->read(embedding, PredType::NATIVE_DOUBLE);

            // Loop through 10 example values
            for( unsigned int a = 0; a < 10; a = a + 1 )
            {
                cout << embedding[a] << ", ";
            }
          
        }

        /*
         * Close the group and file.
         */
        delete file;
    }  // end of try block

    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printErrorStack();
//        return -1;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printErrorStack();
//        return -1;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printErrorStack();
//        return -1;
    }

    // catch failure caused by the Attribute operations
    catch( AttributeIException error )
    {
        error.printErrorStack();
//        return -1;
    }

/////////////////////////////////////////////////////  HDF5 END  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    
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




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////  HDF5 START  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Operator function.
 */
herr_t
file_info(hid_t loc_id, const char *name, const H5L_info2_t *linfo, void *opdata)
{
    hid_t group;

    /*
     * Open the group using its name.
     */
    group = H5Gopen2(loc_id, name, H5P_DEFAULT);

    groupNames.push_back(name);

    /*
     * Display group name.
     */
    cout << "Name : " << name << endl;

    H5Gclose(group);
    return 0;
}
/////////////////////////////////////////////////////  HDF5 END  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
