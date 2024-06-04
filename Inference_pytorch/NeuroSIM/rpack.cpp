
#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "rpack.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile) {
    ifstream infile(inputfile.c_str());      
    string inputline;
    string inputval;
    
    int ROWin=0, COLin=0;      
    if (!infile.good()) {        
        cerr << "Error: the input file cannot be opened!" << endl;
        exit(1);
    }else{
        while (getline(infile, inputline, '\n')) {       
            ROWin++;                                
        }
        infile.clear();
        infile.seekg(0, ios::beg);      
        if (getline(infile, inputline, '\n')) {        
            istringstream iss (inputline);      
            while (getline(iss, inputval, ',')) {       
                COLin++;
            }
        }   
    }
    infile.clear();
    infile.seekg(0, ios::beg);          

    vector<vector<double> > netStructure;               
    for (int row=0; row<ROWin; row++) { 
        vector<double> netStructurerow;
        getline(infile, inputline, '\n');             
        istringstream iss;
        iss.str(inputline);
        for (int col=0; col<COLin; col++) {       
            while(getline(iss, inputval, ',')){ 
                istringstream fs;
                fs.str(inputval);
                double f=0;
                fs >> f;                
                netStructurerow.push_back(f);           
            }           
        }       
        netStructure.push_back(netStructurerow);
    }
    infile.close();
    
    return netStructure;
    netStructure.clear();
}   

string getDateStringFromArg(string fullArg){
    string result;
    string reference = "YYYY_MM_DD_HH_MM_SS.csv";
    string dateReference = "YYYY_MM_DD_HH_MM_SS";
    int start = fullArg.length() - reference.length();
    result = fullArg.substr(start,reference.length());
    return result;
}

Rect::Rect(vector<double> rectLine) {
    bin = rectLine[0];
    posBottomLeftX = rectLine[1];
    posBottomLeftY = rectLine[2];
    sizeX = rectLine[3];
    sizeY = rectLine[4];
    rid = rectLine[5];
}

rpackInfo::rpackInfo(string lastArg) {
    cout << "Creating rpackInfo...";
    string dateStringCsv = getDateStringFromArg(lastArg);

    string nodeExecorderDir = "./rectpack_networks/node_execorder_" + dateStringCsv;
    string ridExecDir = "./rectpack_networks/rid_execorder_" + dateStringCsv;
    string rectListDir = "./rectpack_networks/rect_list_" + dateStringCsv;
    // process the rect list            
    vector<vector<double>> rectListProto = getNetStructure(rectListDir);
    for(int i = 0 ; i < rectListProto.size() ; i++ ){
        rectList.push_back(Rect(rectListProto[i]));
    }
    nodeExecorder = getNetStructure(nodeExecorderDir);
    ridExecorder = getNetStructure(ridExecDir);

    // last bin
    binCount = rectList.back().bin;
};