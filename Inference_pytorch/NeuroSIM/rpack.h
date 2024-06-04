#ifndef RPACK_H_
#define RPACK_H_
#include <vector>
#include <string>

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);

string getDateStringFromArg(string fullArg);

class Rect {
    public:
        int bin;
        int posBottomLeftX;
        int posBottomLeftY;
        int sizeX;
        int sizeY;
        int rid;
        Rect(vector<double> rectLine);
};

class rpackInfo {
    public:
        vector<Rect> rectList;
        vector<vector<double>> nodeExecorder;
        vector<vector<double>> ridExecorder;
        double binCount;
        rpackInfo(string lastArg);
};

#endif