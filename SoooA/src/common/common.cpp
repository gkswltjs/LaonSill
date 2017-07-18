/**
 * @file common.cpp
 * @date 2017-07-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"

using namespace std;

const char* SOOOA_BUILD_PATH_ENVNAME = "SOOOA_BUILD_PATH";
extern const char* SOOOA_HOME_ENVNAME;

string Common::GetSoooARelPath(string path) {
    if (getenv(SOOOA_BUILD_PATH_ENVNAME) != NULL) {
        return string(getenv(SOOOA_BUILD_PATH_ENVNAME)) + "/src/" + path;
    } else if (getenv(SOOOA_HOME_ENVNAME) != NULL) {
        return string(getenv(SOOOA_HOME_ENVNAME)) + "/" + path;
    } else {
        return path;
    }
}
