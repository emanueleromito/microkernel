#include "microkernel/dtype.h"
#include <stdexcept>

size_t element_size(DType dtype) {

    switch(dtype) {

        case DType::Float32 : return 4;
        case DType::Float16 : return 2;
        case DType::Int8 : return 1;
        case DType::Int4 : return 1;
        default : throw std::invalid_argument("Unknown DType."); 
    }

}


std::string dtype_to_string(DType dtype) {

    switch(dtype) {

        case DType::Float32 : return "Float32";
        case DType::Float16 : return "Float16";
        case DType::Int8 : return "Int8";
        case DType::Int4 : return "Int4";
        default : throw std::invalid_argument("Unknown DType."); 
    }

}


