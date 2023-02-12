//
//  Image.hpp
//  DCP_purecpp
//
//  Created by Dario Vazquez on 2/9/23.
//

#ifndef Image_hpp
#define Image_hpp

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <cassert>

class Image
{
private:
    int     _rows = 0;
    int     _cols = 0;
    int     _channels = 0;
    int     _size = 0;
    float*  _data = nullptr;
    bool    _malloc = false;
    
    
    
public:
    Image(int rows, int cols, int channels, float* data)
    {
        _rows = rows;
        _cols = cols;
        _channels = channels;
        _data = data;
        _malloc = false;
    }
    
    Image(int rows, int cols, int channels)
    {
        _rows = rows;
        _cols = cols;
        _channels = channels;
        _size = _channels*_rows*_cols;
        _data = new float [_size];
        _malloc = true;
    }
    
    Image(const Image &obj)
    {
        _rows       = obj.rows();
        _cols       = obj.cols();
        _channels   = obj.chan();
        _data       = &obj(0,0,0);
        _malloc     = obj.alloc();
    }
    
    ~Image()
    {
//        // Release malloc'ed data
//        if(_malloc)
//        {
//            delete[] _data;
//            _data = nullptr;
//            _malloc = false;
//        }
    }
    
    int         &rows()       { return _rows;       }
    const int   &rows() const { return _rows;       }
    int         &cols()       { return _cols;       }
    const int   &cols() const { return _cols;       }
    int         &chan()       { return _channels;   }
    const int   &chan() const { return _channels;   }
    const int   &size() const { return _size;       }
    const bool  &alloc()const { return _malloc;     }
    
    
    float& operator()(int z, int x, int y) const {
        return _data[z*_rows*_cols + x*_cols + y];
    }
    
    // Access data linearly
    float& operator()(int x) const {
        return _data[x];
    }
//    
//    const Image operator*(const Image B)
//    {
//        assert(this->rows() == B.rows());
//        assert(this->cols() == B.cols());
//        assert(this->chan() == B.chan());
//        
//        float * data = new float [this->size()];
//        
//        for(int i = 0; i < this->size(); i++)
//        {
//            data[i] = (*this)(i) * B(i);
//        }
//        
//        return Image(this->rows(), this->cols(), this->chan(), data);
//    }
//    
//    Image operator+(Image B)
//    {
//        assert(this->rows() == B.rows());
//        assert(this->cols() == B.cols());
//        assert(this->chan() == B.chan());
//        
//        float * data = new float [this->size()];
//        
//        for(int i = 0; i < this->size(); i++)
//        {
//            data[i] = (*this)(i) + B(i);
//        }
//        
//        return Image(this->rows(), this->cols(), this->chan(), data);
//    }
//    
//    Image operator+(float val)
//    {
//        float * data = new float [this->size()];
//        
//        for(int i = 0; i < this->size(); i++)
//        {
//            data[i] = (*this)(i) + val;
//        }
//        
//        return Image(this->rows(), this->cols(), this->chan(), data);
//    }
//    
//    const Image operator-(const Image B)
//    {
//        assert(this->rows() == B.rows());
//        assert(this->cols() == B.cols());
//        assert(this->chan() == B.chan());
//        
//        float * data = new float [this->size()];
//        
//        for(int i = 0; i < this->size(); i++)
//        {
//            data[i] = (*this)(i) - B(i);
//        }
//        
//        return Image(this->rows(), this->cols(), this->chan(), data);
//    }
//    
//    const Image operator/(const Image B)
//    {
//        assert(this->rows() == B.rows());
//        assert(this->cols() == B.cols());
//        assert(this->chan() == B.chan());
//        
//        float * data = new float [this->size()];
//        
//        for(int i = 0; i < this->size(); i++)
//        {
//            data[i] = B(i) / (*this)(i) ;
//        }
//        
//        return Image(this->rows(), this->cols(), this->chan(), data);
//    }
    
    void fill(float val)
    {
        std::fill_n(_data,_rows*_cols*_channels,val);
    }
    
    void copy(Image B)
    {
        std::copy(this->_data, this->_data + this->_size, &B(0,0,0));
    }
    
    
};

#endif /* Image_hpp */
