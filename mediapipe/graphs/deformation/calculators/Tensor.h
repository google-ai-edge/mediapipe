#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

using namespace std;
using namespace cv;

template <typename T>
class Tensor
{
private:
    vector<int> dims;
    Mat M;
    int dtype;

    void _set_dtype() {
        if (typeid(T).name() == typeid(double).name())
            dtype = CV_64F;
        else if (typeid(T).name() == typeid(int).name())
            dtype = CV_32S;
        else
            dtype = CV_64F;
    }

public:
    Tensor() {
        _set_dtype();
        dims = vector<int>{ 3, 3 };
        M = Mat(dims, dtype);
    }

    Tensor(vector<int> _dims) {
        _set_dtype();
        dims = _dims;
        M = Mat(dims, dtype);
    }

    Tensor(Mat _M) {
        dims = vector<int>(_M.size.p, _M.size.p + _M.size.dims());
        M = _M;
        dtype = _M.type();
    }

    Tensor(const Tensor<T>& _m) {
        dims = _m.dims;
        M = _m.M;
        dtype = _m.dtype;
    }

    Tensor(T** arr, int _n, int _m) {
        _set_dtype();
        dims = { _n, _m };
        M = Mat(dims, dtype);

        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _m; j++)
                M.at<T>(Vec<int, 2>(i, j)) = arr[i][j];
    }

    vector<T> get_1d_data() {
        return vector<T>(M.ptr<T>(0), M.ptr<T>(0) + dims[1]);
    }

    T at(vector<int> _indexes) {
        return M.at<T>(_indexes.data());
    }

    Tensor index(int i1) {
        return Tensor(M.row(i1));
    }


    Tensor index(vector<int> _indexes) {
        vector<int> _dims;
        _dims = dims;
        _dims[0] = 0;
        Mat _M = Mat(_dims, dtype);

        for (int i = 0; i < _indexes.size(); ++i)
            _M.push_back(M.row(_indexes[i]));

        return Tensor(_M);
    }

    Tensor index(vector<vector<int>> _indexes) {
        vector<int> _dims;
        _dims = dims;
        _dims[0] = 0;
        Mat _M = Mat(_dims, dtype);

        Mat tmp_M, _tmp_M;
        for (int i = 0; i < _indexes.size(); ++i) {
            _M.push_back(this->index(_indexes[i]).M);
        }

        return Tensor(_M.reshape(1, _indexes.size()));
    }

    Tensor index(Range r1, int index2) {
        if (r1.end < 0) {
            r1.end = M.rows + r1.end + 1;
        }
        return Tensor(M(r1, Range::all()).col(index2));
    }

    Tensor index(Range r1, Range r2) {
        if (r1.end < 0) {
            r1.end = M.rows + r1.end + 1;
        }
        if (r2.end < 0) {
            r2.end = M.cols + r2.end + 1;
        }
        if (r1.start < 0 && r1.start > -2000000000) {
            r1.start = M.rows + r1.start;
        }
        if (r2.start < 0 && r2.start > -2000000000) {
            r2.start = M.cols + r2.start;
        }
        return Tensor(M(r1, r2));
    }

    Tensor concat(Tensor t, int dim) {
        Mat dst;
        if (dim == 0) {
            vconcat(M, t.M, dst);
        }
        if (dim == 1) {
            hconcat(M, t.M, dst);
        }
        return Tensor(dst);
    }

    Tensor matmul(Tensor t) {
        return Tensor(M * t.M);
    }

    Tensor inverse() {
        return Tensor(M.inv());
    }

    Tensor transpose() {
        return Tensor(M.t());
    }

    T norm() {
        int ndims = dims.size();
        int size = 1;
        for (int i = 0; i < ndims; i++) {
            size = size * dims[i];
        }

        vector<int> _index(ndims, 0); 
        T _norm = pow(M.at<T>(_index.data()), 2);
        
        for (int i = 0; i < size - 1; i++) {
            _index[ndims - 1] += 1;
            for (int j = ndims-1; j >= 0; j--) {
                if (_index[j] >= dims[j]) {
                    _index[j] = 0;
                    _index[j - 1] += 1;
                }
            }
            _norm += pow(M.at<T>(_index.data()), 2);
        }

        return sqrt(_norm);
    }


    friend ostream& operator<<(ostream& os, const Tensor<T>& _M){
        os << _M.M;
        return os;
    }

    friend const Tensor operator-(const Tensor& t) {
        return Tensor(-t.M);
    }

    friend const Tensor operator/(const Tensor& t1, const Tensor& t2) {
        return Tensor(t1.M.mul(1 / t2.M));
    }

    friend const Tensor operator*(const Tensor& t1, const Tensor& t2) {
        return Tensor(t1.M.mul(t2.M));
    }

    friend const Tensor operator+(const Tensor& t1, const Tensor& t2) {
        return Tensor(t1.M + t2.M);
    }

    friend const Tensor operator-(const Tensor& t1, const Tensor& t2) {
        return Tensor(t1.M - t2.M);
    }

    friend const Tensor operator/(const Tensor& t1, const T& val) {
        return Tensor(t1.M / val);
    }

    friend const Tensor operator*(const Tensor& t1, const T& val) {
        return Tensor(t1.M * val);
    }

    friend const Tensor operator-(const Tensor& t1, const T& val) {
        return Tensor(t1.M - val);
    }

    friend const Tensor operator*(const T& val, const Tensor& t1) {
        return Tensor(t1.M * val);
    }

    static vector<int> sort_indexes(const vector<T>& v) {
        vector<int> idx(v.size());
        iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });
        return idx;
    }

    ~Tensor()
    {
    }
};
