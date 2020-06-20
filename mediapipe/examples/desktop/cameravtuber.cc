#include <boost/python.hpp>

std::string hello() {
    return "hello world";
}

BOOST_PYTHON_MODULE(cameravtuber) {
    using namespace boost::python;
    def("hello", &hello);
}