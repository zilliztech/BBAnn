#include <iostream>
#include <random>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>
#include "util/distance.h"
using namespace std;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec *1000000+ tv.tv_usec ;
}

template<typename T1, typename T2, typename R>
R get_l2_gt(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0, dif;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 7: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 6: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 5: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 4: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 3: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 2: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 1: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
            }
    }
    return dis;
}

template<typename T1, typename T2, typename R>
R get_ip_gt(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dis+=(R)a[i]*(R)b[i]; i++;
                case 7: dis+=(R)a[i]*(R)b[i]; i++;
                case 6: dis+=(R)a[i]*(R)b[i]; i++;
                case 5: dis+=(R)a[i]*(R)b[i]; i++;
                case 4: dis+=(R)a[i]*(R)b[i]; i++;
                case 3: dis+=(R)a[i]*(R)b[i]; i++;
                case 2: dis+=(R)a[i]*(R)b[i]; i++;
                case 1: dis+=(R)a[i]*(R)b[i]; i++;
            }
    }
    return dis;
}

void test_float_float_l2() {
	int d = 96;      // dimension
    int nb = 10000; // database size
    int nq = 10000;  // nb of queries
	//init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }
    //gt :
    double t0 = elapsed() ;
   //double result = floatgtfunction (xb, xq, d, nb);
   double gt = 0.0;
    for (int i=0; i<nb; i++) {
    	gt +=  get_l2_gt<float, float, float>(xb + i*d, xq + i*d, d);
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;
    //test l2 float to float :
    double t1 = elapsed() ;
    double fresult = 0.0;
    for (int i=0; i<nb; i++) {
    	fresult +=  L2sqr<float, float, float> (xb + i*d, xq + i*d, d);
    }
    cout <<"avx2  time "<< elapsed() - t1<<endl;
    cout<<"gt result"<<gt<<" avx2 result"<<fresult<<endl;
    delete [] xb;
    delete [] xq;
}

void test_int8_float_l2() {
	int d = 100;      // dimension
    int nb =  256*10000; // database size
    int nq =  256*10000;  // nb of queries
	//init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));

    int8_t* xb = new int8_t[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = int8_t(rand()%256-100);
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }
    cout<<"cal gt"<<endl;
    //gt :
    double t0 = elapsed();
    float result =  0.0;
    for (int i=0; i<nb; i++) {
    	result += get_l2_gt<int8_t, float, float>(xb + i*d, xq + i*d, d);
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;
    //test l2 float to float :
    double t1 = elapsed() ;
    float fresult = 0.0;
     float* fb = new float[d];
    for (int i=0; i<nb; i++) {
    	 fresult +=  L2sqr<int8_t, float, float> (xb + i*d, xq + i*d, d);
   }
    cout <<"avx2  time "<< elapsed() - t1<<endl;
    cout<<"gt result"<<result<<" avx2 result"<<fresult<<endl;
    delete [] fb;
    delete [] xb;
    delete [] xq;
}

void test_int8_int8_l2() {
    cout<< "int8 testing"<<endl;
	int d = 100;      // dimension
    int nb =  10000; // database size
    int nq =  10000;  // nb of queries

    int8_t* xb = new int8_t[d * nb];
    int8_t* xq = new int8_t[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = int8_t(rand()%256-100);
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = int8_t(rand()%256-22);
    }
    //gt :
    double t0 = elapsed() ;
    int gt = 0;
    for(int i=0;i<nb;i++) {
    	gt+= get_l2_gt<int8_t, int8_t, int>(xb, xq, d);
        }
    cout<< "normal search time"<<elapsed()-t0<<endl;
    //test l2 float to float :
    double t1 = elapsed() ;
    int result =0;
    for (int i=0; i<nb; i++) {
    	result += L2sqr<int8_t, int8_t, int>(xb, xq, d);
       }
    cout <<"avx2  time "<< elapsed() - t1<<endl;
    cout<<"gt result"<<gt<<" avx2 result"<<result<<endl;
    delete [] xb;
    delete [] xq;
}

void test_uint8_float_l2() {
	int d = 128;      // dimension
    int nb =  10000; // database size
    int nq =  10000;  // nb of queries
	//init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));

    uint8_t* xb = new uint8_t[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = uint8_t(rand()%256);
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }
    //gt :
    double t0 = elapsed();
    float result =  0.0;
    for (int i=0; i<nb; i++) {
    	result += get_l2_gt<uint8_t, float, float>(xb, xq + i*d, d);
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;
    //test l2 float to float :
    double t1 = elapsed() ;
    float fresult = 0.0;
     float* fb = new float[d];
    for (int i=0; i<nb; i++) {
    	 fresult +=  L2sqr<uint8_t, float, float> (xb, xq + i*d, d);
   }
    cout <<"avx2  time "<< elapsed() - t1<<endl;
    cout<<"gt result"<<result<<" avx2 result"<<fresult<<endl;
    delete [] fb;
    delete [] xb;
    delete [] xq;

}

void test_uint8_uint8_l2() {
	int d = 128;      // dimension
    int nb =  10000; // database size
    int nq =  10000;  // nb of queries
	//init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));

    uint8_t* xb = new uint8_t[d * nb];
    uint8_t* xq = new uint8_t[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = uint8_t(rand()%256);
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = uint8_t(rand()%256);
    }
    //gt :
    double t0 = elapsed() ;
    int gt = 0;
    for(int i=0;i<nb;i++) {
    	gt+= get_l2_gt<uint8_t, uint8_t, int>(xb, xq, d)/nq;
      }
    cout<< "normal search time"<<elapsed()-t0<<endl;
    //test l2 float to float :
    double t1 = elapsed() ;
    int result =0;
    for (int i=0; i<nb; i++) {
    	result +=  L2sqr<uint8_t, uint8_t, int>(xb, xq, d)/(nq);
     }
    cout <<"avx2  time "<< elapsed() - t1<<endl;
    cout<<"gt result"<<gt<<" avx2 result"<<result<<endl;
    delete [] xb;
    delete [] xq;

}

void test_float_float_ip() {
	int d = 200;      // dimension
    int nb = 10000; // database size
    int nq = 10000;  // nb of queries
	//init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }
    //gt :
    double t0 = elapsed() ;
   //double result = floatgtfunction (xb, xq, d, nb);
   double gt = 0.0;
    for (int i=0; i<nb; i++) {
    	gt +=  get_ip_gt<float, float, float>(xb + i*d, xq + i*d, d);
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;
    //test l2 float to float :
    double t1 = elapsed() ;
    double fresult = 0.0;
    for (int i=0; i<nb; i++) {
    	fresult +=  IP<float, float, float> (xb + i*d, xq + i*d, d);
    }
    cout <<"avx2  time "<< elapsed() - t1<<endl;
    cout<<"gt result"<<gt<<" avx2 result"<<fresult<<endl;
    delete [] xb;
    delete [] xq;
}

void test_uint8_compute_lookuptable_ip() {
    size_t d = 8;             // dimension
    int m = 32;
    int nb =  256;      // database size
    int nq =  10000;  // nb of queries
    //init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));
    //init:
    float* xb = new float[d*nb*m];
    uint8_t* xq = new uint8_t[d*nq*m];
    cout<<"testing lookuptable"<<endl;
    for (int i = 0; i < nq*m; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = uint8_t(rand()%256);
    }

    for (int i = 0; i < nb*m; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    double t0 = elapsed();
    float average = 0.0;
    float* result = new float[nb*m];
    float temp ;

    for (int i=0; i<nq; i++) {
        for (int j=0;j<m;j++) {
            for(int k=0;k<nb;k++) {
                result[j*256+k] = IP<uint8_t, float, float>(xq+i*j*d, xb+(j*256+k)*d, d);
            }
        }
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;

    for(int i=0;i<nb*m;i++) {
        average += result[i];
    }
    average = average/(1.0*nb*m);
    delete [] result;

    //using compute_looktable
    float* ytransform = new float[nb*m*d];
    float* result2 = new float[nb*m];

    for (int i = 0; i<m; i++) {
        int begin = i*nb*d;
        for (int j=0; j<nb;j++) {
            for(int k=0;k<d;k++) {
                ytransform[begin+k*nb+j] = xb[begin+d*j+k];
            }
        }
    }

    cout<<"test gt"<<endl;
    t0 = elapsed();
    for (int i=0;i<nq;i++)
        for(int j=0;j<m;j++) {
            compute_lookuptable_IP<uint8_t>(xq+i*j*d, ytransform+j*256*d,result2+j*256,d,256);
        }
    cout<< "avx2 search time"<<elapsed()-t0<<endl;
    float average2 = 0.0;
    for(int i=0;i<nb*m;i++) {
        average2 += result2[i];
    }
    average2 = average2/(1.0*nb*m);
    cout<<"the average ip reuslt:"<<endl;
    cout<<"gt"<<average<<" "<<average2<<endl;

    delete [] xb;
    delete [] xq;
    delete [] ytransform;
    delete [] result2;



}

void test_int8_compute_lookuptable_ip() {
    size_t d = 8;             // dimension
    int m = 32;
    int nb =  256;      // database size
    int nq =  10000;  // nb of queries
    //init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));
    //init:
    float* xb = new float[d * nb*m];
    int8_t* xq = new int8_t[d * nq*m];

    for (int i = 0; i < nq*m; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = int8_t(rand()%256-100);
    }

    for (int i = 0; i < nb*m; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    double t0 = elapsed();
    float average = 0.0;
    float* result = new float[nb*m];
    float temp ;
    cout<<"test gt"<<endl;
    for (int i=0; i<nq; i++) {
        for (int j=0;j<m;j++) {
            for(int k=0;k<nb;k++) {
                result[j*256+k] = IP<int8_t, float, float>(xq+i*j*d, xb+(j*256+k)*d, d);
            }
        }
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;

    for(int i=0;i<nb*m;i++) {
        average += result[i];
    }
    average = average/(1.0*nb*m);
    delete [] result;

    //using compute_looktable
    float* ytransform = new float[nb*m*d];
    float* result2 = new float[nb*m];
    for (int i = 0; i<m; i++) {
        int begin = i*nb*d;
        for (int j=0; j<nb;j++) {
            for(int k=0;k<d;k++) {
                ytransform[begin+k*nb+j] = xb[begin+d*j+k];
            }
        }
    }

    t0 = elapsed();
    for (int i=0;i<nq;i++)
        for(int j=0;j<m;j++) {
            compute_lookuptable_IP<int8_t>(xq+i*j*d, ytransform+j*256*d,result2+j*256,d,256);
        }
    cout<<"avx2 search time"<<elapsed()-t0<<endl;
    float average2 = 0.0;
    for(int i=0;i<nb*m;i++) {
        average2 += result2[i];
    }
    average2 = average2/(1.0*nb*m);
    cout<<"the average ip reuslt:"<<endl;
    cout<<"gt"<<average<<" "<<average2<<endl;

    delete [] xb;
    delete [] xq;

    delete [] ytransform;
    delete [] result2;

}

void test_float_compute_lookuptable_ip() {

    int d = 4;             // dimension
    int m = 32;
    int nb =  256;      // database size
    int nq =  10000;  // nb of queries
    //init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));
    //init:
    float* xb = new float[d * nb*m];
    float* xq = new float[d * nq*m];

    for (int i = 0; i < nq*m; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
    }

    for (int i = 0; i < nb*m; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    double t0 = elapsed();
    float average = 0.0;
    float* result = new float[nb*m];
    float temp ;
    cout<<"test gt"<<endl;
    for (int i=0; i<nq; i++) {
        for (int j=0;j<m;j++) {
            for(int k=0;k<nb;k++) {
                result[j*256+k] = get_ip_gt<float, float, float>(xq+i*j*d, xb+(j*256+k)*d, d);
            }
        }
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;

    for(int i=0;i<nb*m;i++) {
        average += result[i];
    }
    average = average/(1.0*nb*m);
    delete [] result ;

    //using compute_looktable
    float* ytransform = new float[nb*m*d];
    float* result2 = new float[nb*m];
    for (int i = 0; i<m; i++) {
        int begin = i*nb*d;
        for (int j=0; j<nb;j++) {
            for(int k=0;k<d;k++) {
                ytransform[begin+k*nb+j] = xb[begin+d*j+k];
            }
        }
    }


    t0 = elapsed();
    for (int i=0;i<nq;i++)
        for(int j=0;j<m;j++) {
            compute_lookuptable_IP<float>(xq+i*j*d, ytransform+j*256*d,result2+j*256,d,256);
        }
    cout<< "avx2 search time"<<elapsed()-t0<<endl;
    float average2 = 0.0;
    for(int i=0;i<nb*m;i++) {
        average2 += result2[i];
    }
    average2 = average2/(1.0*nb*m);
    cout<<"the average ip result:"<<endl;
    cout<<"gt"<<average<<" "<<average2<<endl;
    delete [] xb;
    delete [] xq;
    delete [] ytransform;
    delete [] result2;

}


void test_float_compute_lookuptable_l2() {

    int d = 4;             // dimension
    int m = 32;
    int nb =  256;      // database size
    int nq =  10000;  // nb of queries
    //init :
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    srand(time(0));
    //init:
    float* xb = new float[d * nb*m];
    float* xq = new float[d * nq*m];

    for (int i = 0; i < nq*m; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
    }

    for (int i = 0; i < nb*m; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    double t0 = elapsed();
    float average = 0.0;
    float* result = new float[nb*m];
    float temp ;
    cout<<"test gt"<<endl;
    for (int i=0; i<nq; i++) {
        for (int j=0;j<m;j++) {
            for(int k=0;k<nb;k++) {
                result[j*256+k] = get_l2_gt<float, float, float>(xq+i*j*d, xb+(j*256+k)*d, d);
            }
        }
    }
    cout<< "normal search time"<<elapsed()-t0<<endl;

    for(int i=0;i<nb*m;i++) {
        average += result[i];
    }
    average = average/(1.0*nb*m);
    delete [] result ;

    //using compute_looktable
    float* ytransform = new float[nb*m*d];
    float* result2 = new float[nb*m];
    for (int i = 0; i<m; i++) {
        int begin = i*nb*d;
        for (int j=0; j<nb;j++) {
            for(int k=0;k<d;k++) {
                ytransform[begin+k*nb+j] = xb[begin+d*j+k];
            }
        }
    }


    t0 = elapsed();
    for (int i=0;i<nq;i++)
        for(int j=0;j<m;j++) {
            compute_lookuptable_L2<float>(xq+i*j*d, ytransform+j*256*d,result2+j*256,d,256);
        }
    cout<< "avx2 search time"<<elapsed()-t0<<endl;
    float average2 = 0.0;
    for(int i=0;i<nb*m;i++) {
        average2 += result2[i];
    }
    average2 = average2/(1.0*nb*m);
    cout<<"the average ip result:"<<endl;
    cout<<"gt"<<average<<" "<<average2<<endl;
    delete [] xb;
    delete [] xq;
    delete [] ytransform;
    delete [] result2;

}

void base_test() {

    cout<<endl<<"testing l2sqrt<float, float, float>:"<<endl;
    test_float_float_l2();
    cout<<endl<<"testing l2sqrt<uint8, uint8, int>:"<<endl;
    test_uint8_uint8_l2();
    cout<<endl<<"testing l2sqrt<int8, int8, int>:"<<endl;
    test_int8_int8_l2();
    cout<<endl<<"testing l2sqrt<int8, float, float>:"<<endl;
    test_int8_float_l2();
    cout<<endl<<"testing l2sqrt<uint8, float, float>:"<<endl;
    test_uint8_float_l2();
    cout<<endl<<"testing ip<float, float, float>: "<<endl;
    test_float_float_ip();
}
void matrix_test() {

    cout<<endl<<"testing compute_looktable_ip<uint8>: "<<endl;
    test_uint8_compute_lookuptable_ip();
    cout<<endl<<"testing compute_looktable_ip<int8>: "<<endl;
    test_int8_compute_lookuptable_ip();
    cout<<endl<<"testing compute_looktable_ip<float>: "<<endl;
    test_float_compute_lookuptable_ip();
    cout<<endl<<"testing compute_looktable_l2<float>: "<<endl;
    test_float_compute_lookuptable_l2();

}

int main() {

   // base_test();
    matrix_test();


}
