#include <bits/stdc++.h>
using namespace std;

using cd = complex<double>;
const double PI = acos(-1);

// a: polynomial domain
// y: fft domain

void fft(vector<cd>& a) {

    const int n = a.size();
    if (n == 1) {
        return;
    }

    vector<cd> a0(n/2), a1(n/2);
    for (int i = 0; 2*i < n; ++i) {
        a0[i] = a[2*i];
        a1[i] = a[2*i+1];
    }

    fft(a0);
    fft(a1);

    cd w(1);
    cd wn(cos(2*PI/n),sin(2*PI/n));
    for (int i = 0; 2*i < n; ++i) {
        a[i] = a0[i] + w * a1[i];
        a[i+n/2] = a0[i] - w * a1[i];
        w *= wn;
    }

}

void ifft(vector<cd>& y) {
    const int n = y.size();
    if (n == 1) {
        return;
    }

    vector<cd> y0(n/2), y1(n/2);
    for (int i = 0; 2*i < n; ++i) {
        y0[i] = y[2*i];
        y1[i] = y[2*i+1];
    }

    ifft(y0);
    ifft(y1);

    cd w(1);
    cd wn(cos(-2*PI/n),sin(-2*PI/n));
    for (int i = 0; 2*i < n; ++i) {
        y[i] = y0[i] + w * y1[i];
        y[i+n/2] = y0[i] - w * y1[i];
        y[i] /= 2;
        y[i + n/2] /= 2;  
        w *= wn;
    }
}


int32_t main() {

    const int N = 128;

    vector<cd> vals(N);
    
    double xmin = -4.0;
    double xmax = 4.0;
    double step = (xmax - xmin) / N;

    int idx = 0;
    for (double x = xmin; x <= xmax; x += step) {
        double y = 0.0;
        if (x >= -1.0 && x <= 1.0) {
            vals[idx] = 1.0;
        }
        ++idx;
    }

    // fft(vals);
    // ifft(vals);

    /*
    for (int i = 0; i < N; ++i) {
        printf("%f ", vals[i].real());
    }
    printf("\n");
    */


    


    return 0;
}